#!/usr/bin/env python3
"""SFT executor powered by TRL's SFTTrainer/SFTConfig.

This implementation launches single-GPU runs in-process and spawns multi-GPU
tasks via ``torchrun``. DeepSpeed is used for multi-GPU orchestration when a
DeepSpeed configuration is provided in the training spec.
"""

from __future__ import annotations

import logging
import os
import time
import json
import subprocess
import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset, load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from .base_executor import Executor, ExecutionError
from .checkpoint_utils import archive_model_dir, determine_resume_path, get_http_destination

logger = logging.getLogger("worker.sft")

class SFTExecutor(Executor):
    def __init__(self) -> None:
        self._model_name: Optional[str] = None
        self._current_model: Optional[Any] = None
        self._current_trainer: Optional[Any] = None

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        spec = (task or {}).get("spec") or {}
        start_time = time.time()
        training_successful = False
        error_msg: Optional[str] = None
        caught_exc: Optional[Exception] = None

        training_cfg = spec.get("training", {}) or {}
        self._configure_devices(training_cfg)  # only constrain visible devices; no placement here
        deepspeed_cfg = self._resolve_deepspeed_config(training_cfg, logger)
        # Under torchrun/accelerate, set CUDA device from LOCAL_RANK to avoid NCCL warnings
        try:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "-1")))
                if local_rank >= 0:
                    torch.cuda.set_device(local_rank)
                    logger.info("Set CUDA device by LOCAL_RANK/RANK=%d", local_rank)
        except Exception as _e:
            logger.debug("Skipping cuda.set_device: %s", _e)
        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Internal distributed launcher: spawn multi-GPU training as subprocesses
        try:
            allow_multi = bool(training_cfg.get("allow_multi_gpu", False))
            launcher_env_flag = "KV_SFT_DISTRIBUTED"
            already_spawned = os.environ.get(launcher_env_flag) == "1"
            # Determine requested GPU count
            vis = os.environ.get("CUDA_VISIBLE_DEVICES") or training_cfg.get("visible_devices")
            n_gpus = None
            if vis:
                n_gpus = len(str(vis).split(","))
            else:
                try:
                    if torch.cuda.is_available():
                        n_gpus = torch.cuda.device_count()
                except Exception:
                    n_gpus = None

            # If a DeepSpeed config is supplied, assume multi-GPU intent
            deepspeed_intent = bool(deepspeed_cfg)

            logger.info(
                "SFT spawn decision: allow_multi=%s deepspeed_intent=%s already_spawned=%s n_gpus=%s",
                allow_multi, deepspeed_intent, already_spawned, n_gpus,
            )
            if (allow_multi or deepspeed_intent) and not already_spawned and (n_gpus or 0) > 1:
                # Persist the task spec to a file for the launcher entrypoint
                launcher_dir = out_dir / "_launcher"
                launcher_dir.mkdir(parents=True, exist_ok=True)
                task_file = launcher_dir / "task_spec.json"
                with task_file.open("w") as fh:
                    json.dump(task, fh)

                nproc = int(training_cfg.get("nproc_per_node", n_gpus))
                if deepspeed_intent:
                    cmd = [
                        "deepspeed",
                        "--num_gpus", str(nproc),
                        "--module", "worker.executors.sft_dist_entry",
                        str(task_file),
                        str(out_dir),
                    ]
                else:
                    cmd = [
                        "torchrun",
                        "--nproc_per_node", str(nproc),
                        "-m", "worker.executors.sft_dist_entry",
                        str(task_file),
                        str(out_dir),
                    ]
                env = os.environ.copy()
                env[launcher_env_flag] = "1"
                # Ensure repo root on PYTHONPATH so `worker.executors.*` is importable
                try:
                    repo_root = Path(__file__).resolve().parents[2]
                    env["PYTHONPATH"] = f"{str(repo_root)}:" + env.get("PYTHONPATH", "")
                except Exception:
                    pass
                launcher_name = "DeepSpeed" if deepspeed_intent else "torchrun"
                logger.info(
                    "Spawning %s for SFT: %s (CUDA_VISIBLE_DEVICES=%s)",
                    launcher_name,
                    " ".join(cmd),
                    env.get("CUDA_VISIBLE_DEVICES"),
                )
                subprocess.check_call(cmd, env=env)
                # Read back results written by distributed run (child ensures this exists)
                resp_path = out_dir / "responses.json"
                if resp_path.exists():
                    return self.load_json(resp_path)
                # If not present for any reason, return a minimal success envelope
                return {
                    "training_successful": True,
                    "spawned_torchrun": True,
                    "model_name": (spec.get("model", {}).get("source", {}) or {}).get("identifier"),
                    "output_dir": str(out_dir),
                }
        except Exception as spawn_exc:
            logger.exception("Failed to launch distributed SFT via torchrun: %s", spawn_exc)

        try:
            # Proceed with in-process training (single GPU or inside torchrun)
            model_cfg = spec.get("model", {}) or {}
            model_source = model_cfg.get("source", {}) or {}
            self._model_name = model_source.get("identifier", "gpt2")

            resume_path = determine_resume_path(spec, training_cfg, out_dir)
            resume_str = str(resume_path) if resume_path else None

            logger.info("Loading tokenizer and model: %s (resume=%s)", self._model_name, bool(resume_path))
            tokenizer = AutoTokenizer.from_pretrained(resume_str or self._model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if getattr(tokenizer, "padding_side", None) != "right":
                tokenizer.padding_side = "right"

            torch_dtype = None
            if bool(training_cfg.get("bf16", False)) and torch.cuda.is_available():
                torch_dtype = torch.bfloat16
            elif bool(training_cfg.get("fp16", False)) and torch.cuda.is_available():
                torch_dtype = torch.float16

            model = AutoModelForCausalLM.from_pretrained(
                resume_str or self._model_name,
                torch_dtype=torch_dtype,
            )
            model.config.use_cache = False
            if bool(training_cfg.get("gradient_checkpointing", True)):
                model.gradient_checkpointing_enable()

            self._current_model = model

            train_dataset, text_field = self._prepare_dataset(spec)
            logger.info("Loaded training dataset with %d rows", len(train_dataset))

            # Determine if distributed is actually initialized
            is_dist = False
            try:
                import torch.distributed as dist
                is_dist = dist.is_available() and dist.is_initialized()
            except Exception:
                is_dist = False

            # If DeepSpeed is requested but we are not in a distributed context, keep going.
            # Hugging Face will still initialize DeepSpeed on the current rank (often rank 0 only).
            if deepspeed_cfg and not is_dist:
                if os.environ.get("KV_SFT_DISTRIBUTED") == "1":
                    logger.info(
                        "DeepSpeed runtime will initialize torch.distributed (local_rank=%s)",
                        os.environ.get("LOCAL_RANK", "0"),
                    )
                else:
                    logger.warning(
                        "DeepSpeed config provided but torch.distributed is not initialized;"
                        " training continues on a single rank."
                    )

            # Optional DDP knobs
            ddp_kwargs = {}
            if "ddp_find_unused_parameters" in training_cfg:
                ddp_kwargs["ddp_find_unused_parameters"] = bool(training_cfg["ddp_find_unused_parameters"])

            sft_config = SFTConfig(
                output_dir=str(checkpoint_dir),
                num_train_epochs=float(training_cfg.get("num_train_epochs", 1.0)),
                per_device_train_batch_size=int(training_cfg.get("batch_size", 2)),
                gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
                learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
                warmup_steps=int(training_cfg.get("warmup_steps", 0)),
                logging_steps=int(training_cfg.get("logging_steps", 10)),
                save_steps=int(training_cfg.get("save_steps", 100)),
                save_strategy=str(training_cfg.get("save_strategy", "steps")),
                report_to=[],  # disable default wandb integration
                fp16=bool(training_cfg.get("fp16", False)),
                bf16=bool(training_cfg.get("bf16", False)),
                dataset_text_field=text_field,
                max_length=int(training_cfg.get("max_seq_length", 1024)),
                packing=bool(training_cfg.get("packing", False)),
                gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
                pad_token=tokenizer.pad_token,
                eos_token=tokenizer.eos_token,
                deepspeed=deepspeed_cfg,
                **ddp_kwargs,
            )

            # Let the distributed backend handle placement when running under torchrun/deepspeed
            if is_dist:
                logger.info("Distributed mode detected - device placement handled by DeepSpeed/DDP backend.")
            if torch.cuda.is_available() and not is_dist:
                target_device = torch.device("cuda:0")
                model = model.to(target_device)
                if any(p.device != target_device for _, p in model.named_parameters()):
                    logger.warning("Some parameters not moved to %s; check environment.", target_device)
            elif is_dist:
                # placement handled by backend
                pass

            # Construct the trainer while handling TRL signature variations
            trainer = None
            tried = []
            for variant in ("tokenizer", "processing_class", "none"):
                try:
                    if variant == "tokenizer":
                        trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_dataset, tokenizer=tokenizer)
                    elif variant == "processing_class":
                        trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_dataset, processing_class=tokenizer)
                    else:
                        trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_dataset)
                    break
                except TypeError as e:
                    tried.append(str(e))

            if trainer is None:
                raise TypeError("Failed to construct SFTTrainer. Tried variants: " + " | ".join(tried))

            self._current_trainer = trainer

            # Dry-run dataloader shapes to surface obvious padding mistakes early
            try:
                dl = trainer.get_train_dataloader()
                first_batch = next(iter(dl))
                ids = first_batch["input_ids"]; mask = first_batch["attention_mask"]
                logger.info("Dry-run local batch -> input_ids=%s, attention_mask=%s",
                            tuple(ids.shape), tuple(mask.shape))
            except Exception as e:
                logger.warning("Dry-run dataloader check failed (non-fatal): %s", e)

            logger.info(
                "Effective batch -> per_device=%d, grad_accum=%d, deepspeed=%s",
                sft_config.per_device_train_batch_size,
                sft_config.gradient_accumulation_steps,
                bool(deepspeed_cfg),
            )

            logger.info("Starting supervised fine-tuning")
            trainer.train()
            training_successful = True
            logger.info("Training finished")

            final_model_path: Optional[Path] = None
            final_archive_path: Optional[Path] = None
            if bool(training_cfg.get("save_model", True)):
                model_path = out_dir / "final_model"
                trainer.save_model(model_path)
                tokenizer.save_pretrained(model_path)
                final_model_path = model_path
                logger.info("Saved fine-tuned model to %s", model_path)
                if get_http_destination(task):
                    final_archive_path = archive_model_dir(model_path)
                    logger.info("Archived fine-tuned model to %s for HTTP delivery", final_archive_path)
                else:
                    logger.info("No HTTP destination detected; skipping archive generation")
            else:
                logger.info("save_model flag is false; skipping model serialization and archive upload")

            training_time = time.time() - start_time
            result_payload = {
                "task_id": task.get("task_id"),
                "training_successful": training_successful,
                "training_time_seconds": training_time,
                "error_message": error_msg,
                "model_name": self._model_name,
                "dataset_size": len(train_dataset),
                "output_dir": str(out_dir),
                "checkpoints_dir": str(checkpoint_dir),
                "resume_from_path": resume_str,
            }

            if final_model_path is not None:
                result_payload["final_model_path"] = str(final_model_path)
                if final_archive_path is not None:
                    result_payload["final_model_archive"] = final_archive_path.name
                    result_payload["final_model_archive_path"] = str(final_archive_path)
                    self._upload_model_archive(task, final_archive_path, result_payload)

            return result_payload

        except Exception as exc:
            error_msg = str(exc)
            training_successful = False
            caught_exc = exc
            logger.exception("SFT training failed: %s", exc)

        training_time = time.time() - start_time
        result = {
            "task_id": task.get("task_id") if isinstance(task, dict) else None,
            "training_successful": training_successful,
            "training_time_seconds": training_time,
            "error_message": error_msg,
            "model_name": self._model_name,
            "dataset_size": len(train_dataset) if "train_dataset" in locals() else 0,
            "output_dir": str(out_dir),
            "checkpoints_dir": str(checkpoint_dir),
            "resume_from_path": str(resume_path) if 'resume_path' in locals() and resume_path else None,
        }
        self.save_json(out_dir / "responses.json", result)
        if caught_exc is not None:
            raise ExecutionError(error_msg or "SFT training failed") from caught_exc
        raise ExecutionError(error_msg or "SFT training failed")

    def cleanup_after_run(self) -> None:
        self._current_trainer = None
        self._current_model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def _upload_model_archive(self, task: Dict[str, Any], archive_path: Path, payload: Dict[str, Any]) -> None:
        destination = get_http_destination(task)
        task_id = (task or {}).get("task_id")
        if not destination or not task_id:
            return
        upload_url = destination["url"].rstrip("/") + f"/{task_id}/files"
        try:
            import requests
            with archive_path.open("rb") as fh:
                files = {"file": (archive_path.name, fh, "application/zip")}
                headers = {
                    k: v
                    for k, v in (destination.get("headers") or {}).items()
                    if str(k).lower() != "content-type"
                }
                response = requests.post(upload_url, files=files, headers=headers,
                                         timeout=destination["timeout"])
                response.raise_for_status()
            download_url = destination["url"].rstrip("/") + f"/{task_id}/files/{archive_path.name}"
            payload["final_model_archive_url"] = download_url
            logger.info("Uploaded SFT model archive to %s", upload_url)
        except Exception as exc:
            logger.warning("Model archive upload failed for %s: %s", task_id, exc)

    def _prepare_dataset(self, spec: Dict[str, Any]) -> Tuple[Dataset, str]:
        data_cfg = spec.get("data", {}) or {}
        dataset_name = data_cfg.get("dataset_name")
        prompt_col = data_cfg.get("prompt_column")
        response_col = data_cfg.get("response_column")

        if dataset_name:
            split = data_cfg.get("split", "train")
            config_name = data_cfg.get("config_name")
            dataset = load_dataset(dataset_name, config_name, split=split)

            if prompt_col and response_col:
                missing = [c for c in (prompt_col, response_col) if c not in dataset.column_names]
                if missing:
                    raise ValueError(f"Dataset missing columns {missing}")
                sep = data_cfg.get("separator", "\n\n")
                def _combine(ex):
                    return {"text": f"{ex[prompt_col]}{sep}{ex[response_col]}"}
                dataset = dataset.map(_combine, remove_columns=dataset.column_names)
                text_field = "text"
            else:
                text_field = data_cfg.get("text_field", "text")
                if text_field not in dataset.column_names:
                    raise ValueError(f"Dataset missing text field '{text_field}'. Columns: {dataset.column_names}")

            max_samples = data_cfg.get("max_samples")
            if max_samples is not None:
                max_samples = int(max_samples)
                dataset = dataset.select(range(min(len(dataset), max_samples)))

            return dataset, text_field

        if "prompts" in data_cfg:
            prompts = data_cfg["prompts"]
            if not isinstance(prompts, list) or not prompts:
                raise ValueError("spec.data.prompts must be a non-empty list of strings")
            dataset = Dataset.from_dict({"text": [str(p) for p in prompts]})
            return dataset, "text"

        raise ValueError("spec.data must define dataset_name or prompts for SFT tasks")

    @staticmethod
    def _configure_devices(training_cfg: Dict[str, Any]) -> None:
        """Control CUDA_VISIBLE_DEVICES only; no model.to() here."""
        if not torch.cuda.is_available():
            return
        requested = training_cfg.get("visible_devices")
        allow_multi = bool(training_cfg.get("allow_multi_gpu", False))
        if requested:
            devices = ",".join(str(x) for x in requested) if isinstance(requested, (list, tuple)) else str(requested)
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
            logger.info("Using user-specified CUDA_VISIBLE_DEVICES=%s", devices)
            return
        if allow_multi:
            logger.info("Multi-GPU allowed; using all visible GPUs.")
            return
        # Default to a single GPU when multiple devices are visible but not explicitly allowed
        try:
            n = torch.cuda.device_count()
        except Exception:
            n = 0
        if n > 1:
            preferred = training_cfg.get("primary_gpu", 0)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(preferred)
            logger.info("Multiple GPUs detected (%d); restrict to device %s (set allow_multi_gpu=true to opt-in).", n, preferred)

    @staticmethod
    def _resolve_deepspeed_config(training_cfg: Dict[str, Any], log) -> Optional[Any]:
        cfg = training_cfg.get("deepspeed")
        if not cfg:
            return None
        if isinstance(cfg, dict):
            return cfg
        if isinstance(cfg, (str, Path)):
            candidate = Path(str(cfg)).expanduser()
            if candidate.exists():
                log.info("Using DeepSpeed config at %s", candidate)
                return str(candidate)
            # Allow literal identifiers (e.g., 'auto') without file presence.
            log.info("Using DeepSpeed literal configuration '%s'", cfg)
            return str(cfg)
        raise ValueError("training.deepspeed must be a dict, path string, or falsy")
