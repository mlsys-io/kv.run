#!/usr/bin/env python3
"""SFT executor with PyTorch FSDP support via TRL's SFTTrainer/SFTConfig.
   - No DeepSpeed.
   - Multi-GPU handled by torch.distributed (torchrun) or Accelerate.
"""

from __future__ import annotations

import logging
import os
import time
import json
import subprocess
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

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        spec = (task or {}).get("spec") or {}
        start_time = time.time()
        training_successful = False
        error_msg: Optional[str] = None
        caught_exc: Optional[Exception] = None

        training_cfg = spec.get("training", {}) or {}
        self._configure_devices(training_cfg)  # 只控制可见卡，不做模型放置
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

        # Internal torchrun launcher: spawn multi-GPU training as distributed subprocesses
        try:
            allow_multi = bool(training_cfg.get("allow_multi_gpu", False))
            already_spawned = os.environ.get("TORCHRUN_SPAWNED") == "1"
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

            if allow_multi and not already_spawned and (n_gpus or 0) > 1:
                # Persist the task spec to a file for the launcher entrypoint
                launcher_dir = out_dir / "_launcher"
                launcher_dir.mkdir(parents=True, exist_ok=True)
                task_file = launcher_dir / "task_spec.json"
                with task_file.open("w") as fh:
                    json.dump(task, fh)

                nproc = int(training_cfg.get("nproc_per_node", n_gpus))
                cmd = [
                    "torchrun",
                    "--nproc_per_node", str(nproc),
                    "-m", "executors.sft_dist_entry",
                    str(task_file),
                    str(out_dir),
                ]
                env = os.environ.copy()
                env["TORCHRUN_SPAWNED"] = "1"
                logger.info("Spawning torchrun for SFT: %s", " ".join(cmd))
                subprocess.check_call(cmd, env=env)
                # Read back results written by distributed run
                resp_path = out_dir / "responses.json"
                if resp_path.exists():
                    return self.load_json(resp_path)
                # If not present, continue to error handling
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

            train_dataset, text_field = self._prepare_dataset(spec)
            logger.info("Loaded training dataset with %d rows", len(train_dataset))

            # === FSDP 配置读取 ===
            fsdp = training_cfg.get("fsdp")  # e.g., "full_shard auto_wrap"
            fsdp_config = dict(training_cfg.get("fsdp_config") or {})
            # Sanitize mutually exclusive options to avoid HF error:
            # "`min_num_params` and `transformer_layer_cls_to_wrap` are mutually exclusive."
            try:
                if (
                    fsdp_config.get("min_num_params") is not None
                    and fsdp_config.get("transformer_layer_cls_to_wrap")
                ):
                    removed = fsdp_config.pop("min_num_params", None)
                    logger.warning(
                        "FSDP config: removed min_num_params=%s because transformer_layer_cls_to_wrap is set;"
                        " these options are mutually exclusive in Transformers.",
                        removed,
                    )
            except Exception:
                pass

            # === SFTConfig：唯一真源控制 batch 与累积 ===
            # If user provided a top-level min_num_params/fsdp_min_num_params, honor it
            # unless transformer_layer_cls_to_wrap is set (mutually exclusive).
            fsdp_min_num_params = training_cfg.get("fsdp_min_num_params", training_cfg.get("min_num_params"))
            if fsdp_config.get("transformer_layer_cls_to_wrap"):
                if fsdp_min_num_params is not None:
                    logger.warning(
                        "Dropping fsdp_min_num_params=%s because transformer_layer_cls_to_wrap is set (mutually exclusive)",
                        fsdp_min_num_params,
                    )
                # Set to 0 instead of None to satisfy Transformers check `> 0`
                fsdp_min_num_params = 0

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
                report_to=[],  # 关闭默认 wandb
                fp16=bool(training_cfg.get("fp16", False)),
                bf16=bool(training_cfg.get("bf16", False)),
                dataset_text_field=text_field,
                max_length=int(training_cfg.get("max_seq_length", 1024)),
                packing=bool(training_cfg.get("packing", False)),
                gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
                pad_token=tokenizer.pad_token,
                eos_token=tokenizer.eos_token,
                # FSDP 关键项
                fsdp=fsdp,
                fsdp_config=fsdp_config,
                fsdp_min_num_params=fsdp_min_num_params,
            )

            # 仅在单卡时手动放置；多卡/FSDP/DDP 交给分布式后端
            is_multi_gpu = bool(training_cfg.get("allow_multi_gpu", False)) and torch.cuda.device_count() > 1
            if torch.cuda.is_available() and not is_multi_gpu:
                target_device = torch.device("cuda:0")
                model = model.to(target_device)
                if any(p.device != target_device for _, p in model.named_parameters()):
                    logger.warning("Some parameters not moved to %s; check environment.", target_device)
            else:
                logger.info("Distributed mode detected — device placement handled by DDP/FSDP runtime.")

            # 构造 Trainer（兼容 TRL 版本差异）
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

            # Dry-run dataloader shapes（本地微批）
            try:
                dl = trainer.get_train_dataloader()
                first_batch = next(iter(dl))
                ids = first_batch["input_ids"]; mask = first_batch["attention_mask"]
                logger.info("Dry-run local batch -> input_ids=%s, attention_mask=%s",
                            tuple(ids.shape), tuple(mask.shape))
            except Exception as e:
                logger.warning("Dry-run dataloader check failed (non-fatal): %s", e)

            logger.info(
                "Effective batch -> per_device=%d, grad_accum=%d. FSDP='%s'",
                sft_config.per_device_train_batch_size,
                sft_config.gradient_accumulation_steps,
                fsdp,
            )

            logger.info("Starting SFT (FSDP)")
            trainer.train()
            training_successful = True
            logger.info("Training finished")

            final_model_path: Optional[Path] = None
            if bool(training_cfg.get("save_model", True)):
                model_path = out_dir / "final_model"
                trainer.save_model(model_path)
                tokenizer.save_pretrained(model_path)
                final_model_path = model_path
                logger.info("Saved fine-tuned model to %s", model_path)

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
                archive_path = archive_model_dir(final_model_path)
                result_payload["final_model_archive"] = archive_path.name
                result_payload["final_model_archive_path"] = str(archive_path)
                logger.info("Prepared model archive at %s", archive_path)
                self._upload_model_archive(task, archive_path, result_payload)

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
                response = requests.post(upload_url, files=files, headers=destination["headers"],
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
        # 默认限制单卡（若多卡可见且未显式允许）
        try:
            n = torch.cuda.device_count()
        except Exception:
            n = 0
        if n > 1:
            preferred = training_cfg.get("primary_gpu", 0)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(preferred)
            logger.info("Multiple GPUs detected (%d); restrict to device %s (set allow_multi_gpu=true to opt-in).", n, preferred)
