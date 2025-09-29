#!/usr/bin/env python3
"""Supervised fine-tuning (SFT) executor using TRL's SFTTrainer."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from datasets import Dataset, load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from .base_executor import Executor, ExecutionError
from .checkpoint_utils import archive_model_dir, determine_resume_path, get_http_destination

logger = logging.getLogger("worker.sft")

class SFTExecutor(Executor):
    """Execute supervised fine-tuning jobs with TRL's SFTTrainer."""

    def __init__(self) -> None:
        self._model_name: Optional[str] = None

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        spec = (task or {}).get("spec") or {}
        start_time = time.time()
        training_successful = False
        error_msg: Optional[str] = None
        caught_exc: Optional[Exception] = None

        training_cfg = spec.get("training", {}) or {}
        self._configure_devices(training_cfg)
        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            model_cfg = spec.get("model", {}) or {}
            model_source = model_cfg.get("source", {}) or {}
            self._model_name = model_source.get("identifier", "gpt2")

            resume_path = determine_resume_path(spec, training_cfg, out_dir)
            resume_str = str(resume_path) if resume_path else None

            if resume_path:
                logger.info("Resuming SFT from %s", resume_path)
            else:
                logger.info("Loading tokenizer and model for SFT: %s", self._model_name)
            tokenizer = AutoTokenizer.from_pretrained(resume_str or self._model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(resume_str or self._model_name)
            model.config.use_cache = False  # better compatibility with gradient checkpointing

            if bool(training_cfg.get("gradient_checkpointing", True)):
                model.gradient_checkpointing_enable()

            train_dataset, text_field = self._prepare_dataset(spec)
            logger.info("Loaded training dataset with %d rows", len(train_dataset))

            deepspeed_config = self._resolve_deepspeed_config(training_cfg, logger)

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
                report_to=[],  # disable default wandb reporting
                fp16=bool(training_cfg.get("fp16", False)),
                bf16=bool(training_cfg.get("bf16", False)),
                dataset_text_field=text_field,
                max_length=int(training_cfg.get("max_seq_length", 1024)),
                packing=bool(training_cfg.get("packing", False)),
                gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
                pad_token=tokenizer.pad_token,
                eos_token=tokenizer.eos_token,
                deepspeed=deepspeed_config,
            )

            # 训练前强制将模型放到期望设备（单卡时避免DataParallel设备不一致问题）
            if torch.cuda.is_available():
                target_device = torch.device("cuda:0")
                model = model.to(target_device)
                # 校验全部参数一致
                bad = [n for n,p in model.named_parameters() if p.device != target_device]
                if bad:
                    logger.warning("Parameters not on target device after .to(): %s", bad[:5])
                else:
                    logger.debug("All parameters moved to %s", target_device)

            trainer = None
            tried_errors = []
            for variant in ("tokenizer", "processing_class", "none"):
                try:
                    if variant == "tokenizer":
                        trainer = SFTTrainer(
                            model=model,
                            args=sft_config,
                            train_dataset=train_dataset,
                            tokenizer=tokenizer,
                        )
                    elif variant == "processing_class":
                        trainer = SFTTrainer(
                            model=model,
                            args=sft_config,
                            train_dataset=train_dataset,
                            processing_class=tokenizer,
                        )
                    else:  # fallback without explicit tokenizer
                        trainer = SFTTrainer(
                            model=model,
                            args=sft_config,
                            train_dataset=train_dataset,
                        )
                    break
                except TypeError as e:  # 版本不兼容重试
                    tried_errors.append(str(e))
                    trainer = None
                    continue
            if trainer is None:
                raise TypeError("Failed to construct SFTTrainer with tried variants: " + " | ".join(tried_errors))

            logger.info("Starting supervised fine-tuning run")
            need_retry = False
            try:
                trainer.train()
            except RuntimeError as rte:
                # 针对设备不匹配错误进行一次重试
                if "parameters and buffers" in str(rte) and torch.cuda.is_available():
                    logger.error("RuntimeError detected (device mismatch); attempting one retry with model.reto(cuda:0)")
                    model.to(torch.device("cuda:0"))
                    torch.cuda.empty_cache()
                    need_retry = True
                else:
                    raise
            if need_retry:
                try:
                    trainer.model = model  # 确保trainer引用最新设备模型
                    trainer.train()
                except Exception:
                    raise
            training_successful = True
            logger.info("SFT training completed")

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
                "dataset_size": len(train_dataset) if "train_dataset" in locals() else 0,
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

        except Exception as exc:  # pylint: disable=broad-except
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

        if training_successful:
            return result

        # Persist failure details for diagnostics and surface the failure upstream
        self.save_json(out_dir / "responses.json", result)
        message = error_msg or "SFT training failed"
        if caught_exc is not None:
            raise ExecutionError(message) from caught_exc
        raise ExecutionError(message)

    def _upload_model_archive(
        self,
        task: Dict[str, Any],
        archive_path: Path,
        payload: Dict[str, Any],
    ) -> None:
        destination = get_http_destination(task)
        task_id = (task or {}).get("task_id")
        if not destination or not task_id:
            return

        upload_url = destination["url"].rstrip("/") + f"/{task_id}/files"
        try:
            import requests

            with archive_path.open("rb") as fh:
                files = {"file": (archive_path.name, fh, "application/zip")}
                response = requests.post(
                    upload_url,
                    files=files,
                    headers=destination["headers"],
                    timeout=destination["timeout"],
                )
                response.raise_for_status()
            download_url = destination["url"].rstrip("/") + f"/{task_id}/files/{archive_path.name}"
            payload["final_model_archive_url"] = download_url
            logger.info("Uploaded SFT model archive to %s", upload_url)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("SFT model archive upload failed for %s: %s", task_id, exc)

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
                missing = [col for col in (prompt_col, response_col) if col not in dataset.column_names]
                if missing:
                    raise ValueError(f"Dataset missing columns {missing}")

                separator = data_cfg.get("separator", "\n\n")

                def _combine(example: Dict[str, Any]) -> Dict[str, Any]:
                    return {
                        "text": f"{example[prompt_col]}{separator}{example[response_col]}"
                    }

                dataset = dataset.map(_combine, remove_columns=dataset.column_names)
                text_field = "text"
            else:
                text_field = data_cfg.get("text_field", "text")
                if text_field not in dataset.column_names:
                    raise ValueError(
                        f"Dataset missing text field '{text_field}'. Columns: {dataset.column_names}"
                    )

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
        if not torch.cuda.is_available():
            return

        allow_multi = bool(training_cfg.get("allow_multi_gpu", False))
        requested = training_cfg.get("visible_devices")
        if requested:
            if isinstance(requested, (list, tuple)):
                devices = ",".join(str(x) for x in requested)
            else:
                devices = str(requested)
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
            logger.info("Using user-specified CUDA_VISIBLE_DEVICES=%s", devices)
            return

        if allow_multi:
            return

        preferred = training_cfg.get("primary_gpu")
        try:
            available = torch.cuda.device_count()
        except Exception:  # pragma: no cover
            available = 0

        if available <= 1:
            return

        if preferred is None:
            preferred = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(preferred)
        try:
            torch.cuda.set_device(0)
        except Exception as exc:  # pragma: no cover
            logger.debug("set_device failed after limiting GPUs: %s", exc)
        logger.info(
            "Multiple GPUs detected (%d); restrict training to device %s. Set training.allow_multi_gpu=true to opt-in.",
            available,
            preferred,
        )

    @staticmethod
    def _resolve_deepspeed_config(
        training_cfg: Dict[str, Any],
        log: Optional[logging.Logger] = None,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        ds_cfg = training_cfg.get("deepspeed")
        if not ds_cfg:
            return None

        active_logger = log or logger

        if isinstance(ds_cfg, dict):
            active_logger.info("Using inline DeepSpeed configuration")
            return ds_cfg

        if isinstance(ds_cfg, str):
            config_path = Path(ds_cfg).expanduser()
            if not config_path.is_absolute():
                config_path = (Path.cwd() / config_path).resolve()
            if not config_path.exists():
                raise ValueError(f"DeepSpeed config file '{config_path}' not found")
            active_logger.info("Using DeepSpeed configuration from %s", config_path)
            return str(config_path)

        raise ValueError("training.deepspeed must be a mapping or path string")
