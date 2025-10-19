#!/usr/bin/env python3
"""LoRA fine-tuning executor built on TRL's SFTTrainer with PEFT support."""

from __future__ import annotations

import logging
import time
import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from types import MethodType

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig
from transformers import Trainer

from .base_executor import Executor, ExecutionError
from .checkpoint_utils import archive_model_dir, determine_resume_path, get_http_destination
from .sft_executor import SFTExecutor

try:
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
except ImportError:  # pragma: no cover - surfaced during runtime when LoRA runs
    LoraConfig = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment]

logger = logging.getLogger("worker.sft.lora")


class LoRASFTExecutor(Executor):
    """Execute LoRA-based supervised fine-tuning using TRL's SFTTrainer."""

    def __init__(self) -> None:
        self._model_name: Optional[str] = None
        self._current_model: Optional[Any] = None
        self._current_trainer: Optional[Any] = None

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        spec = (task or {}).get("spec") or {}
        start_time = time.time()
        training_successful = False
        error_msg: Optional[str] = None

        if LoraConfig is None or TaskType is None or get_peft_model is None or PeftModel is None:
            raise ExecutionError(
                "peft is required for LoRA SFT tasks. Install the 'peft' package in the worker environment."
            )

        raw_training_cfg = spec.get("training", {}) or {}
        training_cfg = dict(raw_training_cfg)

        if bool(training_cfg.get("allow_multi_gpu", False)):
            logger.warning("LoRA SFT currently runs on a single GPU; ignoring allow_multi_gpu request")
            training_cfg["allow_multi_gpu"] = False

        SFTExecutor._configure_devices(training_cfg)
        if training_cfg.get("deepspeed"):
            logger.info("DeepSpeed configuration detected for LoRA run; forwarding to trainer")
        lora_cfg = spec.get("lora", {}) or {}

        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            model_cfg = spec.get("model", {}) or {}
            model_source = model_cfg.get("source", {}) or {}
            self._model_name = model_source.get("identifier", "gpt2")

            logger.info("Loading tokenizer and model for LoRA SFT: %s", self._model_name)
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(self._model_name)
            model.config.use_cache = False
            self._current_model = model

            resume_path = determine_resume_path(spec, training_cfg, out_dir)
            resume_str = str(resume_path) if resume_path else None

            if bool(training_cfg.get("gradient_checkpointing", False)):
                model.gradient_checkpointing_enable()

            train_dataset, text_field = self._prepare_dataset(spec)
            logger.info("Loaded training dataset with %d rows", len(train_dataset))

            deepspeed_config = SFTExecutor._resolve_deepspeed_config(training_cfg, logger)

            if resume_path:
                logger.info("Resuming LoRA training from %s", resume_path)
                model = PeftModel.from_pretrained(
                    model,
                    str(resume_path),
                    is_trainable=True,
                )
                logger.info("Loaded existing LoRA adapters; continuing fine-tuning")
            else:
                lora_target_modules = lora_cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]
                if not isinstance(lora_target_modules, (list, tuple)):
                    raise ValueError("lora.target_modules must be a list of module names")
                lora_target_modules = [str(mod) for mod in lora_target_modules]

                task_type_raw = str(lora_cfg.get("task_type", "CAUSAL_LM")).upper()
                try:
                    task_type = TaskType[task_type_raw]
                except KeyError as exc:
                    raise ValueError(f"Unsupported LoRA task_type '{task_type_raw}'") from exc

                peft_config = LoraConfig(
                    r=int(lora_cfg.get("r", 16)),
                    lora_alpha=int(lora_cfg.get("alpha", 32)),
                    lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                    bias=str(lora_cfg.get("bias", "none")),
                    target_modules=lora_target_modules,
                    task_type=task_type,
                    use_rslora=bool(lora_cfg.get("use_rslora", False)),
                )

                model = get_peft_model(model, peft_config)
                logger.info("Initialized new LoRA adapters: %s", lora_target_modules)

            sft_config = SFTConfig(
                output_dir=str(checkpoint_dir),
                num_train_epochs=float(training_cfg.get("num_train_epochs", 1.0)),
                per_device_train_batch_size=int(training_cfg.get("batch_size", 2)),
                gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
                learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
                warmup_steps=int(training_cfg.get("warmup_steps", 0)),
                logging_steps=int(training_cfg.get("logging_steps", 10)),
                save_steps=int(training_cfg.get("save_steps", 100)),
                save_strategy=str(training_cfg.get("save_strategy", "steps")),
                report_to=[],
                fp16=bool(training_cfg.get("fp16", False)),
                bf16=bool(training_cfg.get("bf16", False)),
                dataset_text_field=text_field,
                max_length=int(training_cfg.get("max_seq_length", 1024)),
                packing=bool(training_cfg.get("packing", False)),
                pad_token=tokenizer.pad_token,
                eos_token=tokenizer.eos_token,
                deepspeed=deepspeed_config,
            )

            # 使用原生 SFTTrainer，兼容不同版本参数(tokenizer / processing_class)
            from trl import SFTTrainer  # 局部导入避免循环
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
                    else:
                        trainer = SFTTrainer(
                            model=model,
                            args=sft_config,
                            train_dataset=train_dataset,
                        )
                    break
                except TypeError as e:
                    tried_errors.append(str(e))
                    trainer = None
                    continue
            if trainer is None:
                raise TypeError("Failed to construct SFTTrainer with tried variants: " + " | ".join(tried_errors))

            orig_compute_loss = trainer.compute_loss
            self._current_trainer = trainer

            def _compute_loss_with_guard(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                try:
                    return orig_compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
                except RuntimeError as exc:
                    if "size of tensor" not in str(exc):
                        raise
                    logger.warning(
                        "LoRA SFT entropy metric mismatch encountered; falling back to baseline loss computation: %s",
                        exc,
                    )
                    safe_inputs = dict(inputs)
                    safe_inputs["use_cache"] = False
                    loss, outputs = Trainer.compute_loss(
                        self,
                        model,
                        safe_inputs,
                        return_outputs=True,
                        num_items_in_batch=num_items_in_batch,
                    )
                    return (loss, outputs) if return_outputs else loss

            trainer.compute_loss = MethodType(_compute_loss_with_guard, trainer)

            logger.info("Starting LoRA supervised fine-tuning run")
            trainer.train()
            training_successful = True
            logger.info("LoRA SFT training completed")

            final_adapter_path: Optional[Path] = None
            if bool(training_cfg.get("save_model", True)):
                model_path = out_dir / "final_lora"
                trainer.save_model(model_path)
                tokenizer.save_pretrained(model_path)
                final_adapter_path = model_path
                logger.info("Saved LoRA-adapted weights to %s", model_path)

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
            if final_adapter_path is not None:
                result_payload["final_lora_path"] = str(final_adapter_path)
                archive_path = archive_model_dir(final_adapter_path)
                result_payload["final_lora_archive"] = archive_path.name
                result_payload["final_lora_archive_path"] = str(archive_path)
                logger.info("Prepared LoRA archive at %s", archive_path)
                self._upload_model_archive(task, archive_path, result_payload)
            return result_payload

        except Exception as exc:  # pylint: disable=broad-except
            error_msg = str(exc)
            training_successful = False
            logger.exception("LoRA SFT training failed: %s", exc)

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

        self.save_json(out_dir / "responses.json", result)
        message = error_msg or "LoRA SFT training failed"
        raise ExecutionError(message)

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
                headers = {
                    k: v
                    for k, v in (destination.get("headers") or {}).items()
                    if str(k).lower() != "content-type"
                }
                response = requests.post(
                    upload_url,
                    files=files,
                    headers=headers,
                    timeout=destination["timeout"],
                )
                response.raise_for_status()
            download_url = destination["url"].rstrip("/") + f"/{task_id}/files/{archive_path.name}"
            payload["final_lora_archive_url"] = download_url
            logger.info("Uploaded LoRA model archive to %s", upload_url)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("LoRA model archive upload failed for %s: %s", task_id, exc)

    def _prepare_dataset(self, spec: Dict[str, Any]) -> Tuple[Dataset, str]:
        data_cfg = spec.get("data", {}) or {}

        dataset_name = data_cfg.get("dataset_name")
        prompt_col = data_cfg.get("prompt_column")
        response_col = data_cfg.get("response_column")

        if dataset_name:
            split = data_cfg.get("split", "train")
            config_name = data_cfg.get("config_name")
            trust_remote_code = data_cfg.get("trust_remote_code")
            if trust_remote_code is None:
                trust_remote_code = True
            revision = data_cfg.get("revision")
            dataset = load_dataset(
                dataset_name,
                config_name,
                split=split,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )

            if prompt_col and response_col:
                missing = [col for col in (prompt_col, response_col) if col not in dataset.column_names]
                if missing:
                    raise ValueError(f"Dataset missing columns {missing}")

                separator = data_cfg.get("separator", "\n\n")

                def _combine(example: Dict[str, Any]) -> Dict[str, Any]:
                    return {"text": f"{example[prompt_col]}{separator}{example[response_col]}"}

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

        raise ValueError("spec.data must define dataset_name or prompts for LoRA SFT tasks")
