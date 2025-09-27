#!/usr/bin/env python3
"""Supervised fine-tuning (SFT) executor using TRL's SFTTrainer."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset, load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import entropy_from_logits

from .base_executor import Executor, ExecutionError
from .checkpoint_utils import archive_model_dir, determine_resume_path, get_http_destination

logger = logging.getLogger("worker.sft")


class _SafeSFTTrainer(SFTTrainer):
    """SFTTrainer variant with safer entropy aggregation under DataParallel."""

    @staticmethod
    def _entropy_from_mask(per_token_entropy: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(device=per_token_entropy.device, dtype=per_token_entropy.dtype)

        entropy_vals = per_token_entropy.reshape(-1)
        mask_flat = mask.reshape(-1)

        target_len = entropy_vals.size(0)
        if mask_flat.size(0) != target_len:
            if mask_flat.size(0) > target_len:
                mask_flat = mask_flat[:target_len]
            else:
                pad = torch.zeros(target_len - mask_flat.size(0), device=mask_flat.device, dtype=mask_flat.dtype)
                mask_flat = torch.cat((mask_flat, pad), dim=0)

        mask_sum = mask_flat.sum()
        if mask_sum <= 0:
            return torch.tensor(0.0, device=mask_flat.device, dtype=per_token_entropy.dtype)

        return torch.sum(entropy_vals * mask_flat) / mask_sum

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        mode = "train" if self.model.training else "eval"
        inputs["use_cache"] = False
        loss, outputs = super(SFTTrainer, self).compute_loss(  # call into base Trainer
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        try:
            if not self.args.use_liger_kernel:
                with torch.no_grad():
                    per_token_entropy = entropy_from_logits(outputs.logits)
                    if "attention_mask" in inputs:
                        attention_mask = inputs["attention_mask"]
                        if self.num_virtual_tokens:
                            virtual_mask = attention_mask.new_ones(attention_mask.size(0), self.num_virtual_tokens)
                            attention_mask = torch.cat((virtual_mask, attention_mask), dim=1)
                        entropy = self._entropy_from_mask(per_token_entropy, attention_mask)
                    elif "position_ids" in inputs:
                        entropy = torch.mean(per_token_entropy)
                    else:
                        raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
                    entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
                self._metrics[mode]["entropy"].append(entropy)

            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

            if "labels" in inputs and not self.args.use_liger_kernel:
                with torch.no_grad():
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = inputs["labels"][..., 1:].contiguous()
                    shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

                    batch_dim = shift_logits.size(0)
                    if shift_labels.size(0) != batch_dim:
                        shift_labels = shift_labels[:batch_dim]

                    seq_dim = shift_logits.size(1)
                    if shift_labels.size(1) != seq_dim:
                        seq_dim = min(seq_dim, shift_labels.size(1))
                        shift_logits = shift_logits[:, :seq_dim, :]
                        shift_labels = shift_labels[:, :seq_dim]

                    predictions = shift_logits.argmax(dim=-1)
                    mask = shift_labels != -100
                    if mask.size() != predictions.size():
                        mask = mask[: predictions.size(0), : predictions.size(1)]
                        shift_labels = shift_labels[: predictions.size(0), : predictions.size(1)]

                    correct_predictions = (predictions == shift_labels) & mask
                    total_tokens = mask.sum()
                    correct_tokens = correct_predictions.sum()
                    correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
                    total_tokens = self.accelerator.gather_for_metrics(total_tokens)
                    total_sum = total_tokens.sum()
                    accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
                    self._metrics[mode]["mean_token_accuracy"].append(accuracy)
        except Exception as exc:  # defensive fallback for metric edge cases
            logger.debug("Metric computation skipped: %s", exc)

        if return_outputs:
            return (loss, outputs)
        return loss


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
            )

            trainer = _SafeSFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )

            logger.info("Starting supervised fine-tuning run")
            trainer.train()
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
