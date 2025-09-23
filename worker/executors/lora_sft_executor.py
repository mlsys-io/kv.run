#!/usr/bin/env python3
"""LoRA fine-tuning executor built on TRL's SFTTrainer with PEFT support."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from .base_executor import Executor, ExecutionError
from .sft_executor import _SafeSFTTrainer, SFTExecutor

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - surfaced during runtime when LoRA runs
    LoraConfig = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]

logger = logging.getLogger("worker.sft.lora")


class LoRASFTExecutor(Executor):
    """Execute LoRA-based supervised fine-tuning using TRL's SFTTrainer."""

    def __init__(self) -> None:
        self._model_name: Optional[str] = None

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        spec = (task or {}).get("spec") or {}
        start_time = time.time()
        training_successful = False
        error_msg: Optional[str] = None

        if LoraConfig is None or TaskType is None or get_peft_model is None:
            raise ExecutionError(
                "peft is required for LoRA SFT tasks. Install the 'peft' package in the worker environment."
            )

        training_cfg = spec.get("training", {}) or {}
        SFTExecutor._configure_devices(training_cfg)
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

            if bool(training_cfg.get("gradient_checkpointing", False)):
                model.gradient_checkpointing_enable()

            train_dataset, text_field = self._prepare_dataset(spec)
            logger.info("Loaded training dataset with %d rows", len(train_dataset))

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
            logger.info("LoRA adapters attached: %s", lora_target_modules)

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
            )

            trainer = _SafeSFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )

            logger.info("Starting LoRA supervised fine-tuning run")
            trainer.train()
            training_successful = True
            logger.info("LoRA SFT training completed")

            if bool(training_cfg.get("save_model", True)):
                model_path = out_dir / "final_lora"
                trainer.save_model(model_path)
                tokenizer.save_pretrained(model_path)
                logger.info("Saved LoRA-adapted weights to %s", model_path)

        except Exception as exc:  # pylint: disable=broad-except
            error_msg = str(exc)
            training_successful = False
            logger.exception("LoRA SFT training failed: %s", exc)

        training_time = time.time() - start_time

        result = {
            "training_successful": training_successful,
            "training_time_seconds": training_time,
            "error_message": error_msg,
            "model_name": self._model_name,
            "dataset_size": len(train_dataset) if "train_dataset" in locals() else 0,
            "output_dir": str(out_dir),
        }

        if training_successful:
            return result

        self.save_json(out_dir / "responses.json", result)
        message = error_msg or "LoRA SFT training failed"
        raise ExecutionError(message)

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
