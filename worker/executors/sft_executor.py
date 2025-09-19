#!/usr/bin/env python3
"""Supervised fine-tuning (SFT) executor using TRL's SFTTrainer."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

logger = logging.getLogger("worker.sft")


class SFTExecutor:
    """Execute supervised fine-tuning jobs with TRL's SFTTrainer."""

    def __init__(self) -> None:
        self._model_name: Optional[str] = None

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        spec = (task or {}).get("spec") or {}
        start_time = time.time()
        training_successful = False
        error_msg: Optional[str] = None

        training_cfg = spec.get("training", {}) or {}
        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            model_cfg = spec.get("model", {}) or {}
            model_source = model_cfg.get("source", {}) or {}
            self._model_name = model_source.get("identifier", "gpt2")

            logger.info("Loading tokenizer and model for SFT: %s", self._model_name)
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(self._model_name)
            model.config.use_cache = False  # better compatibility with gradient checkpointing

            if bool(training_cfg.get("gradient_checkpointing", True)):
                model.gradient_checkpointing_enable()

            train_dataset, text_field = self._prepare_dataset(spec)
            logger.info("Loaded training dataset with %d rows", len(train_dataset))

            training_args = TrainingArguments(
                output_dir=str(checkpoint_dir),
                num_train_epochs=float(training_cfg.get("num_train_epochs", 1.0)),
                per_device_train_batch_size=int(training_cfg.get("batch_size", 2)),
                gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
                learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
                warmup_steps=int(training_cfg.get("warmup_steps", 0)),
                logging_steps=int(training_cfg.get("logging_steps", 10)),
                save_steps=int(training_cfg.get("save_steps", 100)),
                save_strategy=training_cfg.get("save_strategy", "steps"),
                report_to=[],  # disable default wandb reporting
                fp16=bool(training_cfg.get("fp16", False)),
                bf16=bool(training_cfg.get("bf16", False)),
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                dataset_text_field=text_field,
                packing=bool(training_cfg.get("packing", False)),
                max_seq_length=int(training_cfg.get("max_seq_length", 1024)),
            )

            logger.info("Starting supervised fine-tuning run")
            trainer.train()
            training_successful = True
            logger.info("SFT training completed")

            if bool(training_cfg.get("save_model", True)):
                model_path = out_dir / "final_model"
                trainer.save_model(model_path)
                tokenizer.save_pretrained(model_path)
                logger.info("Saved fine-tuned model to %s", model_path)

        except Exception as exc:  # pylint: disable=broad-except
            error_msg = str(exc)
            training_successful = False
            logger.exception("SFT training failed: %s", exc)

        training_time = time.time() - start_time

        result = {
            "training_successful": training_successful,
            "training_time_seconds": training_time,
            "error_message": error_msg,
            "model_name": self._model_name,
            "dataset_size": len(train_dataset) if "train_dataset" in locals() else 0,
            "output_dir": str(out_dir),
        }

        return result

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
