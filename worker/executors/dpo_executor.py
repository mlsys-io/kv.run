#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Executor using TRL's simple approach

Simple implementation using TRL's DPOTrainer with built-in train() method.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

from .base_executor import Executor, ExecutionError

logger = logging.getLogger("worker.dpo")


class DPOExecutor(Executor):
    """DPO training executor using TRL library."""

    def __init__(self):
        super().__init__()
        self._model_name = None

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        logger.info("Starting DPO training task")
        spec = (task or {}).get("spec") or {}
        start_time = time.time()
        training_config = spec.get("training", {}) or {}
        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get model configuration
            model_config = spec.get("model", {})
            model_source = model_config.get("source", {})
            self._model_name = model_source.get("identifier", "microsoft/DialoGPT-small")

            logger.info("Loading model and tokenizer...")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model and reference model
            model = AutoModelForCausalLM.from_pretrained(self._model_name)
            ref_model = AutoModelForCausalLM.from_pretrained(self._model_name)

            logger.info("Models loaded: %s", self._model_name)

            # Load dataset
            logger.info("Loading dataset...")
            dataset = self._load_dataset(spec)
            logger.info("Dataset loaded with %d samples", len(dataset))

            logger.info("Creating DPOConfig...")
            dpo_config = DPOConfig(
                learning_rate=float(training_config.get("learning_rate", 5e-7)),
                per_device_train_batch_size=int(training_config.get("batch_size", 4)),
                gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 1)),
                num_train_epochs=int(training_config.get("num_train_epochs", 1)),
                output_dir=str(checkpoint_dir),
                save_steps=int(training_config.get("save_freq", 500)),
                logging_steps=10,
            )
            logger.info("DPOConfig created successfully")

            # Initialize DPO trainer
            logger.info("Creating DPOTrainer...")
            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=dpo_config,
                train_dataset=dataset,
                tokenizer=tokenizer,
            )
            logger.info("DPOTrainer created successfully")

            # Simple training - just call train()
            logger.info("Starting DPO training...")
            dpo_trainer.train()
            logger.info("DPO training completed")

            training_successful = True
            error_msg = None

            # Save model if requested
            if training_config.get("save_model", True):
                try:
                    logger.info("Saving trained model...")
                    model_save_path = checkpoint_dir / "final_model"
                    dpo_trainer.save_model(model_save_path)
                    logger.info("Model saved to: %s", model_save_path)
                except Exception as exc:  # pragma: no cover - best effort save
                    logger.warning("Failed to save model: %s", exc)

        except Exception as exc:
            training_successful = False
            error_msg = str(exc)
            logger.exception("DPO training failed: %s", exc)
            training_time = time.time() - start_time
            results = {
                "training_successful": training_successful,
                "training_time_seconds": training_time,
                "error_message": error_msg,
                "model_name": self._model_name,
                "dataset_size": len(dataset) if "dataset" in locals() else 0,
                "output_dir": str(out_dir),
            }
            self.save_json(out_dir / "responses.json", results)
            raise ExecutionError(error_msg or "DPO training failed") from exc

        training_time = time.time() - start_time

        results = {
            "training_successful": training_successful,
            "training_time_seconds": training_time,
            "error_message": error_msg,
            "model_name": self._model_name,
            "dataset_size": len(dataset),
            "output_dir": str(out_dir),
        }
        self.save_json(out_dir / "responses.json", results)

        logger.info("DPO training task completed in %.2f seconds", training_time)
        return results

    def _load_dataset(self, spec: Dict[str, Any]) -> Dataset:
        """Load training dataset in DPO format"""
        data_config = spec.get("data", {})
        
        if "dataset_name" in data_config:
            # Load from Hugging Face datasets (like ultrafeedback_binarized)
            from datasets import load_dataset
            dataset_name = data_config["dataset_name"]
            split = data_config.get("split", "train")
            config_name = data_config.get("config_name")
            
            dataset = load_dataset(dataset_name, config_name, split=split)
            if data_config.get("max_samples"):
                dataset = dataset.select(range(min(len(dataset), data_config["max_samples"])))
                
        elif "preferences" in data_config:
            # Use provided preference data directly
            preferences = data_config["preferences"]
            dataset = Dataset.from_dict({
                "prompt": [item["prompt"] for item in preferences],
                "chosen": [item["chosen"] for item in preferences],
                "rejected": [item["rejected"] for item in preferences],
            })
        else:
            # Default demo dataset with preference pairs
            demo_data = [
                {
                    "prompt": "Write a helpful response about Python programming:",
                    "chosen": "Python is a versatile programming language known for its readability and extensive libraries. It's great for beginners and widely used in data science, web development, and automation.",
                    "rejected": "Python is just another programming language. Nothing special about it."
                },
                {
                    "prompt": "Explain machine learning in simple terms:",
                    "chosen": "Machine learning is like teaching computers to recognize patterns and make predictions from data, similar to how humans learn from experience.",
                    "rejected": "Machine learning is complicated math stuff that computers do."
                },
                {
                    "prompt": "Give advice on learning to code:",
                    "chosen": "Start with a beginner-friendly language like Python, practice regularly with small projects, and don't be afraid to make mistakes - they're part of learning!",
                    "rejected": "Just memorize syntax and you'll be fine."
                },
                {
                    "prompt": "Describe the importance of documentation:",
                    "chosen": "Good documentation is essential for code maintainability, team collaboration, and helping future developers (including yourself) understand what the code does and why.",
                    "rejected": "Documentation is just extra work that slows down development."
                }
            ]
            
            dataset = Dataset.from_dict({
                "prompt": [item["prompt"] for item in demo_data],
                "chosen": [item["chosen"] for item in demo_data],
                "rejected": [item["rejected"] for item in demo_data],
            })

        return dataset
