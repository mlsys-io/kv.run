#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Executor using TRL's simple approach

Simplified implementation using TRL's built-in training methods.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any

from datasets import Dataset
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from .base_executor import Executor, ExecutionError

logger = logging.getLogger("worker.ppo")


class PPOExecutor(Executor):
    """PPO training executor using TRL library."""

    def __init__(self):
        super().__init__()
        self._model_name = None

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        logger.info("Starting PPO training task")
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

            # Load models - PPO requires policy model and reference model
            model = AutoModelForCausalLMWithValueHead.from_pretrained(self._model_name)
            ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(self._model_name)

            logger.info("Models loaded: %s", self._model_name)

            # Load dataset
            logger.info("Loading dataset...")
            dataset = self._load_dataset(spec)
            logger.info("Dataset loaded with %d samples", len(dataset))

            logger.info("Creating PPOConfig...")
            ppo_config = PPOConfig(
                learning_rate=float(training_config.get("learning_rate", 1.41e-5)),
                batch_size=int(training_config.get("batch_size", 4)),
                mini_batch_size=int(training_config.get("mini_batch_size", 1)),
                output_dir=str(checkpoint_dir),
            )
            logger.info("PPOConfig created successfully")

            # Initialize PPO trainer with correct API
            logger.info("Creating PPOTrainer...")
            ppo_trainer = PPOTrainer(
                args=ppo_config,
                tokenizer=tokenizer,
                model=model,
                ref_model=ref_model,
                train_dataset=dataset,
            )
            logger.info("PPOTrainer created successfully")

            # Simple training - just call train()
            logger.info("Starting PPO training...")
            ppo_trainer.train()
            logger.info("PPO training completed")

            training_successful = True
            error_msg = None

            # Save model if requested
            if training_config.get("save_model", True):
                try:
                    logger.info("Saving trained model...")
                    model_save_path = checkpoint_dir / "final_model"
                    ppo_trainer.save_pretrained(model_save_path)
                    logger.info("Model saved to: %s", model_save_path)
                except Exception as exc:  # pragma: no cover - best effort save
                    logger.warning("Failed to save model: %s", exc)

        except Exception as exc:
            training_successful = False
            error_msg = str(exc)
            logger.exception("PPO training failed: %s", exc)
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
            raise ExecutionError(error_msg or "PPO training failed") from exc

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

        logger.info("PPO training task completed in %.2f seconds", training_time)
        return results

    def _load_dataset(self, spec: Dict[str, Any]) -> Dataset:
        """Load training dataset"""
        data_config = spec.get("data", {})
        
        if "prompts" in data_config:
            # Use provided prompts directly
            prompts = data_config["prompts"]
            # Convert to the format expected by TRL PPO
            dataset_dict = {"query": prompts}
            dataset = Dataset.from_dict(dataset_dict)
        else:
            # Default demo dataset
            prompts = [
                "Write a positive review for a restaurant:",
                "Describe a beautiful sunset:",
                "Tell me about a helpful AI assistant:",
                "Write a motivational quote:",
            ]
            dataset_dict = {"query": prompts}
            dataset = Dataset.from_dict(dataset_dict)

        return dataset
