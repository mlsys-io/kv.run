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
from transformers import AutoTokenizer, GenerationConfig
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

            # Some TRL versions expect `generation_config` on the policy/ref models.
            # AutoModelForCausalLMWithValueHead may not set it; ensure presence.
            def _ensure_generation_config(m):
                if not hasattr(m, "generation_config") or m.generation_config is None:
                    try:
                        gen_cfg = GenerationConfig.from_pretrained(self._model_name)
                    except Exception:
                        try:
                            gen_cfg = GenerationConfig.from_model_config(getattr(m, "config", None))
                        except Exception:
                            gen_cfg = GenerationConfig()
                    try:
                        m.generation_config = gen_cfg
                    except Exception:
                        # As a fallback, set eos_token_id on config if available
                        if hasattr(m, "config") and hasattr(gen_cfg, "eos_token_id"):
                            m.config.eos_token_id = getattr(gen_cfg, "eos_token_id", None)

            _ensure_generation_config(model)
            _ensure_generation_config(ref_model)

            # Ensure TRL's PolicyAndValueWrapper can locate the backbone via
            # `<model>.base_model_prefix` attribute and a matching attribute on the instance.
            def _ensure_backbone(m):
                # Common attribute names for the wrapped transformer backbone
                candidates = [
                    getattr(m, "pretrained_model", None),
                    getattr(m, "transformer", None),
                    getattr(m, "model", None),
                    getattr(m, "base_model", None),
                ]
                backbone = next((c for c in candidates if c is not None), None)
                if backbone is None:
                    return
                # Prefer the underlying module's own base_model_prefix if present
                prefix = getattr(backbone, "base_model_prefix", None) or "model"
                try:
                    setattr(m, prefix, backbone)
                    m.base_model_prefix = prefix
                except Exception:
                    # Fallback to a neutral attribute name
                    setattr(m, "backbone", backbone)
                    m.base_model_prefix = "backbone"

            _ensure_backbone(model)
            _ensure_backbone(ref_model)

            # Ensure flag expected by TRL's wrapper exists on policy/value models
            def _ensure_grad_ckpt_flag(m):
                try:
                    if hasattr(m, "is_gradient_checkpointing"):
                        return
                    # Derive from backbone or config if possible
                    val = False
                    try:
                        backbone = None
                        if hasattr(m, "base_model_prefix") and hasattr(m, m.base_model_prefix):
                            backbone = getattr(m, m.base_model_prefix)
                        if backbone is not None and hasattr(backbone, "is_gradient_checkpointing"):
                            val = bool(getattr(backbone, "is_gradient_checkpointing"))
                        elif hasattr(m, "config") and hasattr(m.config, "gradient_checkpointing"):
                            val = bool(getattr(m.config, "gradient_checkpointing"))
                    except Exception:
                        pass
                    setattr(m, "is_gradient_checkpointing", val)
                except Exception:
                    pass

            _ensure_grad_ckpt_flag(model)
            _ensure_grad_ckpt_flag(ref_model)

            # Ensure models return ModelOutput (not tuple) so TRL can access `.logits`
            def _ensure_return_dict(m):
                try:
                    if hasattr(m, "config") and hasattr(m.config, "use_return_dict"):
                        m.config.use_return_dict = True
                    # Also try backbone config if present
                    if hasattr(m, "base_model_prefix") and hasattr(m, m.base_model_prefix):
                        backbone = getattr(m, m.base_model_prefix)
                        if hasattr(backbone, "config") and hasattr(backbone.config, "use_return_dict"):
                            backbone.config.use_return_dict = True
                except Exception:
                    pass

            _ensure_return_dict(model)
            _ensure_return_dict(ref_model)

            logger.info("Models loaded: %s", self._model_name)

            # Load dataset
            logger.info("Loading dataset...")
            dataset = self._load_dataset(spec)
            logger.info("Dataset loaded with %d samples", len(dataset))

            # Some TRL versions expect tokenized inputs in the dataset and will
            # route through a padding collator. Ensure input_ids/attention_mask exist.
            try:
                max_len = int(training_config.get("max_seq_length", 512))
                def _tok_fn(batch):
                    texts = batch.get("query") or []
                    enc = tokenizer(
                        texts,
                        padding=False,
                        truncation=True,
                        max_length=max_len,
                    )
                    return enc

                if "input_ids" not in dataset.column_names:
                    dataset = dataset.map(_tok_fn, batched=True, remove_columns=[])
                    logger.info("Tokenized dataset with max_length=%d", max_len)
            except Exception as _e:
                logger.warning("Skipping dataset tokenization step: %s", _e)

            # Provide a simple collator to avoid Transformers' padding collator
            # attempting to pad string fields like 'query'. This lets TRL handle
            # tokenization/length management internally during PPO.
            def _simple_collate(features):
                if not features:
                    return {}
                f0 = features[0]
                # If already tokenized, pad into tensors
                if "input_ids" in f0:
                    items = []
                    for f in features:
                        item = {"input_ids": f["input_ids"]}
                        if "attention_mask" in f:
                            item["attention_mask"] = f["attention_mask"]
                        items.append(item)
                    batch = tokenizer.pad(items, padding=True, return_tensors="pt")
                    return batch
                # Otherwise, pass raw queries through
                if "query" in f0:
                    return {"query": [f["query"] for f in features]}
                # Fallback: return all fields as lists
                keys = f0.keys()
                return {k: [f[k] for f in features] for k in keys}

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
            # TRL PPOTrainer signature varies widely across versions. Use
            # introspection to build arguments positionally/with kwargs to fit.
            import inspect

            def build_trainer() -> PPOTrainer:
                sig = inspect.signature(PPOTrainer.__init__)
                params = list(sig.parameters.values())[1:]  # drop self

                mapping = {
                    "config": ppo_config,
                    "args": ppo_config,
                    "ppo_config": ppo_config,
                    "processing_class": tokenizer,
                    "tokenizer": tokenizer,
                    "reward_model": ref_model,
                    "value_model": model,
                    "model": model,
                    "ref_model": ref_model,
                    "train_dataset": dataset,
                    "dataset": dataset,
                    "output_dir": str(checkpoint_dir),
                    "data_collator": _simple_collate,
                    "collate_fn": _simple_collate,
                }

                positional = []
                kwargs = {}
                missing_required = []

                for p in params:
                    if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                        continue
                    if p.default is inspect._empty:
                        if p.name in mapping:
                            positional.append(mapping[p.name])
                        else:
                            # Some older versions expect (config, processing_class, reward_model, value_model)
                            # Try to satisfy common required names by aliases
                            alias = None
                            if p.name == "processing_class" and "tokenizer" in mapping:
                                alias = mapping["tokenizer"]
                            elif p.name == "dataset" and "train_dataset" in mapping:
                                alias = mapping["train_dataset"]
                            if alias is not None:
                                positional.append(alias)
                            else:
                                missing_required.append(p.name)
                    else:
                        # Optional: provide via kwargs if we have a value
                        if p.name in mapping:
                            kwargs[p.name] = mapping[p.name]

                if missing_required:
                    # As a fallback, try known positional legacy order
                    # (config/args, processing_class, reward_model, value_model, model, ref_model, train_dataset)
                    legacy_seq = []
                    for key in ("args", "processing_class", "reward_model", "value_model", "model", "ref_model", "train_dataset"):
                        if key in mapping:
                            legacy_seq.append(mapping[key])
                    try:
                        return PPOTrainer(*legacy_seq, **kwargs)
                    except TypeError:
                        # Raise with details for debugging
                        raise TypeError(f"PPOTrainer signature mismatch; missing required params: {missing_required}")

                return PPOTrainer(*positional, **kwargs)

            ppo_trainer = build_trainer()
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
