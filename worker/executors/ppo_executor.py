#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Executor using TRL's simple approach

Simplified implementation using TRL's built-in training methods.
"""

import json
import os
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

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
        training_config = spec.get("training", {}) or {}
        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        launcher_flag = "KV_PPO_DISTRIBUTED"
        already_spawned = os.environ.get(launcher_flag) == "1"
        gpu_count = self._detect_gpu_count(training_config)
        allow_multi_cfg = training_config.get("allow_multi_gpu")
        allow_multi = bool(allow_multi_cfg) if allow_multi_cfg is not None else gpu_count > 1

        if allow_multi and not already_spawned and gpu_count > 1:
            self._spawn_distributed(task, out_dir, gpu_count, launcher_flag, training_config)
            resp_path = out_dir / "responses.json"
            if resp_path.exists():
                return self.load_json(resp_path)
            return {
                "training_successful": True,
                "spawned_torchrun": True,
                "model_name": (spec.get("model", {}).get("source", {}) or {}).get("identifier"),
                "output_dir": str(out_dir),
            }

        start_time = time.time()

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

            # Ensure hidden states are returned for reward computation in TRL 0.23
            def _ensure_output_hidden_states(m):
                try:
                    if hasattr(m, "config") and hasattr(m.config, "output_hidden_states"):
                        m.config.output_hidden_states = True
                    if hasattr(m, "base_model_prefix") and hasattr(m, m.base_model_prefix):
                        backbone = getattr(m, m.base_model_prefix)
                        if hasattr(backbone, "config") and hasattr(backbone.config, "output_hidden_states"):
                            backbone.config.output_hidden_states = True
                except Exception:
                    pass

            _ensure_output_hidden_states(model)
            _ensure_output_hidden_states(ref_model)

            # As a last resort, monkey-patch forward to always expose `.logits`
            try:
                from types import SimpleNamespace
                import torch

                def _patch_forward_returns_logits(m):
                    orig_forward = m.forward
                    def wrapped_forward(*args, **kwargs):
                        # Normalize calls to avoid duplicate bindings of input_ids.
                        # Extract potential input_ids/attention_mask from positional[0] if present.
                        new_kwargs = dict(kwargs)
                        if len(args) > 0:
                            first = args[0]
                            if isinstance(first, torch.Tensor):
                                new_kwargs["input_ids"] = first
                            elif isinstance(first, dict):
                                if "input_ids" in first:
                                    new_kwargs["input_ids"] = first["input_ids"]
                                if "attention_mask" in first and "attention_mask" not in new_kwargs:
                                    new_kwargs["attention_mask"] = first["attention_mask"]
                        # Ignore all positional args; call forward with kwargs only.
                        out = orig_forward(**new_kwargs)
                        if isinstance(out, tuple):
                            return SimpleNamespace(logits=out[0])
                        return out
                    m.forward = wrapped_forward.__get__(m, m.__class__)

                _patch_forward_returns_logits(model)
                _patch_forward_returns_logits(ref_model)
            except Exception:
                pass

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
            # Optional args to control saving behavior and memory
            ppo_optional = {}
            if "save_safetensors" in training_config:
                ppo_optional["save_safetensors"] = bool(training_config["save_safetensors"])  # transformers arg
            else:
                # Default off to avoid shared-tensor safetensors error when embeddings are tied
                ppo_optional["save_safetensors"] = False

            ppo_config = PPOConfig(
                learning_rate=float(training_config.get("learning_rate", 1.41e-5)),
                batch_size=int(training_config.get("batch_size", 4)),
                mini_batch_size=int(training_config.get("mini_batch_size", 1)),
                output_dir=str(checkpoint_dir),
                **ppo_optional,
            )
            logger.info("PPOConfig created successfully")

            # Initialize PPO trainer with correct API
            logger.info("Creating PPOTrainer...")
            # TRL PPOTrainer signature varies widely across versions. Use
            # introspection to build arguments positionally/with kwargs to fit.
            import inspect

            # Build a lightweight reward adapter expected by TRL 0.23 (needs .score)
            import torch
            class _RewardAdapter(torch.nn.Module):
                def __init__(self, value_head_model):
                    super().__init__()
                    # Keep reference to underlying LM/value-head model for delegation
                    self._lm = value_head_model
                    head = getattr(value_head_model, "v_head", None)
                    if head is None:
                        # Try common alternatives or construct a linear head
                        head = getattr(value_head_model, "value_head", None)
                    if head is None:
                        hidden = None
                        try:
                            hidden = getattr(value_head_model.config, "hidden_size", None)
                        except Exception:
                            pass
                        if hidden is None:
                            # Try backbone config
                            try:
                                if hasattr(value_head_model, "base_model_prefix") and hasattr(value_head_model, value_head_model.base_model_prefix):
                                    backbone = getattr(value_head_model, value_head_model.base_model_prefix)
                                    hidden = getattr(getattr(backbone, "config", None), "hidden_size", None)
                            except Exception:
                                pass
                        if hidden is None:
                            raise AttributeError("Cannot infer hidden_size for reward head")
                        head = torch.nn.Linear(int(hidden), 1, bias=False)
                    self.v_head = head
                    # Mirror key attributes expected by TRL utils
                    self.base_model_prefix = getattr(value_head_model, "base_model_prefix", "model")
                    if hasattr(value_head_model, self.base_model_prefix):
                        setattr(self, self.base_model_prefix, getattr(value_head_model, self.base_model_prefix))
                    for attr in ("config", "generation_config"):
                        if hasattr(value_head_model, attr):
                            setattr(self, attr, getattr(value_head_model, attr))
                def score(self, hidden_states):
                    return self.v_head(hidden_states)
                def __getattr__(self, name):
                    try:
                        return super().__getattr__(name)
                    except AttributeError:
                        return getattr(self.__dict__.get("_lm", object()), name)

            reward_adapter = _RewardAdapter(model)
            # Attach .score to underlying models too, in case TRL bypasses reward_model
            try:
                import types as _types_mod
                def _score_impl(self, hidden_states):
                    head = getattr(self, "v_head", None) or getattr(self, "value_head", None)
                    if head is None:
                        raise AttributeError("Model lacks v_head/value_head for reward scoring")
                    return head(hidden_states)
                if not hasattr(model, "score"):
                    model.score = _types_mod.MethodType(_score_impl, model)
                if not hasattr(ref_model, "score"):
                    ref_model.score = _types_mod.MethodType(_score_impl, ref_model)
            except Exception:
                pass

            def build_trainer() -> PPOTrainer:
                sig = inspect.signature(PPOTrainer.__init__)
                params = list(sig.parameters.values())[1:]  # drop self

                mapping = {
                    "config": ppo_config,
                    "args": ppo_config,
                    "ppo_config": ppo_config,
                    "processing_class": tokenizer,
                    "tokenizer": tokenizer,
                    "reward_model": reward_adapter,
                    "value_model": model,
                    "model": model,
                    "ref_model": ref_model,
                    "train_dataset": dataset,
                    "eval_dataset": dataset,
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
            # Ensure eval dataset/dataloader exist for TRL 0.23 `generate_completions`.
            try:
                if getattr(ppo_trainer, "eval_dataset", None) is None:
                    setattr(ppo_trainer, "eval_dataset", dataset)
                # Try to build eval_dataloader via trainer helper if available
                if getattr(ppo_trainer, "eval_dataloader", None) is None:
                    build_fn = getattr(ppo_trainer, "get_eval_dataloader", None) or getattr(ppo_trainer, "_build_eval_dataloader", None)
                    if callable(build_fn):
                        try:
                            edl = build_fn()
                            if edl is not None:
                                setattr(ppo_trainer, "eval_dataloader", edl)
                        except Exception:
                            pass
            except Exception:
                pass
            # Ensure TRL uses our reward adapter (with .score) even if constructor
            # variant didn't bind it as expected.
            try:
                if hasattr(ppo_trainer, "reward_model"):
                    # Move adapter to same device as policy/value if possible
                    try:
                        import torch
                        dev = None
                        if hasattr(model, "device"):
                            dev = model.device
                        elif hasattr(model, "parameters"):
                            dev = next(model.parameters()).device
                        if dev is not None and isinstance(reward_adapter, torch.nn.Module):
                            reward_adapter.to(dev)
                    except Exception:
                        pass
                    ppo_trainer.reward_model = reward_adapter
            except Exception:
                pass
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
                    # Prefer save_model to avoid safetensors shared-tensor errors
                    ppo_trainer.save_model(model_save_path)
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

    def _spawn_distributed(
        self,
        task: Dict[str, Any],
        out_dir: Path,
        n_gpus: int,
        launcher_flag: str,
        training_config: Dict[str, Any],
    ) -> None:
        launcher_dir = out_dir / "_launcher"
        launcher_dir.mkdir(parents=True, exist_ok=True)
        task_file = launcher_dir / "task_spec.json"
        with task_file.open("w", encoding="utf-8") as fh:
            json.dump(task, fh)

        nproc = int(training_config.get("nproc_per_node", n_gpus))
        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(nproc),
            "-m",
            "worker.executors.ppo_dist_entry",
            str(task_file),
            str(out_dir),
        ]
        env = os.environ.copy()
        env[launcher_flag] = "1"
        try:
            repo_root = Path(__file__).resolve().parents[2]
            env["PYTHONPATH"] = f"{str(repo_root)}:" + env.get("PYTHONPATH", "")
        except Exception:
            pass
        logger.info(
            "Spawning torchrun for PPO: %s (CUDA_VISIBLE_DEVICES=%s)",
            " ".join(cmd),
            env.get("CUDA_VISIBLE_DEVICES"),
        )
        subprocess.check_call(cmd, env=env)

    @staticmethod
    def _detect_gpu_count(training_config: Dict[str, Any]) -> int:
        vis = training_config.get("visible_devices") or os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis:
            tokens = [dev.strip() for dev in str(vis).split(",") if dev.strip()]
            if tokens:
                return len(tokens)
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            pass
        return 0
