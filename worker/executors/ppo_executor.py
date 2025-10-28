#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Executor using TRL's simple approach

Simplified implementation using TRL's built-in training methods.
"""

import gc
import json
import os
import subprocess
import time
import logging
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from .base_executor import Executor, ExecutionError
from .checkpoint_utils import archive_model_dir, get_http_destination

logger = logging.getLogger("worker.ppo")


class _ExternalRewardModel(torch.nn.Module):
    """Wraps a sequence classification model to score decoded PPO responses."""

    def __init__(self, reward_cfg: Dict[str, Any], policy_tokenizer: AutoTokenizer):
        super().__init__()
        identifier = reward_cfg.get("identifier")
        if not identifier:
            raise ValueError("reward_model.identifier is required for external reward models")

        self.policy_tokenizer = policy_tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(identifier)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(identifier)
        self.reward_model.to("cpu")
        self.base_model_prefix = "reward_model"
        self._patch_reward_forward()
        for param in self.reward_model.parameters():
            param.requires_grad_(False)
        self.reward_model.eval()

        self.reward_type = str(reward_cfg.get("type", "classification")).lower()
        self.scale = float(reward_cfg.get("scale", 1.0))
        self.max_length = int(
            reward_cfg.get(
                "max_length",
                min(getattr(self.reward_tokenizer, "model_max_length", 512) or 512, 512),
            )
        )
        self.positive_label = reward_cfg.get("positive_label")
        self.negative_label = reward_cfg.get("negative_label")

        id2label = getattr(self.reward_model.config, "id2label", {}) or {}
        self._id2label = {int(k): str(v) for k, v in id2label.items()}
        self._label2id = {v.lower(): k for k, v in self._id2label.items()}

    def forward(self, *args, **kwargs):
        raise RuntimeError("External reward model is only used via compute_reward_from_tokens.")

    def to(self, *args, **kwargs):  # type: ignore[override]
        # Keep the reward model on CPU to avoid GPU-side driver asserts with classification heads.
        self.reward_model.to("cpu")
        return self

    def _patch_reward_forward(self) -> None:
        try:
            original_forward = self.reward_model.forward
        except AttributeError:
            return

        def wrapped_forward(*args, **kwargs):
            kwargs.pop("use_cache", None)
            kwargs.pop("output_hidden_states", None)
            return original_forward(*args, **kwargs)

        try:
            self.reward_model.forward = wrapped_forward  # type: ignore[assignment]
        except Exception:
            pass

        try:
            if hasattr(self.reward_model.config, "use_cache"):
                self.reward_model.config.use_cache = False
        except Exception:
            pass

        try:
            backbone = getattr(self.reward_model, "base_model", None)
            if backbone is not None and hasattr(backbone, "config") and hasattr(backbone.config, "use_cache"):
                backbone.config.use_cache = False
        except Exception:
            pass

    def compute_reward_from_tokens(
        self,
        query_responses: torch.Tensor,
        pad_token_id: int,
        context_length: int,
    ):
        device = query_responses.device
        batch_size, seq_len = query_responses.shape

        response_texts = []
        for sample in query_responses:
            response_tokens = sample[context_length:]
            if pad_token_id is not None:
                response_tokens = response_tokens[response_tokens != pad_token_id]
            text = self.policy_tokenizer.decode(
                response_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            response_texts.append(text)

        reward_scores = self._score_texts(response_texts).to(device)

        reward_logits = reward_scores.view(batch_size, 1, 1).expand(batch_size, seq_len, 1).contiguous()
        non_pad = (query_responses != pad_token_id).int()
        sequence_lengths = non_pad.sum(dim=1) - 1
        sequence_lengths = torch.clamp(sequence_lengths, min=0)

        return reward_logits, reward_scores, sequence_lengths

    def _score_texts(self, texts):
        model_device = next(self.reward_model.parameters()).device
        if not texts:
            return torch.zeros(0, dtype=torch.float32, device=model_device)

        encoded = self.reward_tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.reward_model(**encoded)
            logits = outputs.logits.float()

        if logits.shape[-1] == 1:
            rewards = logits.squeeze(-1)
        else:
            if self.reward_type == "sentiment":
                pos_idx = self._resolve_label(self.positive_label, "positive")
                neg_idx = self._resolve_label(self.negative_label, "negative")
                pos_score = logits[:, pos_idx] if pos_idx is not None else logits.max(dim=-1).values
                neg_score = logits[:, neg_idx] if neg_idx is not None else torch.zeros_like(pos_score)
                rewards = pos_score - neg_score
            else:
                label_idx = self._resolve_label(self.positive_label, None)
                rewards = logits[:, label_idx] if label_idx is not None else logits.max(dim=-1).values

        return rewards.float().to(model_device) * self.scale

    def _resolve_label(self, label, fallback_keyword: Optional[str]) -> Optional[int]:
        if label is None and fallback_keyword:
            return next((idx for idx, name in self._id2label.items() if fallback_keyword in name.lower()), None)

        if label is None:
            return None

        if isinstance(label, int):
            return label if label in self._id2label else None

        key = str(label).lower()
        if key in self._label2id:
            return self._label2id[key]
        return None


@contextmanager
def _patched_reward_dispatch():
    from trl.trainer import utils as trl_utils
    from trl.trainer import ppo_trainer as trl_ppo

    original_get_reward = trl_utils.get_reward
    original_get_reward_ppo = getattr(trl_ppo, "get_reward", None)

    def _wrapped_get_reward(model, query_responses, pad_token_id, context_length):
        if hasattr(model, "compute_reward_from_tokens"):
            return model.compute_reward_from_tokens(query_responses, pad_token_id, context_length)
        return original_get_reward(model, query_responses, pad_token_id, context_length)

    trl_utils.get_reward = _wrapped_get_reward
    if original_get_reward_ppo is not None:
        trl_ppo.get_reward = _wrapped_get_reward
    try:
        yield
    finally:
        trl_utils.get_reward = original_get_reward
        if original_get_reward_ppo is not None:
            trl_ppo.get_reward = original_get_reward_ppo


class _RewardAdapter(torch.nn.Module):
    """Fallback reward adapter that reuses the policy value head."""

    def __init__(self, value_head_model: torch.nn.Module):
        super().__init__()
        self._lm = value_head_model
        head = getattr(value_head_model, "v_head", None) or getattr(value_head_model, "value_head", None)
        if head is None:
            hidden = None
            try:
                hidden = getattr(value_head_model.config, "hidden_size", None)
            except Exception:
                pass
            if hidden is None:
                try:
                    if hasattr(value_head_model, "base_model_prefix") and hasattr(
                        value_head_model, value_head_model.base_model_prefix
                    ):
                        backbone = getattr(value_head_model, value_head_model.base_model_prefix)
                        hidden = getattr(getattr(backbone, "config", None), "hidden_size", None)
                except Exception:
                    pass
            if hidden is None:
                raise AttributeError("Cannot infer hidden_size for reward head")
            head = torch.nn.Linear(int(hidden), 1, bias=False)
        self.v_head = head
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


class PPOExecutor(Executor):
    """PPO training executor using TRL library."""

    def __init__(self):
        super().__init__()
        self._model_name = None
        self._policy_model = None
        self._ref_model = None
        self._tokenizer = None
        self._ppo_trainer = None
        self._reward_module = None
        self._task_out_dir: Optional[Path] = None

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        logger.info("Starting PPO training task")
        spec = (task or {}).get("spec") or {}
        training_config = spec.get("training", {}) or {}
        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._task_out_dir = out_dir

        launcher_flag = "KV_PPO_DISTRIBUTED"
        already_spawned = os.environ.get(launcher_flag) == "1"
        gpu_count = self._detect_gpu_count(training_config)
        allow_multi_cfg = training_config.get("allow_multi_gpu")
        allow_multi = bool(allow_multi_cfg) if allow_multi_cfg is not None else gpu_count > 1

        if allow_multi and not already_spawned and gpu_count > 1:
            self._spawn_distributed(task, out_dir, gpu_count, launcher_flag, training_config)
            resp_path = out_dir / "responses.json"
            if resp_path.exists():
                result = self.load_json(resp_path)
                self._task_out_dir = None
                return result
            self._task_out_dir = None
            return {
                "training_successful": True,
                "spawned_torchrun": True,
                "model_name": (spec.get("model", {}).get("source", {}) or {}).get("identifier"),
                "output_dir": str(out_dir),
            }

        start_time = time.time()

        final_model_path: Optional[Path] = None
        final_archive_path: Optional[Path] = None

        try:
            # Get model configuration
            model_config = spec.get("model", {})
            model_source = model_config.get("source", {})
            self._model_name = model_source.get("identifier", "microsoft/DialoGPT-small")

            logger.info("Loading model and tokenizer...")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._tokenizer = tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load models - PPO requires policy model and reference model
            model = AutoModelForCausalLMWithValueHead.from_pretrained(self._model_name)
            ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(self._model_name)
            self._policy_model = model
            self._ref_model = ref_model

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

            # Ensure value head exposes score() for TRL reward helpers
            self._ensure_value_head_score(model, ref_model)

            # Load dataset
            logger.info("Loading dataset...")
            dataset = self._load_dataset(spec)
            dataset_size = len(dataset)
            logger.info("Dataset loaded with %d samples", dataset_size)

            def _safe_int(value: Any, *, default: Optional[int] = None, minimum: Optional[int] = None) -> Optional[int]:
                if value is None:
                    return default
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    return default
                if minimum is not None:
                    parsed = max(parsed, minimum)
                return parsed

            per_device_batch = _safe_int(
                training_config.get("per_device_train_batch_size"),
                default=_safe_int(training_config.get("batch_size"), default=1, minimum=1),
                minimum=1,
            )
            if dataset_size and per_device_batch and per_device_batch > dataset_size:
                logger.info(
                    "Clipping per_device_train_batch_size from %d to dataset size %d to avoid empty PPO batches",
                    per_device_batch,
                    dataset_size,
                )
                per_device_batch = dataset_size

            grad_acc_steps = _safe_int(training_config.get("gradient_accumulation_steps"), default=1, minimum=1)
            num_mini_batches = _safe_int(training_config.get("num_mini_batches"), default=1, minimum=1)

            def _safe_float(value: Any, *, default: Optional[float] = None) -> Optional[float]:
                if value is None:
                    return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

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

            reward_module, reward_is_external, reward_ctx = self._prepare_reward_model(
                spec,
                tokenizer,
                model,
                ref_model,
            )
            self._reward_module = reward_module
            if hasattr(reward_module, "eval"):
                reward_module.eval()

            logger.info("Creating PPOConfig...")
            # Optional args to control saving behavior and memory
            ppo_optional = {}
            if "save_safetensors" in training_config:
                ppo_optional["save_safetensors"] = bool(training_config["save_safetensors"])  # transformers arg
            else:
                # Default off to avoid shared-tensor safetensors error when embeddings are tied
                ppo_optional["save_safetensors"] = False
            ppo_optional["remove_unused_columns"] = False

            ppo_config = PPOConfig(
                learning_rate=float(training_config.get("learning_rate", 1.41e-5)),
                batch_size=int(training_config.get("batch_size", per_device_batch or 1)),
                mini_batch_size=int(training_config.get("mini_batch_size", num_mini_batches or 1)),
                output_dir=str(checkpoint_dir),
                seed=int(training_config.get("seed", 42)),
                **ppo_optional,
            )

            if per_device_batch is not None:
                ppo_config.per_device_train_batch_size = per_device_batch
            if grad_acc_steps is not None:
                ppo_config.gradient_accumulation_steps = grad_acc_steps
            if num_mini_batches is not None:
                ppo_config.num_mini_batches = num_mini_batches

            ppo_epochs = _safe_int(training_config.get("ppo_epochs"), minimum=1)
            if ppo_epochs is not None:
                ppo_config.num_ppo_epochs = ppo_epochs

            train_epochs = _safe_float(training_config.get("num_train_epochs"), default=None)
            if train_epochs is not None:
                ppo_config.num_train_epochs = max(train_epochs, 1.0)
            else:
                # Default to 1 pass over the data unless overridden
                ppo_config.num_train_epochs = 1.0

            kl_coef = _safe_float(training_config.get("kl_coef"), default=None)
            if kl_coef is not None and kl_coef > 0:
                ppo_config.kl_coef = kl_coef

            response_cfg = spec.get("generation", {}) or {}
            response_length = _safe_int(response_cfg.get("max_new_tokens"), default=None, minimum=1)
            if response_length is not None:
                ppo_config.response_length = response_length
            else:
                ppo_config.response_length = int(training_config.get("max_seq_length", 64))

            temperature = _safe_float(response_cfg.get("temperature"), default=None)
            if temperature is None:
                temperature = _safe_float(training_config.get("temperature"), default=None)
            if temperature is not None and temperature > 0:
                ppo_config.temperature = temperature

            stop_token = response_cfg.get("stop")
            if isinstance(stop_token, str):
                ppo_config.stop_token = stop_token

            logger.info(
                "Final PPO batch parameters: per_device=%s, grad_acc=%s, num_mini_batches=%s",
                per_device_batch,
                grad_acc_steps,
                num_mini_batches,
            )

            steps_requested = _safe_int(training_config.get("steps"), default=None, minimum=1)
            if steps_requested is not None and per_device_batch:
                total_episodes = steps_requested * max(1, per_device_batch)
                ppo_config.total_episodes = total_episodes
                logger.info(
                    "Configuring PPO to run %d update steps (~%d episodes)",
                    steps_requested,
                    total_episodes,
                )
            else:
                logger.info(
                    "Using num_train_epochs=%.2f over %d samples (per_device_batch=%s, grad_acc=%s)",
                    float(getattr(ppo_config, "num_train_epochs", 1.0)),
                    dataset_size,
                    per_device_batch,
                    grad_acc_steps,
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
                    "reward_model": reward_module,
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
            self._ppo_trainer = ppo_trainer
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
            try:
                if hasattr(ppo_trainer, "reward_model"):
                    ppo_trainer.reward_model = reward_module
            except Exception:
                pass
            logger.info("PPOTrainer created successfully")

            if reward_is_external:
                logger.info("External reward model enabled for PPO training")

            logger.info("Starting PPO training...")
            with reward_ctx:
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
                    final_model_path = model_save_path
                    destination = get_http_destination(task)
                    if destination:
                        try:
                            final_archive_path = archive_model_dir(model_save_path)
                            logger.info("Archived PPO model to %s for HTTP delivery", final_archive_path)
                        except Exception as arch_exc:
                            logger.warning("Failed to archive PPO model for upload: %s", arch_exc)
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
            self._task_out_dir = None
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
        if final_model_path is not None:
            results["final_model_path"] = str(final_model_path)
        if final_archive_path is not None:
            results["final_model_archive"] = final_archive_path.name
            results["final_model_archive_path"] = str(final_archive_path)
            self._upload_model_archive(task, final_archive_path, results)
        self.save_json(out_dir / "responses.json", results)

        logger.info("PPO training task completed in %.2f seconds", training_time)
        self._task_out_dir = None
        return results

    def _ensure_jsonl_local(self, jsonl_cfg: Dict[str, Any], default_name: str) -> Path:
        path_value = str(jsonl_cfg.get("path") or "").strip()
        url_value = str(jsonl_cfg.get("url") or jsonl_cfg.get("download_url") or "").strip()

        candidate: Optional[Path] = None
        if path_value:
            candidate_path = Path(path_value).expanduser()
            if not candidate_path.is_absolute() and self._task_out_dir:
                candidate_path = (self._task_out_dir / candidate_path).resolve()
            else:
                candidate_path = candidate_path.resolve()
            if candidate_path.exists():
                jsonl_cfg["path"] = str(candidate_path)
                return candidate_path
            candidate = candidate_path

        if url_value:
            if self._task_out_dir:
                cache_dir = (self._task_out_dir / "_jsonl_cache").resolve()
            else:
                cache_dir = Path(tempfile.mkdtemp(prefix="mloc_jsonl_")).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            parsed = urlparse(url_value)
            name = Path(parsed.path).name or (candidate.name if candidate else default_name)
            if not name:
                name = default_name
            target = cache_dir / name
            try:
                import requests

                with requests.get(url_value, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    with target.open("wb") as fh:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                fh.write(chunk)
            except Exception as exc:  # pragma: no cover
                raise ExecutionError(f"Failed to download JSONL dataset from {url_value}: {exc}") from exc
            jsonl_cfg["path"] = str(target)
            return target.resolve()

        if candidate is not None:
            raise ExecutionError(f"JSONL dataset not found: {candidate}")
        raise ExecutionError("data.jsonl.path is required when using JSONL input")

    def _load_dataset(self, spec: Dict[str, Any]) -> Dataset:
        """Load training dataset"""
        data_config = spec.get("data", {})
        
        jsonl_cfg = data_config.get("jsonl")
        jsonl_path = data_config.get("jsonl_path")
        if jsonl_cfg or jsonl_path:
            if jsonl_cfg is None:
                jsonl_cfg = {}
            else:
                jsonl_cfg = dict(jsonl_cfg)
            if jsonl_path:
                jsonl_cfg.setdefault("path", jsonl_path)

            default_name = Path(str(jsonl_cfg.get("path") or "dataset.jsonl")).name or "dataset.jsonl"
            jsonl_file = self._ensure_jsonl_local(jsonl_cfg, default_name=default_name)

            query_field = (
                jsonl_cfg.get("query_field")
                or data_config.get("query_field")
                or "query"
            )
            response_field = (
                jsonl_cfg.get("response_field")
                or data_config.get("response_field")
            )

            prompts: List[str] = []
            references: Optional[List[str]] = [] if response_field else None

            with jsonl_file.open("r", encoding="utf-8") as fh:
                for line_number, raw in enumerate(fh, start=1):
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ExecutionError(
                            f"Invalid JSON on line {line_number} of {jsonl_file}: {exc}"
                        ) from exc

                    if query_field not in record:
                        raise ExecutionError(
                            f"JSONL record missing required field '{query_field}' on line {line_number}"
                        )
                    prompts.append(str(record[query_field]))
                    if references is not None:
                        references.append(str(record.get(response_field, "")))

            if not prompts:
                raise ExecutionError(f"JSONL dataset at {jsonl_file} is empty")

            dataset_dict: Dict[str, Any] = {"query": prompts}
            if references is not None:
                dataset_dict["reference_response"] = references
            dataset = Dataset.from_dict(dataset_dict)

        elif "prompts" in data_config:
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

    def _prepare_reward_model(
        self,
        spec: Dict[str, Any],
        tokenizer: AutoTokenizer,
        value_model: AutoModelForCausalLMWithValueHead,
        ref_model: AutoModelForCausalLMWithValueHead,
    ):
        reward_cfg = (spec or {}).get("reward_model") or {}
        if reward_cfg:
            identifier = reward_cfg.get("identifier")
            try:
                external_model = _ExternalRewardModel(reward_cfg, tokenizer)
                external_model.base_model_prefix = "reward_model"
                logger.info(
                    "Using external reward model '%s' (type=%s)",
                    identifier,
                    external_model.reward_type,
                )
                return external_model, True, _patched_reward_dispatch()
            except Exception as exc:
                logger.warning(
                    "Failed to initialize reward model '%s' (type=%s); falling back to value-head rewards. Error: %s",
                    identifier,
                    reward_cfg.get("type"),
                    exc,
                )
        reward_adapter = _RewardAdapter(value_model)
        self._ensure_value_head_score(value_model, ref_model)
        return reward_adapter, False, nullcontext()

    @staticmethod
    def _ensure_value_head_score(
        value_model: AutoModelForCausalLMWithValueHead,
        ref_model: Optional[AutoModelForCausalLMWithValueHead],
    ) -> None:
        import types

        def _score_impl(self, hidden_states):
            head = getattr(self, "v_head", None) or getattr(self, "value_head", None)
            if head is None:
                raise AttributeError("Model lacks v_head/value_head for reward scoring")
            return head(hidden_states)

        for target in (value_model, ref_model):
            if target is None:
                continue
            if not hasattr(target, "score"):
                try:
                    target.score = types.MethodType(_score_impl, target)
                except Exception:
                    pass

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

    def _upload_model_archive(self, task: Dict[str, Any], archive_path: Path, payload: Dict[str, Any]) -> None:
        destination = get_http_destination(task)
        task_id = (task or {}).get("task_id")
        if not destination or not task_id:
            return
        base_url = destination["url"].rstrip("/")
        upload_url = base_url + f"/{task_id}/files"
        try:
            import requests

            file_size = archive_path.stat().st_size if archive_path.exists() else 0
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
            download_url = base_url + f"/{task_id}/files/{archive_path.name}"
            payload["final_model_archive_url"] = download_url
            payload["final_model_archive_bytes"] = file_size
            payload["final_model_archive_uploaded"] = True
            logger.info("Uploaded PPO model archive to %s (%d bytes)", upload_url, file_size)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("PPO model archive upload failed for %s: %s", task_id, exc)

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

    def cleanup_after_run(self) -> None:
        dropped_objects = []
        for attr in (
            "_ppo_trainer",
            "_policy_model",
            "_ref_model",
            "_tokenizer",
            "_reward_module",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                dropped_objects.append(obj)
            setattr(self, attr, None)

        for obj in dropped_objects:
            try:
                del obj
            except Exception:
                pass

        try:
            import torch

            dist = getattr(torch, "distributed", None)
            if dist is not None:
                try:
                    if dist.is_available() and dist.is_initialized():
                        dist.destroy_process_group()
                        logger.debug("Destroyed torch distributed process group during cleanup")
                except Exception:
                    logger.debug("Failed to destroy torch distributed process group", exc_info=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    logger.debug("torch.cuda.ipc_collect failed", exc_info=True)
                if hasattr(torch.cuda, "reset_peak_memory_stats"):
                    try:
                        for idx in range(torch.cuda.device_count()):
                            torch.cuda.reset_peak_memory_stats(idx)
                    except Exception:
                        logger.debug("Failed to reset CUDA peak memory stats", exc_info=True)
        except Exception:
            pass

        gc.collect()
