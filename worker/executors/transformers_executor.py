#!/usr/bin/env python3
"""
HFTransformersExecutor

- Uses Hugging Face Transformers to run LLM text generation.
- Mirrors the YAML-reading behavior of VLLMExecutor to ease swapping.
- Supports CPU/GPU execution with optional device_map/quantization.

YAML spec expectations (compatible with your existing schema):

spec:
  model:
    source:
      identifier: meta-llama/Llama-3-8b-instruct   # or local path
    transformers:                                  # optional
      device_map: auto | cuda | cpu                # default: auto if accelerate present else single device
      dtype: auto | float16 | bfloat16 | float32   # default: auto
      trust_remote_code: false                     # default: false
      low_cpu_mem_usage: true                      # default: true
      load_in_8bit: false                          # optional (requires bitsandbytes)
      load_in_4bit: false                          # optional (requires bitsandbytes)
      use_flash_attention_2: false                 # optional (if supported by model)
  inference:
    temperature: 0.7
    top_p: 0.95
    top_k: 50
    max_tokens: 512                                # alias of max_new_tokens
    max_new_tokens: 512                            # takes precedence if provided
    do_sample: true                                # optional, will be inferred from temperature/top_p/top_k if missing
    repetition_penalty: 1.0                        # transformers-specific
    stop: ["\n\nUser:", "</s>"]                 # optional stop strings (post-process truncation)
  data:
    type: dataset | list
    # if dataset:
    url: glue                                      # HF hub name or path
    name: sst2                                     # optional config name
    split: validation                              # default train
    column: text                                   # default text
    shuffle: false                                 # optional
    seed: 42                                       # optional
    buffer_size: 1000                              # optional
    # if list:
    items: ["Hello, world!"]

Output file: out_dir/responses.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from datasets import load_dataset

from .base_executor import Executor, ExecutionError
from .graph_templates import build_prompts_from_graph_template

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
    )
    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - import-time optional dependency
    torch = None  # type: ignore
    AutoModelForCausalLM = object  # type: ignore[misc,assignment]
    AutoTokenizer = object  # type: ignore[misc,assignment]
    GenerationConfig = object  # type: ignore[misc,assignment]
    _HAS_TRANSFORMERS = False


class HFTransformersExecutor(Executor):
    """Executor that runs text generation via Hugging Face Transformers."""

    name = "transformers"

    def __init__(self) -> None:
        super().__init__()
        self._tok = None
        self._model = None
        self._device: Optional[str] = None
        self._model_name: Optional[str] = None
        self._prompts: List[str] = []
        self._inf: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def prepare(self) -> None:  # type: ignore[override]
        if not _HAS_TRANSFORMERS:
            raise ExecutionError("transformers/torch is not installed (`pip install transformers torch`).")

    def _pick_device(self, cfg: Dict[str, Any]) -> str:
        # Explicit device_map overrides simple device if provided
        device_map = cfg.get("device_map")
        if device_map in {"auto", "balanced", "balanced_low_0"}:  # acceptable values for accelerate
            return "auto"
        # Simple single-device selection
        if device_map == "cuda":
            if torch and torch.cuda.is_available():
                return "cuda"
            raise ExecutionError("Requested CUDA but no GPU is available.")
        if device_map == "cpu":
            return "cpu"
        # Default preference: CUDA if available
        return "cuda" if (torch and torch.cuda.is_available()) else "cpu"

    def _to_torch_dtype(self, s: Optional[str]):
        if not s or s == "auto":
            return "auto"
        s = str(s).lower()
        if s in {"float16", "fp16"}:
            return torch.float16
        if s in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if s in {"float32", "fp32"}:
            return torch.float32
        raise ExecutionError(f"Unsupported dtype: {s}")

    def _ensure_model(self, spec: Dict[str, Any]) -> None:
        if self._model is not None and self._tok is not None:
            return

        model_src = (spec.get("model") or {}).get("source", {}) or {}
        ident = model_src.get("identifier") or os.getenv("HF_MODEL")
        if not ident:
            raise ExecutionError("spec.model.source.identifier (or HF_MODEL) is required.")

        tcfg = (spec.get("model") or {}).get("transformers", {}) or {}
        device = self._pick_device(tcfg)
        dtype = self._to_torch_dtype(tcfg.get("dtype", "auto"))
        trust_remote_code = bool(tcfg.get("trust_remote_code", False))

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": bool(tcfg.get("low_cpu_mem_usage", True)),
        }
        # quantization options (optional, require bitsandbytes)
        if bool(tcfg.get("load_in_8bit", False)):
            load_kwargs["load_in_8bit"] = True
        if bool(tcfg.get("load_in_4bit", False)):
            load_kwargs["load_in_4bit"] = True
        if tcfg.get("use_flash_attention_2") is True:
            load_kwargs["use_flash_attention_2"] = True

        # Device placement
        if device == "auto":
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = dtype
        else:
            load_kwargs["torch_dtype"] = dtype

        try:
            self._tok = AutoTokenizer.from_pretrained(ident, use_fast=True, trust_remote_code=trust_remote_code)
            # Ensure we have a pad token for batch generation
            if self._tok.pad_token_id is None:
                # Fallback to eos token, common for decoder-only LMs
                self._tok.pad_token = self._tok.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(ident, **load_kwargs)
            if device != "auto":  # single-device path
                self._model.to(device)
            self._model.eval()
        except Exception as e:
            raise ExecutionError(f"Failed to load model/tokenizer: {e}")

        self._device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self._model_name = ident

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    def prepare_data(self, spec: Dict[str, Any]) -> None:
        data = spec.get("data") or {}
        if not data:
            raise ExecutionError("spec.data is required.")

        dtype = data.get("type")
        append_system_prompt = True
        if dtype == "dataset":
            data_url = data.get("url")
            if not data_url:
                raise ExecutionError("spec.data.url is required for type == 'dataset'.")
            name = data.get("name", None)
            split = data.get("split", "train")
            shuffle = bool(data.get("shuffle", False))
            trust_remote_code = data.get("trust_remote_code")
            if trust_remote_code is None:
                trust_remote_code = True
            revision = data.get("revision")
            dataset = load_dataset(
                data_url,
                name=name,
                split=split,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
            if shuffle:
                seed = int(data.get("seed", 42))
                buffer_size = data.get("buffer_size", None)
                dataset = dataset.shuffle(seed=seed) if buffer_size is None else dataset.shuffle(seed=seed, buffer_size=int(buffer_size))
            dataset = self._maybe_apply_dataset_shard(dataset, spec)
            column = data.get("column", "text")
            if column not in dataset.column_names:
                raise ExecutionError(f"Column '{column}' not found in dataset. Available: {dataset.column_names}")
            self._prompts = [str(x) for x in dataset[column]]
        elif dtype == "list":
            items = data.get("items", [])
            if not isinstance(items, list) or any(not isinstance(x, str) for x in items):
                raise ExecutionError("spec.data.items must be a list of strings for type == 'list'.")
            self._prompts = items
        elif dtype == "graph_template":
            self._prompts = build_prompts_from_graph_template(data, spec)
            template_cfg = data.get("template") or {}
            append_system_prompt = bool(template_cfg.get("append_system_prompt", False))
        else:
            raise ExecutionError(f"Unsupported spec.data.type: {dtype!r}")

        self._inf = spec.get("inference", {}) or {}
        system_prompt = self._inf.get("system_prompt")
        if system_prompt and append_system_prompt:
            self._prompts = [f"{system_prompt}\n{p}" for p in self._prompts]

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def _normalize_stops(self, stop_val: Any) -> List[str]:
        if stop_val is None:
            return []
        if isinstance(stop_val, str):
            return [stop_val]
        if isinstance(stop_val, Sequence):
            return [str(x) for x in stop_val]
        return []

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:  # type: ignore[override]
        spec = (task or {}).get("spec") or {}
        self._ensure_model(spec)
        self.prepare_data(spec)

        # Generation configuration
        max_new_tokens = int(self._inf.get("max_new_tokens", self._inf.get("max_tokens", 512)))
        temperature = float(self._inf.get("temperature", 0.7))
        top_p = float(self._inf.get("top_p", 0.95))
        top_k = int(self._inf.get("top_k", 50))
        do_sample = self._inf.get("do_sample")
        if do_sample is None:
            # Infer sampling: enable when temperature != 0 or top_p<1 or top_k>0
            do_sample = (temperature > 0.0) or (top_p < 1.0) or (top_k > 0)
        repetition_penalty = float(self._inf.get("repetition_penalty", 1.0))
        stops = self._normalize_stops(self._inf.get("stop"))

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k >= 0 else 0,
            do_sample=bool(do_sample),
            repetition_penalty=repetition_penalty,
            pad_token_id=self._tok.pad_token_id,  # type: ignore[attr-defined]
            eos_token_id=self._tok.eos_token_id,  # type: ignore[attr-defined]
        )

        if not self._prompts:
            raise ExecutionError("No prompts prepared. Check spec.data configuration.")

        # Tokenize batch
        enc = self._tok(
            self._prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        enc = {k: v.to(device) for k, v in enc.items()}  # type: ignore[arg-type]

        with torch.no_grad():
            outputs = self._model.generate(**enc, generation_config=gen_cfg)  # type: ignore[attr-defined]

        items: List[Dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0

        # For each sequence in the batch, split prompt vs generated part
        input_ids = enc["input_ids"]
        for i in range(outputs.shape[0]):
            input_len = int(input_ids[i].shape[0])
            seq = outputs[i]
            gen_part = seq[input_len:]
            text = self._tok.decode(gen_part, skip_special_tokens=True)

            # Apply simple stop-string truncation on decoded text
            if stops:
                cut = len(text)
                for s in stops:
                    idx = text.find(s)
                    if idx != -1:
                        cut = min(cut, idx)
                text = text[:cut]

            items.append({"index": i, "output": text, "finish_reason": None})
            prompt_tokens += int(input_len)
            completion_tokens += int(gen_part.shape[0])

        result: Dict[str, Any] = {
            "ok": True,
            "model": self._model_name,
            "items": items,
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens),
                "num_requests": len(self._prompts),
            },
        }

        return result

    def _maybe_apply_dataset_shard(self, dataset, spec: Dict[str, Any]):
        """Apply dataset.shard when spec includes shard metadata."""
        shard_cfg = spec.get("shard") if isinstance(spec, dict) else None
        if not shard_cfg:
            return dataset
        try:
            total = int(shard_cfg.get("total", 1))
            index = int(shard_cfg.get("index", 0))
        except Exception as exc:  # pragma: no cover - validation
            raise ExecutionError(f"Invalid shard metadata: {exc}") from exc
        if total <= 1:
            return dataset
        contiguous = bool(shard_cfg.get("contiguous", True))
        try:
            return dataset.shard(num_shards=total, index=index, contiguous=contiguous)
        except Exception as exc:  # pragma: no cover - defensive
            raise ExecutionError(f"Failed to shard dataset ({index}/{total}): {exc}")
