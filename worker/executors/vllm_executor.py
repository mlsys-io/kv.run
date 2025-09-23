#!/usr/bin/env python3
"""
VLLMExecutor (YAML schema specific)

- Reads model config from spec.model.vllm / spec.model.source
- Reads sampling params and input prompts from spec.inference & spec.data
- Writes generation results to out_dir/responses.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from .base_executor import Executor, ExecutionError
from .graph_templates import build_prompts_from_graph_template

try:
    # vLLM is optional at import time; errors are raised in prepare()
    from vllm import LLM, SamplingParams  # type: ignore
    _HAS_VLLM = True
except Exception:
    LLM = object  # type: ignore[misc,assignment]
    SamplingParams = object  # type: ignore[misc,assignment]
    _HAS_VLLM = False

class VLLMExecutor(Executor):
    """Executor that runs text generation using vLLM based on a YAML spec."""

    name = "vllm"

    def __init__(self) -> None:
        super().__init__()
        self._llm: Optional[LLM] = None
        self._model_name: Optional[str] = None
        self._prompts: List[str] = []
        self._inf: Dict[str, Any] = {}

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #
    def prepare(self) -> None:  # type: ignore[override]
        """Validate runtime prerequisites (vLLM)."""
        if not _HAS_VLLM:
            raise ExecutionError("vLLM is not installed (`pip install vllm`).")

    def _ensure_llm(self, spec: Dict[str, Any]) -> None:
        """Instantiate the LLM lazily from spec.model.*."""
        if self._llm is not None:
            return

        model_src = (spec.get("model") or {}).get("source", {}) or {}
        ident = model_src.get("identifier") or os.getenv("VLLM_MODEL")
        if not ident:
            raise ExecutionError("spec.model.source.identifier (or VLLM_MODEL) is required.")

        vllm_cfg = (spec.get("model") or {}).get("vllm", {}) or {}
        # Map config values with sane defaults and explicit casting
        kwargs: Dict[str, Any] = dict(
            model=ident,
            tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.9)),
            trust_remote_code=bool(vllm_cfg.get("trust_remote_code", False)),
        )
        if "max_model_len" in vllm_cfg:
            kwargs["max_model_len"] = int(vllm_cfg["max_model_len"])
        if "dtype" in vllm_cfg:
            kwargs["dtype"] = str(vllm_cfg["dtype"])
        if "download_dir" in vllm_cfg:
            kwargs["download_dir"] = str(vllm_cfg["download_dir"])

        self._llm = LLM(**kwargs)  # type: ignore[call-arg]
        self._model_name = ident

    # --------------------------------------------------------------------- #
    # Data preparation
    # --------------------------------------------------------------------- #
    def prepare_data(self, spec: Dict[str, Any]) -> None:
        """
        Build the list of prompts to feed into the model.

        Supported sources:
          - spec.data.type == "dataset": load via Hugging Face datasets
          - spec.data.type == "list":   use a provided list of strings
        """
        data = spec.get("data") or {}
        if not data:
            raise ExecutionError("spec.data is required.")

        dtype = data.get("type")
        append_system_prompt = True
        if dtype == "dataset":
            # Load dataset from URL / HF identifier
            data_url = data.get("url")
            if not data_url:
                raise ExecutionError("spec.data.url is required for type == 'dataset'.")

            name = data.get("name", None)
            split = data.get("split", "train")
            shuffle = bool(data.get("shuffle", False))

            # load_dataset accepts path_or_name + optional name + split
            dataset = load_dataset(data_url, name=name, split=split)

            if shuffle:
                seed = int(data.get("seed", 42))
                buffer_size = data.get("buffer_size", None)
                if buffer_size is None:
                    dataset = dataset.shuffle(seed=seed)
                else:
                    dataset = dataset.shuffle(seed=seed, buffer_size=int(buffer_size))

            column = data.get("column", "text")
            if column not in dataset.column_names:
                raise ExecutionError(
                    f"Column '{column}' not found in dataset. Available: {dataset.column_names}"
                )
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

        # Inference params are copied as-is (validated where used)
        self._inf = spec.get("inference", {}) or {}
        system_prompt = self._inf.get("system_prompt")
        if system_prompt and append_system_prompt:
            self._prompts = [f"{system_prompt}\n{p}" for p in self._prompts]

    # --------------------------------------------------------------------- #
    # Execution
    # --------------------------------------------------------------------- #
    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:  # type: ignore[override]
        """
        Execute text generation with vLLM and persist a JSON result file.

        Returns:
            A dictionary containing success flag, model name, items, and usage stats.
        """
        spec = (task or {}).get("spec") or {}
        self._ensure_llm(spec)
        self.prepare_data(spec)

        # Build sampling params (defensive casts + defaults)
        sampling = SamplingParams(  # type: ignore[call-arg]
            temperature=float(self._inf.get("temperature", 0.7)),
            top_p=float(self._inf.get("top_p", 0.95)),
            top_k=int(self._inf.get("top_k", -1)),
            max_tokens=int(self._inf.get("max_tokens", 512)),
            presence_penalty=float(self._inf.get("presence_penalty", 0.0)),
            frequency_penalty=float(self._inf.get("frequency_penalty", 0.0)),
            stop=self._inf.get("stop"),
        )

        if not self._prompts:
            # Allow empty input but fail fast with a clear message
            raise ExecutionError("No prompts prepared. Check spec.data configuration.")

        t0 = time.time()
        # vLLM returns a list of RequestOutput objects
        outputs = self._llm.generate(self._prompts, sampling_params=sampling)  # type: ignore[attr-defined]
        latency = time.time() - t0

        items: List[Dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0

        for i, out in enumerate(outputs):
            # Defensive guards for missing attributes
            out_outputs = getattr(out, "outputs", None)
            if not out_outputs:
                items.append({"index": i, "output": "", "finish_reason": None})
                continue

            best = out_outputs[0]
            text = getattr(best, "text", "") or ""
            finish_reason = getattr(best, "finish_reason", None)

            items.append(
                {
                    "index": i,
                    "output": text,
                    "finish_reason": finish_reason,
                }
            )

            prompt_token_ids = getattr(out, "prompt_token_ids", None) or []
            best_token_ids = getattr(best, "token_ids", None) or []
            prompt_tokens += len(prompt_token_ids)
            completion_tokens += len(best_token_ids)

        result: Dict[str, Any] = {
            "ok": True,
            "model": self._model_name,
            "items": items,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency_sec": latency,
                "num_requests": len(self._prompts),
            },
        }

        return result
