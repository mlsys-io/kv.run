#!/usr/bin/env python3
"""
VLLMExecutor (YAML schema specific)

- Reads model config from spec.model.vllm / spec.model.source
- Reads sampling params and input prompts from spec.inference & spec.data
- Writes generation results to out_dir/responses.json

This rewrite follows vLLM's **LoRA Adapters** guidance:
- Per-request LoRA with `LoRARequest(name, adapter_id, adapter_path)` and `enable_lora=True`.
- Supports multiple runtime LoRA adapters in one call when the local vLLM build allows passing a list of `LoRARequest`.
- Still supports `apply=merge` via native merge APIs if present, otherwise offline-merge with PEFT.
- Optional `spec.model.vllm.release_after_run` to keep/release the LLM instance.
- Defensive compatibility for older vLLM builds (graceful fallbacks and clear errors).
"""

from __future__ import annotations

import logging
import os
import time
import gc
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

logger = logging.getLogger(__name__)

from .base_executor import Executor, ExecutionError
from .graph_templates import build_prompts_from_graph_template
from .checkpoint_utils import (
    download_and_unpack,
    detect_archive_format,
    select_extracted_subdir,
)

try:
    # vLLM is optional at import time; errors are raised in prepare()
    from vllm import LLM, SamplingParams  # type: ignore
    _HAS_VLLM = True
    try:
        # Modern LoRA request helper (per-request adapters)
        from vllm.lora.request import LoRARequest  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        LoRARequest = None  # type: ignore
except Exception:
    LLM = object  # type: ignore[misc,assignment]
    SamplingParams = object  # type: ignore[misc,assignment]
    _HAS_VLLM = False
    LoRARequest = None  # type: ignore


class VLLMExecutor(Executor):
    """Executor that runs text generation using vLLM based on a YAML spec."""

    name = "vllm"

    def __init__(self) -> None:
        super().__init__()
        self._llm: Optional[LLM] = None
        self._model_name: Optional[str] = None
        self._prompts: List[str] = []
        self._inf: Dict[str, Any] = {}
        self._adapter_specs: List[Dict[str, Any]] = []
        self._merged_adapters: List[str] = []
        self._runtime_specs: List[Dict[str, Any]] = []
        self._release_llm_after_run = True
        self._llm_kwargs: Dict[str, Any] = {}
        self._offline_model_dir: Optional[Path] = None

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #
    def prepare(self) -> None:  # type: ignore[override]
        if not _HAS_VLLM:
            raise ExecutionError("vLLM is not installed (`pip install vllm`).")

    def _ensure_llm(self, spec: Dict[str, Any]) -> None:
        if self._llm is not None:
            return

        model_src = (spec.get("model") or {}).get("source", {}) or {}
        ident = model_src.get("identifier") or os.getenv("VLLM_MODEL")
        if not ident:
            raise ExecutionError("spec.model.source.identifier (or VLLM_MODEL) is required.")

        adapter_specs = self._extract_adapter_specs(spec)
        vllm_cfg = (spec.get("model") or {}).get("vllm", {}) or {}

        kwargs: Dict[str, Any] = dict(
            model=ident,
            tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.9)),
            trust_remote_code=bool(vllm_cfg.get("trust_remote_code", False)),
        )
        # Per vLLM docs: enable LoRA at init time when using per-request adapters
        if adapter_specs:
            kwargs["enable_lora"] = True
        if "max_model_len" in vllm_cfg:
            kwargs["max_model_len"] = int(vllm_cfg["max_model_len"])
        if "dtype" in vllm_cfg:
            kwargs["dtype"] = str(vllm_cfg["dtype"])
        if "download_dir" in vllm_cfg:
            kwargs["download_dir"] = str(vllm_cfg["download_dir"])
        self._release_llm_after_run = bool(vllm_cfg.get("release_after_run", True))

        self._llm_kwargs = dict(kwargs)
        try:
            self._llm = LLM(**kwargs)  # type: ignore[call-arg]
        except TypeError as exc:
            # Some older builds don't accept enable_lora
            if "enable_lora" in kwargs:
                kwargs.pop("enable_lora", None)
                self._llm_kwargs = dict(kwargs)
                self._llm = LLM(**kwargs)  # type: ignore[call-arg]
            else:
                raise
        self._model_name = ident

    # --------------------------------------------------------------------- #
    # Data preparation
    # --------------------------------------------------------------------- #
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
            dataset = load_dataset(data_url, name=name, split=split)
            if shuffle:
                seed = int(data.get("seed", 42))
                buffer_size = data.get("buffer_size", None)
                dataset = dataset.shuffle(seed=seed) if buffer_size is None else dataset.shuffle(seed=seed, buffer_size=int(buffer_size))
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

        self._inf = spec.get("inference", {}) or {}
        system_prompt = self._inf.get("system_prompt")
        if system_prompt and append_system_prompt:
            self._prompts = [f"{system_prompt}\n{p}" for p in self._prompts]

    # --------------------------------------------------------------------- #
    # Execution
    # --------------------------------------------------------------------- #
    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:  # type: ignore[override]
        spec = (task or {}).get("spec") or {}
        self._ensure_llm(spec)
        self.prepare_data(spec)
        self._prepare_adapters(spec, out_dir)

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
            raise ExecutionError("No prompts prepared. Check spec.data configuration.")

        # Build per-request LoRA payload(s)
        lora_payload = self._build_lora_requests(out_dir)
        generate_kwargs: Dict[str, Any] = {}
        if lora_payload is not None:
            generate_kwargs["lora_request"] = lora_payload

        t0 = time.time()
        try:
            outputs = self._llm.generate(
                self._prompts,
                sampling_params=sampling,
                **generate_kwargs,
            )  # type: ignore[attr-defined]
        except TypeError as exc:
            # Some builds don't accept list/arg name for lora_request; try a single adapter fallback
            single = self._maybe_single_lora(lora_payload)
            if single is not None:
                outputs = self._llm.generate(
                    self._prompts,
                    sampling_params=sampling,
                    lora_request=single,
                )
            else:
                raise ExecutionError(
                    "Installed vLLM build does not accept per-request LoRA; upgrade vLLM or set adapter.apply='merge'."
                ) from exc
        latency = time.time() - t0

        items: List[Dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0

        for i, out in enumerate(outputs):
            out_outputs = getattr(out, "outputs", None)
            if not out_outputs:
                items.append({"index": i, "output": "", "finish_reason": None})
                continue
            best = out_outputs[0]
            text = getattr(best, "text", "") or ""
            finish_reason = getattr(best, "finish_reason", None)
            items.append({"index": i, "output": text, "finish_reason": finish_reason})
            prompt_token_ids = getattr(out, "prompt_token_ids", None) or []
            best_token_ids = getattr(best, "token_ids", None) or []
            prompt_tokens += len(prompt_token_ids)
            completion_tokens += len(best_token_ids)

        return {
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

    def cleanup_after_run(self) -> None:
        if self._release_llm_after_run:
            self._llm = None
        self._llm_kwargs = {}
        self._prompts = []
        self._adapter_specs = []
        self._runtime_specs = []
        self._merged_adapters = []
        if self._offline_model_dir and self._offline_model_dir.exists():
            shutil.rmtree(self._offline_model_dir, ignore_errors=True)
        self._offline_model_dir = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    # ------------------------------------------------------------------ #
    # Adapter utilities (per vLLM LoRA docs)
    # ------------------------------------------------------------------ #
    def _extract_adapter_specs(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._adapter_specs:
            return self._adapter_specs
        model_cfg = spec.get("model") or {}
        adapters = model_cfg.get("adapters") or []
        if not isinstance(adapters, list):
            raise ExecutionError("spec.model.adapters must be a list when provided")

        normalized: List[Dict[str, Any]] = []
        for idx, adapter in enumerate(adapters):
            if not isinstance(adapter, dict):
                raise ExecutionError("Each entry in spec.model.adapters must be a mapping")
            if str(adapter.get("type", "")).lower() != "lora":
                continue
            name = adapter.get("name") or f"lora_{idx}"
            apply_mode = str(adapter.get("apply", "runtime")).lower()
            source_path = adapter.get("path")
            source_url = adapter.get("url")
            adapter_id = adapter.get("id") or adapter.get("adapter_id") or name  # globally unique id
            archive_format = adapter.get("archive_format", "auto")
            headers = {str(k): str(v) for k, v in (adapter.get("headers") or {}).items()}
            scale = float(adapter.get("scale", 1.0))
            if not source_path and not source_url:
                raise ExecutionError(f"LoRA adapter '{name}' must provide either 'path' or 'url'")
            normalized.append(
                {
                    "name": str(name),
                    "id": str(adapter_id),
                    "apply": apply_mode,  # 'runtime' (per-request) or 'merge'
                    "path": source_path,
                    "url": source_url,
                    "headers": headers,
                    "archive_format": str(archive_format),
                    "scale": scale,
                }
            )
        self._adapter_specs = normalized
        return normalized

    def _prepare_adapters(self, spec: Dict[str, Any], out_dir: Path) -> None:
        # Split runtime vs merge upfront (runtime: handled at generate() via LoRARequest)
        runtime_specs: List[Dict[str, Any]] = []
        merge_specs: List[Dict[str, Any]] = []
        for a in self._adapter_specs:
            (merge_specs if a.get("apply") == "merge" else runtime_specs).append(a)
        self._runtime_specs = runtime_specs

        if merge_specs:
            # Try native merge first; otherwise offline merge
            if self._supports_native_merge():
                for a in merge_specs:
                    adapter_dir = self._resolve_adapter_directory(a, out_dir)
                    self._load_and_merge_native(a["name"], adapter_dir)
            else:
                self._offline_merge_adapters(merge_specs, spec, out_dir)

    def _supports_native_merge(self) -> bool:
        llm = self._llm
        if llm is None:
            return False
        return any(
            hasattr(llm, attr)
            for attr in ("merge_lora_weights", "merge_lora_adapter", "merge_lora")
        )

    def _offline_merge_adapters(
        self,
        adapters: List[Dict[str, Any]],
        spec: Dict[str, Any],
        out_dir: Path,
    ) -> None:
        if not adapters:
            return
        base_identifier = self._model_name
        if not base_identifier:
            model_src = (spec.get("model") or {}).get("source", {}) or {}
            base_identifier = model_src.get("identifier")
        if not base_identifier:
            raise ExecutionError("Unable to determine base model identifier for offline LoRA merge")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as exc:
            raise ExecutionError("Offline LoRA merge requires transformers and peft to be installed") from exc

        logger.info("Offline merging %d LoRA adapter(s) into %s", len(adapters), base_identifier)

        self._llm = None
        merge_dir = out_dir / "_merged_vllm_model"
        if merge_dir.exists():
            shutil.rmtree(merge_dir, ignore_errors=True)

        logger.info("Loading base model %s onto CPU for offline merge", base_identifier)
        model = AutoModelForCausalLM.from_pretrained(  # type: ignore[assignment]
            base_identifier,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

        for a in adapters:
            adapter_dir = self._resolve_adapter_directory(a, out_dir)
            logger.info("Merging LoRA adapter '%s' from %s", a.get("name"), adapter_dir)
            model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()  # type: ignore[assignment]

        logger.info("Saving merged model to %s", merge_dir)
        model.save_pretrained(merge_dir)
        try:
            logger.info("Saving tokenizer for merged model")
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_identifier, use_fast=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(base_identifier, use_fast=False)
            tokenizer.save_pretrained(merge_dir)
        except Exception as exc:
            logger.warning("Failed to save tokenizer for merged model: %s", exc)

        new_kwargs = dict(self._llm_kwargs)
        new_kwargs.pop("enable_lora", None)
        new_kwargs["model"] = str(merge_dir)
        self._offline_model_dir = merge_dir
        self._llm_kwargs = new_kwargs
        logger.info("Reloading vLLM from merged weights at %s", merge_dir)
        self._llm = LLM(**new_kwargs)  # type: ignore[call-arg]
        self._model_name = str(merge_dir)

    def _resolve_adapter_directory(self, adapter: Dict[str, Any], out_dir: Path) -> Path:
        name = adapter["name"]
        path_value = adapter.get("path")
        url_value = adapter.get("url")
        archive_format = adapter.get("archive_format", "auto")

        if path_value:
            path = Path(str(path_value)).expanduser()
            if not path.exists():
                raise ExecutionError(f"LoRA adapter path not found: {path}")
            return self._ensure_or_extract(path, out_dir, name, archive_format)
        # Remote download
        assert url_value is not None
        download_root = out_dir / "_lora_cache" / name
        download_root.mkdir(parents=True, exist_ok=True)
        load_cfg = {"url": url_value, "archive_format": archive_format, "headers": adapter.get("headers") or {}}
        extracted = download_and_unpack(load_cfg, download_root)
        return self._ensure_adapter_root(extracted)

    def _ensure_or_extract(self, p: Path, out_dir: Path, name: str, archive_format: str) -> Path:
        if p.is_dir():
            return self._ensure_adapter_root(p)
        # file -> archive
        target_root = out_dir / "_lora_cache" / name
        if target_root.exists():
            for child in target_root.iterdir():
                (child.unlink() if child.is_file() else shutil.rmtree(child, ignore_errors=True))
        target_root.mkdir(parents=True, exist_ok=True)
        fmt = detect_archive_format(archive_format, p.name)
        if fmt == "zip":
            import zipfile
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(target_root)
        elif fmt == "tar":
            import tarfile
            with tarfile.open(p, "r:*") as tf:
                tf.extractall(target_root)
        else:
            raise ExecutionError(f"Unsupported LoRA archive format for {p}; expected .zip/.tar or a directory")
        return self._ensure_adapter_root(select_extracted_subdir(target_root, None))

    def _ensure_adapter_root(self, path: Path) -> Path:
        if path.is_file():
            raise ExecutionError(f"Expected LoRA adapter directory but found file: {path}")
        if (path / "adapter_config.json").exists():
            return path
        candidates = list(path.rglob("adapter_config.json"))
        if not candidates:
            raise ExecutionError(f"Could not locate adapter_config.json under {path} for LoRA adapter")
        return candidates[0].parent

    # ---------------------------- Native merge ---------------------------- #
    def _load_and_merge_native(self, adapter_name: str, adapter_dir: Path) -> None:
        if self._llm is None:
            raise ExecutionError("vLLM not initialized before merging LoRA adapters")
        merge_fn = (
            getattr(self._llm, "merge_lora_weights", None)
            or getattr(self._llm, "merge_lora_adapter", None)
            or getattr(self._llm, "merge_lora", None)
        )
        load_fn = getattr(self._llm, "load_lora_adapter", None) or getattr(self._llm, "load_lora_adapters", None)
        if load_fn is None and not merge_fn:
            raise ExecutionError("vLLM merge_lora_* API not found; cannot apply LoRA adapter with apply=merge")
        if load_fn is not None:
            try:
                load_fn(adapter_name=adapter_name, adapter_path=str(adapter_dir))
            except TypeError:
                load_fn(str(adapter_dir), adapter_name)  # type: ignore[misc]
        try:
            merge_fn(adapter_name=adapter_name)  # type: ignore[arg-type]
        except TypeError:
            merge_fn(adapter_name)  # type: ignore[arg-type]
        unload_fn = getattr(self._llm, "unload_lora_adapter", None)
        if unload_fn is not None:
            try:
                unload_fn(adapter_name=adapter_name)  # type: ignore[arg-type]
            except TypeError:
                unload_fn(adapter_name)  # type: ignore[arg-type]

    # ------------------------- Per-request building ----------------------- #
    def _build_lora_requests(self, out_dir: Path) -> Optional[Any]:
        """Builds `lora_request` payload for vLLM.generate following the docs.
        - If multiple runtime adapters are configured, returns a list of LoRARequest
          when supported; otherwise, returns a single LoRARequest (first one).
        - If LoRARequest helper is unavailable, returns None and expects merge path.
        """
        if not self._runtime_specs:
            return None
        if LoRARequest is None:
            raise ExecutionError(
                "vLLM LoRARequest helper unavailable; upgrade vLLM or set adapter.apply='merge'"
            )

        requests: List[Any] = []
        for a in self._runtime_specs:
            adapter_dir = self._resolve_adapter_directory(a, out_dir)
            req = self._make_lora_request(a["name"], a["id"], adapter_dir, a.get("scale", 1.0))
            requests.append(req)

        # Prefer returning the full list (newer builds), otherwise single
        return requests if len(requests) > 1 else requests[0]

    def _maybe_single_lora(self, payload: Optional[Any]) -> Optional[Any]:
        if payload is None:
            return None
        if isinstance(payload, list) and payload:
            return payload[0]
        return payload

    def _make_lora_request(self, name: str, adapter_id: str, adapter_dir: Path, scale: float) -> Any:
        """Create LoRARequest(name, adapter_id, adapter_path) with fallbacks.
        Some builds use (name, path, scaling) or (name, adapter_id, path, scaling).
        We try common signatures in order.
        """
        path = str(adapter_dir)
        # Try (name, adapter_id, path)
        try:
            return LoRARequest(name, adapter_id, path)  # type: ignore[call-arg]
        except TypeError:
            pass
        # Try (name, adapter_id, path, scale)
        try:
            return LoRARequest(name, adapter_id, path, scale)  # type: ignore[call-arg]
        except TypeError:
            pass
        # Try legacy (name, path, scale)
        try:
            return LoRARequest(name, path, scale)  # type: ignore[call-arg]
        except TypeError as exc:
            raise ExecutionError(
                f"Unsupported LoRARequest signature for adapter '{name}': {exc}"
            ) from exc
