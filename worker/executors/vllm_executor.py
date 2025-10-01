#!/usr/bin/env python3
"""
VLLMExecutor (YAML schema specific)

- Reads model config from spec.model.vllm / spec.model.source
- Reads sampling params and input prompts from spec.inference & spec.data
- Writes generation results to out_dir/responses.json
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

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
    try:  # LoRA helpers are optional prior to vLLM 0.3
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
        self._runtime_requests: List[Any] = []

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

        adapter_specs = self._extract_adapter_specs(spec)
        vllm_cfg = (spec.get("model") or {}).get("vllm", {}) or {}
        # Map config values with sane defaults and explicit casting
        kwargs: Dict[str, Any] = dict(
            model=ident,
            tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.9)),
            trust_remote_code=bool(vllm_cfg.get("trust_remote_code", False)),
        )
        if adapter_specs:
            kwargs["enable_lora"] = True
        if "max_model_len" in vllm_cfg:
            kwargs["max_model_len"] = int(vllm_cfg["max_model_len"])
        if "dtype" in vllm_cfg:
            kwargs["dtype"] = str(vllm_cfg["dtype"])
        if "download_dir" in vllm_cfg:
            kwargs["download_dir"] = str(vllm_cfg["download_dir"])

        try:
            self._llm = LLM(**kwargs)  # type: ignore[call-arg]
        except TypeError as exc:
            if "enable_lora" in kwargs:
                kwargs.pop("enable_lora", None)
                self._llm = LLM(**kwargs)  # type: ignore[call-arg]
            else:
                raise
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
        self._prepare_adapters(spec, out_dir)

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
        generate_kwargs: Dict[str, Any] = {}
        lora_request = self._build_runtime_lora_request()
        if lora_request is not None:
            generate_kwargs["lora_request"] = lora_request

        try:
            outputs = self._llm.generate(
                self._prompts,
                sampling_params=sampling,
                **generate_kwargs,
            )  # type: ignore[attr-defined]
        except TypeError as exc:
            if generate_kwargs:
                raise ExecutionError(
                    "Installed vLLM build does not accept runtime LoRA requests via generate(); "
                    "upgrade vLLM or switch adapter.apply to 'merge'."
                ) from exc
            raise
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

    # ------------------------------------------------------------------ #
    # Adapter utilities
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

            adapter_type = str(adapter.get("type", "")).lower()
            if adapter_type != "lora":
                continue

            name = adapter.get("name") or f"lora_{idx}"
            apply_mode = str(adapter.get("apply", "merge")).lower()
            source_path = adapter.get("path")
            source_url = adapter.get("url")
            archive_format = adapter.get("archive_format", "auto")
            headers = adapter.get("headers") or {}

            if not source_path and not source_url:
                raise ExecutionError(
                    f"LoRA adapter '{name}' must provide either 'path' or 'url'"
                )

            normalized.append(
                {
                    "name": str(name),
                    "apply": apply_mode,
                    "path": source_path,
                    "url": source_url,
                    "headers": {str(k): str(v) for k, v in headers.items()},
                    "archive_format": str(archive_format),
                }
            )

        self._adapter_specs = normalized
        return normalized

    def _prepare_adapters(self, spec: Dict[str, Any], out_dir: Path) -> None:
        if not self._adapter_specs:
            return
        if self._llm is None:
            raise ExecutionError("vLLM executor not initialized before adapter preparation")

        ready_requests: List[Any] = []
        for adapter in self._adapter_specs:
            apply_mode = adapter.get("apply", "merge")
            if apply_mode not in {"merge", "runtime", "apply", "activate"}:
                raise ExecutionError(
                    f"Unsupported LoRA apply mode '{apply_mode}' (expected 'merge' or 'runtime')"
                )

            adapter_dir = self._resolve_adapter_directory(adapter, out_dir)
            adapter_name = adapter["name"]

            if apply_mode == "merge":
                self._load_and_merge_adapter(adapter_name, adapter_dir)
                continue

            # runtime / apply / activate -> keep request for generation
            request = self._load_runtime_adapter(adapter_name, adapter_dir)
            if request is not None:
                ready_requests.append(request)
        self._runtime_requests = ready_requests

    def _resolve_adapter_directory(self, adapter: Dict[str, Any], out_dir: Path) -> Path:
        name = adapter["name"]
        path_value = adapter.get("path")
        url_value = adapter.get("url")
        archive_format = adapter.get("archive_format", "auto")

        if path_value:
            path = Path(str(path_value)).expanduser()
            if not path.exists():
                raise ExecutionError(f"LoRA adapter path not found: {path}")
            if path.is_file():
                return self._extract_archive_file(path, out_dir, name, archive_format)
            return self._ensure_adapter_root(path)

        # Remote download
        assert url_value is not None  # guarded earlier
        download_root = out_dir / "_lora_cache" / name
        download_root.mkdir(parents=True, exist_ok=True)
        load_cfg = {
            "url": url_value,
            "archive_format": archive_format,
            "headers": adapter.get("headers") or {},
        }
        extracted = download_and_unpack(load_cfg, download_root)
        return self._ensure_adapter_root(extracted)

    def _extract_archive_file(
        self,
        archive_path: Path,
        out_dir: Path,
        name: str,
        archive_format: str,
    ) -> Path:
        target_root = out_dir / "_lora_cache" / name
        if target_root.exists():
            for child in target_root.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    import shutil

                    shutil.rmtree(child, ignore_errors=True)
        target_root.mkdir(parents=True, exist_ok=True)

        fmt = detect_archive_format(archive_format, archive_path.name)
        if fmt == "zip":
            import zipfile

            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(target_root)
        elif fmt == "tar":
            import tarfile

            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(target_root)
        else:
            raise ExecutionError(
                f"Unsupported LoRA archive format for {archive_path}; expected .zip/.tar or a directory"
            )

        return self._ensure_adapter_root(select_extracted_subdir(target_root, None))

    def _ensure_adapter_root(self, path: Path) -> Path:
        if path.is_file():
            # Some workflows produce adapter.json; require directory
            raise ExecutionError(
                f"Expected LoRA adapter directory but found file: {path}"
            )
        if (path / "adapter_config.json").exists():
            return path
        candidates = list(path.rglob("adapter_config.json"))
        if not candidates:
            raise ExecutionError(
                f"Could not locate adapter_config.json under {path} for LoRA adapter"
            )
        return candidates[0].parent

    def _load_and_merge_adapter(self, adapter_name: str, adapter_dir: Path) -> None:
        if adapter_name in self._merged_adapters:
            return
        if self._llm is None:
            raise ExecutionError("vLLM not initialized before merging LoRA adapters")

        load_fn = getattr(self._llm, "load_lora_adapter", None)
        if load_fn is None and LoRARequest is not None and hasattr(self._llm, "load_lora_adapters"):
            load_fn = self._llm.load_lora_adapters

        if load_fn is None:
            raise ExecutionError(
                "Installed vLLM build does not expose load_lora_adapter(s); upgrade to a version with LoRA support."
            )

        self._invoke_load_lora(load_fn, adapter_name, adapter_dir)

        merge_fn = (
            getattr(self._llm, "merge_lora_weights", None)
            or getattr(self._llm, "merge_lora_adapter", None)
            or getattr(self._llm, "merge_lora", None)
        )
        if merge_fn is None:
            raise ExecutionError(
                "vLLM merge_lora_* API not found; cannot apply LoRA adapter with apply=merge"
            )

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

        self._merged_adapters.append(adapter_name)

    def _load_runtime_adapter(self, adapter_name: str, adapter_dir: Path) -> Optional[Any]:
        if self._llm is None:
            raise ExecutionError("vLLM not initialized before loading LoRA adapters")

        if LoRARequest is None:
            raise ExecutionError(
                "vLLM LoRARequest helper unavailable; unable to keep adapter active at runtime"
            )

        load_fn = getattr(self._llm, "load_lora_adapter", None)
        if load_fn is None and hasattr(self._llm, "load_lora_adapters"):
            load_fn = self._llm.load_lora_adapters
        if load_fn is None:
            raise ExecutionError(
                "Installed vLLM build lacks load_lora_adapter(s); cannot activate LoRA runtime adapter"
            )

        self._invoke_load_lora(load_fn, adapter_name, adapter_dir)
        return LoRARequest(adapter_name, str(adapter_dir), 1.0)  # type: ignore[call-arg]

    def _invoke_load_lora(self, load_fn: Any, adapter_name: str, adapter_dir: Path) -> None:
        try:
            load_fn(adapter_name=adapter_name, adapter_path=str(adapter_dir))
        except TypeError:
            try:
                load_fn(str(adapter_dir), adapter_name)  # type: ignore[misc]
            except TypeError as exc:
                raise ExecutionError(
                    f"Unsupported load_lora_adapter signature for adapter '{adapter_name}': {exc}"
                ) from exc

    def _build_runtime_lora_request(self) -> Optional[Any]:
        requests = getattr(self, "_runtime_requests", None)
        if not requests:
            return None
        if len(requests) == 1:
            return requests[0]
        raise ExecutionError("Multiple runtime LoRA adapters are not supported by this executor")
