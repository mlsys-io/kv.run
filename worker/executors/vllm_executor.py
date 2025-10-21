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

import copy
import logging
import os
import time
import gc
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from datasets import load_dataset

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

logger = logging.getLogger(__name__)

from .base_executor import Executor, ExecutionError
from .graph_templates import build_prompts_from_graph_template
from .checkpoint_utils import (
    download_and_unpack,
    detect_archive_format,
    resolve_checkpoint_load,
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
        self._batched_prompts: List[str] = []
        self._prompt_owners: List[str] = []
        self._adapter_specs: List[Dict[str, Any]] = []
        self._merged_adapters: List[str] = []
        self._runtime_specs: List[Dict[str, Any]] = []
        self._release_llm_after_run = True
        self._llm_kwargs: Dict[str, Any] = {}
        self._offline_model_dir: Optional[Path] = None
        self._base_inference: Dict[str, Any] = {}

    @staticmethod
    def _detect_available_gpus() -> int:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis:
            candidates = [dev.strip() for dev in str(vis).split(",") if dev.strip()]
            if candidates:
                return len(candidates)
        try:
            if torch is not None and torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            pass
        return 1

    @staticmethod
    def _compute_safe_utilization(requested: float) -> Tuple[float, Optional[float]]:
        if torch is None or not torch.cuda.is_available():
            return requested, None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            return requested, None
        if total_bytes <= 0:
            return requested, None
        free_ratio = float(free_bytes) / float(total_bytes)
        safety_margin = 0.05  # keep at least 5% headroom
        min_util_floor = 0.30
        max_util = free_ratio - safety_margin
        max_util = min(0.98, max_util)
        adjusted = requested
        if max_util <= 0:
            adjusted = max(min_util_floor, requested * 0.8)
        else:
            adjusted = min(requested, max_util)
            if adjusted < min_util_floor and max_util >= min_util_floor:
                adjusted = min_util_floor
            elif adjusted <= 0:
                adjusted = max(min_util_floor, max_util * 0.8)
        return adjusted, free_ratio

    @staticmethod
    def _requested_gpu_count(spec: Dict[str, Any]) -> Optional[int]:
        resources = spec.get("resources") or {}
        hardware = resources.get("hardware") or {}
        gpu_cfg = hardware.get("gpu")
        if isinstance(gpu_cfg, dict):
            count = gpu_cfg.get("count")
            if count is not None:
                try:
                    value = int(count)
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    return None
        return None

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
        checkpoint_cfg = ((spec.get("checkpoint") or {}).get("load") or {})
        load_cfg = ((spec.get("checkpoint") or {}).get("load") or {})

        requested_tp = vllm_cfg.get("tensor_parallel_size")
        requested_gpu_count = self._requested_gpu_count(spec)
        try:
            tensor_parallel_size = int(requested_tp) if requested_tp is not None else 0
        except Exception:
            tensor_parallel_size = 0
        if tensor_parallel_size <= 0:
            available_gpus = self._detect_available_gpus()
            if requested_gpu_count is not None:
                tensor_parallel_size = max(1, min(available_gpus, requested_gpu_count))
            else:
                tensor_parallel_size = max(1, available_gpus)
        else:
            if requested_gpu_count is not None:
                tensor_parallel_size = max(1, min(tensor_parallel_size, requested_gpu_count))
        if adapter_specs and tensor_parallel_size > 1:
            logger.info(
                "LoRA adapters detected; forcing tensor_parallel_size=1 (was %s) to avoid multi-GPU merging.",
                tensor_parallel_size,
            )
            tensor_parallel_size = 1

        local_checkpoint_dir: Optional[Path] = None
        if checkpoint_cfg:
            load_cfg_local = dict(checkpoint_cfg)
            load_cfg_local["_logger"] = logger
            cfg_type = str(load_cfg_local.get("type", "http")).lower()
            if cfg_type == "http" and not (
                load_cfg_local.get("path") or load_cfg_local.get("taskId") or load_cfg_local.get("task_id")
            ):
                local_checkpoint_dir = None
            else:
                if cfg_type == "http":
                    load_cfg_local["type"] = "local"
                try:
                    candidate = resolve_checkpoint_load(load_cfg_local, Path(tempfile.gettempdir()), logger=logger)
                    candidate_path = Path(candidate) if candidate else None
                    if candidate_path and candidate_path.exists() and candidate_path.is_dir():
                        local_checkpoint_dir = candidate_path
                        logger.info("Resolved model checkpoint locally: %s", local_checkpoint_dir)
                except Exception as exc:
                    logger.debug("Local checkpoint resolution failed: %s", exc)
                    local_checkpoint_dir = None

        if local_checkpoint_dir is not None:
            logger.info("Initializing vLLM from local checkpoint directory %s", local_checkpoint_dir)
        else:
            logger.info("Initializing vLLM from identifier %s", ident)

        requested_util = float(vllm_cfg.get("gpu_memory_utilization", 0.9))

        kwargs_base: Dict[str, Any] = dict(
            model=str(local_checkpoint_dir or ident),
            trust_remote_code=bool(vllm_cfg.get("trust_remote_code", False)),
        )
        # Per vLLM docs: enable LoRA at init time when using per-request adapters
        if adapter_specs:
            kwargs_base["enable_lora"] = True
        if "max_model_len" in vllm_cfg:
            kwargs_base["max_model_len"] = int(vllm_cfg["max_model_len"])
        if "dtype" in vllm_cfg:
            kwargs_base["dtype"] = str(vllm_cfg["dtype"])
        if "download_dir" in vllm_cfg:
            kwargs_base["download_dir"] = str(vllm_cfg["download_dir"])
        self._release_llm_after_run = bool(vllm_cfg.get("release_after_run", True))

        tp_candidates: List[int] = []
        seen_tp: Set[int] = set()
        initial_tp = max(1, tensor_parallel_size)
        if initial_tp not in seen_tp:
            tp_candidates.append(initial_tp)
            seen_tp.add(initial_tp)
        if initial_tp != 1 and 1 not in seen_tp:
            tp_candidates.append(1)
            seen_tp.add(1)

        last_exc: Optional[Exception] = None
        success = False
        chosen_kwargs: Dict[str, Any] = {}

        for tp_idx, tp_value in enumerate(tp_candidates, start=1):
            kwargs = dict(kwargs_base)
            kwargs["tensor_parallel_size"] = tp_value

            safe_util, free_ratio = self._compute_safe_utilization(requested_util)
            if safe_util < requested_util - 1e-3:
                if free_ratio is not None:
                    logger.warning(
                        "Requested gpu_memory_utilization=%.3f but only %.2f%% of GPU memory is free; "
                        "reducing utilization to %.3f",
                        requested_util,
                        free_ratio * 100.0,
                        safe_util,
                    )
                else:
                    logger.warning(
                        "Requested gpu_memory_utilization=%.3f but available GPU memory is constrained; "
                        "reducing utilization to %.3f",
                        requested_util,
                        safe_util,
                    )

            util_candidates: List[float] = []
            tried_utils: Set[float] = set()
            util_candidates.append(safe_util)
            for delta in (0.05, 0.1, 0.15):
                candidate = max(0.3, safe_util - delta)
                if candidate < util_candidates[0] - 1e-3 and candidate not in util_candidates:
                    util_candidates.append(candidate)

            for util_idx, util in enumerate(util_candidates, start=1):
                if util in tried_utils:
                    continue
                tried_utils.add(util)

                attempt_kwargs = dict(kwargs)
                attempt_kwargs["gpu_memory_utilization"] = util
                self._llm_kwargs = dict(attempt_kwargs)
                logger.info(
                    "Initializing vLLM (TP candidate %d/%d, attempt %d/%d) "
                    "with tensor_parallel_size=%d, gpu_memory_utilization=%.3f",
                    tp_idx,
                    len(tp_candidates),
                    util_idx,
                    len(util_candidates),
                    tp_value,
                    util,
                )
                try:
                    self._llm = LLM(**attempt_kwargs)  # type: ignore[call-arg]
                    chosen_kwargs = dict(attempt_kwargs)
                    success = True
                    break
                except TypeError as exc:
                    last_exc = exc
                    if "enable_lora" in attempt_kwargs:
                        attempt_kwargs.pop("enable_lora", None)
                        self._llm_kwargs = dict(attempt_kwargs)
                        try:
                            self._llm = LLM(**attempt_kwargs)  # type: ignore[call-arg]
                            chosen_kwargs = dict(attempt_kwargs)
                            success = True
                            break
                        except Exception as inner_exc:
                            last_exc = inner_exc
                            msg_inner = str(inner_exc).lower()
                            if "memory profiling" in msg_inner and tp_value != 1:
                                logger.warning(
                                    "vLLM initialization failed due to memory profiling "
                                    "under tensor_parallel_size=%d; retrying with smaller parallelism.",
                                    tp_value,
                                )
                                break
                            continue
                    msg = str(exc).lower()
                    if "memory profiling" in msg and tp_value != 1:
                        logger.warning(
                            "vLLM initialization failed due to memory profiling under tensor_parallel_size=%d; "
                            "retrying with smaller parallelism.",
                            tp_value,
                        )
                        break
                    continue
                except (ValueError, RuntimeError) as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    if "memory profiling" in msg and tp_value != 1:
                        logger.warning(
                            "vLLM initialization encountered memory profiling race under tensor_parallel_size=%d; "
                            "retrying with smaller parallelism.",
                            tp_value,
                        )
                        break
                    logger.warning(
                        "vLLM initialization attempt failed (tp=%d, util=%.3f): %s",
                        tp_value,
                        util,
                        exc,
                    )
                    continue

            if success:
                break

        if not success or self._llm is None:
            message = (
                f"Failed to initialize vLLM after trying tensor_parallel_size candidates {tp_candidates} "
                f"and gpu_memory_utilization adjustments (last error: {last_exc})"
            )
            raise ExecutionError(message)

        self._llm_kwargs = chosen_kwargs
        self._model_name = ident
        last_exc: Optional[Exception] = None
        for util in util_candidates:
            if util in tried_utils:
                continue
            tried_utils.add(util)
            kwargs["gpu_memory_utilization"] = util
            self._llm_kwargs = dict(kwargs)
            logger.info(
                "Initializing vLLM (attempt %d/%d) with gpu_memory_utilization=%.3f",
                len(tried_utils),
                len(util_candidates),
                util,
            )
            try:
                self._llm = LLM(**kwargs)  # type: ignore[call-arg]
                break
            except TypeError as exc:
                if "enable_lora" in kwargs:
                    kwargs.pop("enable_lora", None)
                    self._llm_kwargs = dict(kwargs)
                    try:
                        self._llm = LLM(**kwargs)  # type: ignore[call-arg]
                        break
                    except Exception as inner_exc:
                        last_exc = inner_exc
                        continue
                last_exc = exc
            except ValueError as exc:
                last_exc = exc
                if "gpu memory utilization" not in str(exc).lower():
                    break
                logger.warning(
                    "vLLM initialization failed with gpu_memory_utilization=%.3f: %s",
                    util,
                    exc,
                )
                continue
            except RuntimeError as exc:
                last_exc = exc
                if "gpu" not in str(exc).lower():
                    break
                logger.warning(
                    "vLLM initialization encountered runtime error with gpu_memory_utilization=%.3f: %s",
                    util,
                    exc,
                )
                continue
        else:
            if last_exc is not None:
                raise last_exc
            raise ExecutionError("Failed to initialize vLLM; GPU memory constraints unresolved.")

        self._model_name = ident

    # --------------------------------------------------------------------- #
    # Data preparation
    # --------------------------------------------------------------------- #
    def prepare_data(self, spec: Dict[str, Any]) -> None:
        entry = self._collect_prompts_for_spec(spec, task_id="__standalone__")
        self._batched_prompts = list(entry["prompts"])
        self._prompt_owners = [entry["task_id"]] * len(entry["prompts"])
        self._base_inference = entry["inference"]

    def _collect_prompts_for_spec(self, spec: Dict[str, Any], *, task_id: str) -> Dict[str, Any]:
        data = spec.get("data") or {}
        if not data:
            raise ExecutionError("spec.data is required.")

        dtype = data.get("type")
        append_system_prompt = True
        prompts: List[str] = []
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
                raise ExecutionError(
                    f"Column '{column}' not found in dataset. Available: {dataset.column_names}"
                )
            prompts = [str(x) for x in dataset[column]]
        elif dtype == "list":
            items = data.get("items", [])
            if not isinstance(items, list) or any(not isinstance(x, str) for x in items):
                raise ExecutionError("spec.data.items must be a list of strings for type == 'list'.")
            prompts = [str(x) for x in items]
        elif dtype == "graph_template":
            prompts = build_prompts_from_graph_template(data, spec)
            template_cfg = data.get("template") or {}
            append_system_prompt = bool(template_cfg.get("append_system_prompt", False))
        else:
            raise ExecutionError(f"Unsupported spec.data.type: {dtype!r}")

        inference_cfg = copy.deepcopy(spec.get("inference", {}) or {})
        system_prompt = inference_cfg.get("system_prompt") or inference_cfg.get("systemPrompt")
        if system_prompt and append_system_prompt:
            prompts = [f"{system_prompt}\n{p}" for p in prompts]

        return {
            "task_id": task_id,
            "prompts": prompts,
            "inference": inference_cfg,
            "data": copy.deepcopy(data),
        }

    def _normalize_inference_for_sampling(self, inference: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in inference.items():
            if key in {"system_prompt", "systemPrompt"}:
                continue
            normalized[key] = copy.deepcopy(value)
        return normalized

    def _build_sampling_params(self, inference: Dict[str, Any]) -> SamplingParams:
        cfg = copy.deepcopy(inference)
        cfg.pop("system_prompt", None)
        cfg.pop("systemPrompt", None)
        return SamplingParams(  # type: ignore[call-arg]
            temperature=float(cfg.get("temperature", 0.7)),
            top_p=float(cfg.get("top_p", 0.95)),
            top_k=int(cfg.get("top_k", -1)),
            max_tokens=int(cfg.get("max_tokens", 512)),
            presence_penalty=float(cfg.get("presence_penalty", 0.0)),
            frequency_penalty=float(cfg.get("frequency_penalty", 0.0)),
            stop=cfg.get("stop"),
        )

    # --------------------------------------------------------------------- #
    # Execution
    # --------------------------------------------------------------------- #
    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:  # type: ignore[override]
        spec = (task or {}).get("spec") or {}
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            raise ExecutionError("task_id is required for inference execution")

        self._ensure_llm(spec)

        merge_children = task.get("merge_children") or []
        entries: List[Dict[str, Any]] = []
        parent_entry = self._collect_prompts_for_spec(spec, task_id=task_id)
        entries.append(parent_entry)

        for child in merge_children:
            if not isinstance(child, dict):
                continue
            child_id = str(child.get("task_id") or "").strip()
            child_spec = child.get("spec") or {}
            if not child_id or not child_spec:
                continue
            try:
                entries.append(self._collect_prompts_for_spec(child_spec, task_id=child_id))
            except ExecutionError as exc:
                raise ExecutionError(f"Failed to prepare merged child task {child_id}: {exc}") from exc

        self._batched_prompts = []
        self._prompt_owners = []
        for entry in entries:
            owner = entry["task_id"]
            for prompt in entry["prompts"]:
                self._batched_prompts.append(prompt)
                self._prompt_owners.append(owner)

        if not self._batched_prompts:
            raise ExecutionError("No prompts prepared. Check spec.data configuration.")

        self._base_inference = copy.deepcopy(parent_entry["inference"])
        base_sampling_cfg = self._normalize_inference_for_sampling(self._base_inference)
        for entry in entries[1:]:
            other_cfg = self._normalize_inference_for_sampling(entry["inference"])
            if other_cfg != base_sampling_cfg:
                raise ExecutionError("Merged tasks must share inference parameters (excluding system_prompt).")

        sampling = self._build_sampling_params(self._base_inference)

        self._prepare_adapters(spec, out_dir)

        # Build per-request LoRA payload(s)
        lora_payload = self._build_lora_requests(out_dir)
        generate_kwargs: Dict[str, Any] = {}
        if lora_payload is not None:
            generate_kwargs["lora_request"] = lora_payload

        t0 = time.time()
        try:
            outputs = self._llm.generate(
                self._batched_prompts,
                sampling_params=sampling,
                **generate_kwargs,
            )  # type: ignore[attr-defined]
        except TypeError as exc:
            single = self._maybe_single_lora(lora_payload)
            if single is not None:
                outputs = self._llm.generate(
                    self._batched_prompts,
                    sampling_params=sampling,
                    lora_request=single,
                )
            else:
                raise ExecutionError(
                    "Installed vLLM build does not accept per-request LoRA; upgrade vLLM or set adapter.apply='merge'."
                ) from exc
        latency = time.time() - t0

        per_task_items: Dict[str, List[Dict[str, Any]]] = {}
        usage_by_task: Dict[str, Dict[str, int]] = {}
        counts_by_task: Dict[str, int] = {}

        total_prompt_tokens = 0
        total_completion_tokens = 0

        for idx, out in enumerate(outputs):
            owner = self._prompt_owners[idx] if idx < len(self._prompt_owners) else task_id
            owner_items = per_task_items.setdefault(owner, [])
            local_index = len(owner_items)
            out_outputs = getattr(out, "outputs", None)
            if not out_outputs:
                owner_items.append({"index": local_index, "output": "", "finish_reason": None})
                usage_by_task.setdefault(owner, {"prompt_tokens": 0, "completion_tokens": 0})
                counts_by_task[owner] = counts_by_task.get(owner, 0) + 1
                continue

            best = out_outputs[0]
            text = getattr(best, "text", "") or ""
            finish_reason = getattr(best, "finish_reason", None)
            owner_items.append({"index": local_index, "output": text, "finish_reason": finish_reason})

            prompt_token_ids = getattr(out, "prompt_token_ids", None) or []
            best_token_ids = getattr(best, "token_ids", None) or []
            prompt_len = len(prompt_token_ids)
            completion_len = len(best_token_ids)

            total_prompt_tokens += prompt_len
            total_completion_tokens += completion_len

            usage_entry = usage_by_task.setdefault(owner, {"prompt_tokens": 0, "completion_tokens": 0})
            usage_entry["prompt_tokens"] += prompt_len
            usage_entry["completion_tokens"] += completion_len
            counts_by_task[owner] = counts_by_task.get(owner, 0) + 1

        for owner, usage in usage_by_task.items():
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            usage["latency_sec"] = latency
            usage["num_requests"] = counts_by_task.get(owner, 0)

        parent_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "latency_sec": latency,
            "num_requests": len(self._batched_prompts),
        }

        result: Dict[str, Any] = {
            "ok": True,
            "model": self._model_name,
            "items": per_task_items.get(task_id, []),
            "usage": parent_usage,
        }

        child_results: Dict[str, Any] = {}
        for child in merge_children:
            if not isinstance(child, dict):
                continue
            child_id = str(child.get("task_id") or "").strip()
            if not child_id:
                continue
            child_payload = {
                "items": per_task_items.get(child_id, []),
            }
            usage = usage_by_task.get(child_id)
            if usage:
                child_payload["usage"] = usage
            child_results[child_id] = child_payload

        if child_results:
            result["children"] = child_results

        return result

    def _maybe_apply_dataset_shard(self, dataset, spec: Dict[str, Any]):
        shard_cfg = spec.get("shard") if isinstance(spec, dict) else None
        if not shard_cfg:
            return dataset
        try:
            total = int(shard_cfg.get("total", 1))
            index = int(shard_cfg.get("index", 0))
        except Exception as exc:
            raise ExecutionError(f"Invalid shard metadata: {exc}") from exc
        if total <= 1:
            return dataset
        contiguous = bool(shard_cfg.get("contiguous", True))
        try:
            return dataset.shard(num_shards=total, index=index, contiguous=contiguous)
        except Exception as exc:  # pragma: no cover - defensive
            raise ExecutionError(f"Failed to shard dataset ({index}/{total}): {exc}")

    def cleanup_after_run(self) -> None:
        if self._release_llm_after_run:
            self._shutdown_llm()
        self._llm_kwargs = {}
        self._batched_prompts = []
        self._prompt_owners = []
        self._adapter_specs = []
        self._runtime_specs = []
        self._merged_adapters = []
        self._base_inference = {}
        if self._offline_model_dir and self._offline_model_dir.exists():
            shutil.rmtree(self._offline_model_dir, ignore_errors=True)
        self._offline_model_dir = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                try:
                    for idx in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(idx)
                except Exception:
                    pass
        except Exception:
            pass
        gc.collect()

    def _shutdown_llm(self) -> None:
        llm = getattr(self, "_llm", None)
        if llm is None:
            return
        try:
            shutdown = getattr(llm, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
                close_fn = getattr(llm, "close", None)
                if callable(close_fn):
                    close_fn()
        except Exception:
            logger.debug("Failed to shutdown vLLM instance cleanly", exc_info=True)
        finally:
            self._llm = None

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
            if path.exists():
                return self._ensure_or_extract(path, out_dir, name, archive_format)

        task_hint = adapter.get("taskId") or adapter.get("task_id")
        if task_hint:
            task_str = str(task_hint).strip()
            if task_str:
                base_dir = Path(os.getenv("RESULTS_DIR", "./results_workers")).expanduser().resolve()
                candidates = [
                    base_dir / task_str / "final_lora",
                    base_dir / task_str / "final_model",
                    base_dir / task_str,
                ]
                for candidate in candidates:
                    if candidate.exists():
                        return self._ensure_or_extract(candidate, out_dir, name, archive_format)
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
        if fmt == "tar":
            import tarfile
            with tarfile.open(p, "r:*") as tf:
                tf.extractall(target_root)
        else:
            raise ExecutionError(f"Unsupported LoRA archive format for {p}; expected .tar(.gz) or a directory")
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
import tempfile
