#!/usr/bin/env python3
"""
VLLMExecutor (YAML schema specific)

- 从 spec.model.vllm 读取模型配置
- 从 spec.inference 读取采样参数与输入 (prompts/messages)
- 生成结果写到 out_dir/responses.json
"""

import os
import time
import json
from pathlib import Path
from typing import Any, Dict, List

from .base_executor import Executor, ExecutionError

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except Exception:
    LLM = object  # type: ignore
    SamplingParams = object  # type: ignore
    _HAS_VLLM = False


def _messages_to_prompt(conversation: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in conversation)


class VLLMExecutor(Executor):
    name = "vllm"

    def __init__(self) -> None:
        super().__init__()
        self._llm: LLM | None = None
        self._model_name: str | None = None

    def prepare(self) -> None:  # type: ignore[override]
        if not _HAS_VLLM:
            raise ExecutionError("vLLM is not installed (`pip install vllm`).")

    def _ensure_llm(self, spec: Dict[str, Any]) -> None:
        if self._llm is not None:
            return
        model_src = (spec.get("model") or {}).get("source", {})
        ident = model_src.get("identifier") or os.getenv("VLLM_MODEL")
        if not ident:
            raise ExecutionError("spec.model.source.identifier (or VLLM_MODEL) required")

        vllm_cfg = (spec.get("model") or {}).get("vllm", {})
        kwargs = dict(
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

        self._llm = LLM(**kwargs)
        self._model_name = ident

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:  # type: ignore[override]
        spec = (task or {}).get("spec") or {}
        self._ensure_llm(spec)

        inf = spec.get("inference") or {}
        if "prompts" in inf:
            prompts = inf["prompts"]
        elif "messages" in inf:
            prompts = [_messages_to_prompt(c) for c in inf["messages"]]
        else:
            raise ExecutionError("spec.inference must contain prompts or messages")

        sampling = SamplingParams(
            temperature=float(inf.get("temperature", 0.7)),
            top_p=float(inf.get("top_p", 0.95)),
            top_k=int(inf.get("top_k", -1)),
            max_tokens=int(inf.get("max_tokens", 512)),
            presence_penalty=float(inf.get("presence_penalty", 0.0)),
            frequency_penalty=float(inf.get("frequency_penalty", 0.0)),
            stop=inf.get("stop"),
        )

        t0 = time.time()
        outputs = self._llm.generate(prompts, sampling_params=sampling)  # type: ignore
        dt = time.time() - t0

        items: List[Dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0
        for i, out in enumerate(outputs):
            if not getattr(out, "outputs", None):
                items.append({"index": i, "output": "", "finish_reason": None})
                continue
            best = out.outputs[0]
            items.append({
                "index": i,
                "output": getattr(best, "text", ""),
                "finish_reason": getattr(best, "finish_reason", None),
            })
            prompt_tokens += getattr(out, "prompt_token_ids", []) and len(out.prompt_token_ids) or 0
            completion_tokens += getattr(best, "token_ids", []) and len(best.token_ids) or 0

        result = {
            "ok": True,
            "model": self._model_name,
            "items": items,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency_sec": dt,
                "num_requests": len(prompts),
            },
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "responses.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
