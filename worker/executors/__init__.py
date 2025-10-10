import importlib
from typing import Dict, Optional

_IMPORT_ERRORS: Dict[str, str] = {}

def _safe_import(name: str, module: str) -> Optional[type]:
    try:
        pkg = importlib.import_module(module)
        return getattr(pkg, name)
    except Exception as exc:  # noqa: broad-except
        _IMPORT_ERRORS[name] = str(exc)
        return None


VLLMExecutor = _safe_import("VLLMExecutor", "worker.executors.vllm_executor")
PPOExecutor = _safe_import("PPOExecutor", "worker.executors.ppo_executor")
DPOExecutor = _safe_import("DPOExecutor", "worker.executors.dpo_executor")
SFTExecutor = _safe_import("SFTExecutor", "worker.executors.sft_executor")
LoRASFTExecutor = _safe_import("LoRASFTExecutor", "worker.executors.lora_sft_executor")
HFTransformersExecutor = _safe_import("HFTransformersExecutor", "worker.executors.transformers_executor")
RAGExecutor = _safe_import("RAGExecutor", "worker.executors.rag_executor")
AgentExecutor = _safe_import("AgentExecutor", "worker.executors.agent_executor")
EchoExecutor = _safe_import("EchoExecutor", "worker.executors.echo_executor")

EXECUTOR_REGISTRY: Dict[str, Optional[type]] = {
    "vllm": VLLMExecutor,
    "ppo": PPOExecutor,
    "dpo": DPOExecutor,
    "sft": SFTExecutor,
    "lora_sft": LoRASFTExecutor,
    "default": HFTransformersExecutor,
    "rag": RAGExecutor,
    "agent": AgentExecutor,
    "echo": EchoExecutor,
}

EXECUTOR_CLASS_NAMES: Dict[str, str] = {
    "vllm": "VLLMExecutor",
    "ppo": "PPOExecutor",
    "dpo": "DPOExecutor",
    "sft": "SFTExecutor",
    "lora_sft": "LoRASFTExecutor",
    "default": "HFTransformersExecutor",
    "rag": "RAGExecutor",
    "agent": "AgentExecutor",
    "echo": "EchoExecutor",
}

IMPORT_ERRORS: Dict[str, str] = dict(_IMPORT_ERRORS)

__all__ = [
    name for name, cls in {
        "VLLMExecutor": VLLMExecutor,
        "PPOExecutor": PPOExecutor,
        "DPOExecutor": DPOExecutor,
        "SFTExecutor": SFTExecutor,
        "LoRASFTExecutor": LoRASFTExecutor,
        "HFTransformersExecutor": HFTransformersExecutor,
        "RAGExecutor": RAGExecutor,
        "AgentExecutor": AgentExecutor,
        "EchoExecutor": EchoExecutor,
    }.items() if cls is not None
] + ["EXECUTOR_REGISTRY", "IMPORT_ERRORS", "EXECUTOR_CLASS_NAMES"]
