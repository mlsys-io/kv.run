import torch
import enum

from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from typing import Optional

from text_generation_server.models.model import Model
from text_generation_server.models_flashinfer.flashinfer_llama import FlashinferLlama
from text_generation_server.models_flashinfer.flashinfer_gemma import FlashinferGemma
from text_generation_server.models_flashinfer.flashinfer_mistral import (
    FlashinferMistral,
)
from text_generation_server.models_flashinfer.flashinfer_phi import FlashinferPhi
from text_generation_server.models_flashinfer.flashinfer_qwen2 import FlashinferQwen2

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Disable gradients
torch.set_grad_enabled(False)

__all__ = [
    "Model",
    "FlashinferLlama",
    "FlashinferGemma",
    "FlashinferMistral",
    "FlashinferPhi",
    "FlashinferQwen2",
    "get_model",
]


class ModelType(enum.Enum):
    LLAVA_NEXT = {
        "type": "llava_next",
        "name": "Llava Next (1.6)",
        "url": "https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf",
        "multimodal": True,
    }
    LLAMA = {
        "type": "llama",
        "name": "Llama",
        "url": "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
    }
    PHI3 = {
        "type": "phi3",
        "name": "Phi 3",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
    }
    GEMMA = {
        "type": "gemma",
        "name": "Gemma",
        "url": "https://huggingface.co/google/gemma-7b",
    }
    MISTRAL = {
        "type": "mistral",
        "name": "Mistral",
        "url": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
    }
    PHI = {
        "type": "phi",
        "name": "Phi",
        "url": "https://huggingface.co/microsoft/phi-1_5",
    }
    BAICHUAN = {
        "type": "baichuan",
        "name": "Baichuan",
        "url": "https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat",
    }
    QWEN2 = {
        "type": "qwen2",
        "name": "Qwen 2",
        "url": "https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1",
    }


def get_model(
    model_id: str,
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    dtype: Optional[str],
    trust_remote_code: bool,
    lora_ids: Optional[str],
) -> Model:
    if dtype is None:
        if quantize in ["awq", "exl2", "gptq"]:
            # These quantizers only work with float16 params.
            dtype = torch.float16
        else:
            # Keep it as default for now and let
            # every model resolve their own default dtype.
            dtype = None
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype {dtype}")

    config_dict, _ = PretrainedConfig.get_config_dict(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    model_type = config_dict.get("model_type", None)
    if model_type is None:
        raise RuntimeError(
            f"Could not determine model type for {model_id} revision {revision}"
        )
    quantization_config = config_dict.get("quantization_config", None)
    if quantization_config is not None and quantize is None:
        method = quantization_config.get("quant_method", None)
        if method in {"gptq", "awq", "exl2"}:
            logger.info(f"Auto selecting quantization method {method}")
            quantize = method
        else:
            logger.info(f"Unknown quantization method {method}")

    if quantize == "exl2" and sharded:
        raise RuntimeError(
            "Sharding is currently not supported with `exl2` quantization"
        )

    if model_type == PHI:
        return FlashinferPhi(
            model_id,
            lora_ids.split(";") if lora_ids else None,
            quantize=quantize,
            dtype=dtype,
        )
    elif model_type == LLAMA or model_type == BAICHUAN or model_type == PHI3:
        return FlashinferLlama(
            model_id,
            lora_ids.split(";") if lora_ids else None,
            quantize=quantize,
            dtype=dtype,
        )
    elif model_type == GEMMA:
        return FlashinferGemma(
            model_id,
            lora_ids.split(";") if lora_ids else None,
            quantize=quantize,
            dtype=dtype,
        )
    elif model_type == MISTRAL:
        return FlashinferMistral(
            model_id,
            lora_ids.split(";") if lora_ids else None,
            quantize=quantize,
            dtype=dtype,
        )
    elif model_type == QWEN2:
        return FlashinferQwen2(
            model_id,
            lora_ids.split(";") if lora_ids else None,
            quantize=quantize,
            dtype=dtype,
        )

    raise ValueError(f"Unsupported model type {model_type}")
