import torch
import torch.distributed

from typing import Optional, List
from text_generation_server.models_flashinfer.flashinfer_causal_lm import FlashinferLM
from text_generation_server.models_flashinfer.custom_modeling.flashinfer_qwen2_modeling import (
    Qwen2Config,
    FlashQwen2ForCausalLM,
)

from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

from transformers import AutoTokenizer, AutoConfig


class FlashinferQwen2(FlashinferLM):
    def __init__(
        self,
        model_id: str,
        lora_ids: List[str] = None,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        trust_remote_code: bool = False,
    ):
        dtype = dtype or torch.float16
        process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.bfloat16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashQwen2 is only available on GPU")

        qwenConfig = Qwen2Config.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        qwenConfig.quantize = quantize
        qwenConfig.speculator = speculator

        torch.distributed.barrier(group=process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=process_group)
        if qwenConfig.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id, revision)

        model = FlashQwen2ForCausalLM(qwenConfig, weights)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        super(FlashinferQwen2, self).__init__(
            model=model,
            tokenizer=tokenizer,
            config=qwenConfig,
            dtype=dtype,
            device=device,
            lora_ids=lora_ids,
        )
