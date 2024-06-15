import torch
import torch.distributed
from typing import Optional, List
from text_generation_server.models.flashinfer_causal_lm import FlashinferLM
from text_generation_server.models.custom_modeling.flashinfer_mistral_modeling import (
    MistralConfig,
    FlashMistralForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

from transformers import AutoTokenizer


class FlashinferMistral(FlashinferLM):
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
        else:
            raise NotImplementedError("Flashinfer Mistral is only available on GPU")

        mistralConfig = MistralConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        mistralConfig.quantize = quantize
        mistralConfig.speculator = speculator

        torch.distributed.barrier(group=process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=process_group)
        if quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id, revision)

        model = FlashMistralForCausalLM(None, mistralConfig, weights)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        super(FlashinferMistral, self).__init__(
            model=model,
            tokenizer=tokenizer,
            config=mistralConfig,
            dtype=dtype,
            device=device,
            lora_ids=lora_ids,
        )
