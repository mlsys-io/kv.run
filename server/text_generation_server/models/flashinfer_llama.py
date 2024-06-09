import torch
from typing import Optional, List

from text_generation_server.models.flashinfer_causal_lm import FlashinferLM
from text_generation_server.models.custom_modeling.flashinfer_llama_modeling import (
    LlamaForCausalLM,
)

from transformers import AutoConfig, AutoTokenizer, GenerationConfig

class FlashinferLlama(FlashinferLM):
    def __init__(
        self,
        model_id: str,
        lora_ids: List[str] = None,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        trust_remote_code: bool = False,
    ):
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise NotImplementedError("Flashinfer Llama is only available on GPU")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        
        llamaConfig = model.config
        llamaConfig.quantize = quantize
        
        super(FlashinferLlama, self).__init__(
            model=model,
            tokenizer=tokenizer,
            config = llamaConfig,
            dtype=dtype,
            device=device,
            lora_ids = lora_ids,
        )