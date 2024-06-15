import torch
import torch.distributed

from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.models.llama import LlamaTokenizer
from typing import Optional, List

from text_generation_server.models.flashinfer_causal_lm import FlashinferLM
from text_generation_server.models.custom_modeling.flashinfer_llama_modeling import (
    FlashLlamaForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)


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
        dtype = dtype or torch.float16
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
        else:
            raise NotImplementedError("Flashinfer Llama is only available on Cuda")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                model_id,
                revision=revision,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=trust_remote_code,
            )
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id, revision=revision, trust_remote_code=trust_remote_code
            )
            if isinstance(generation_config.eos_token_id, (list, set)):
                # TODO Huge hack
                tokenizer._eos_token_ids = set(generation_config.eos_token_id)
        except Exception:
            pass

        config = AutoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize
        config.speculator = None
        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        if config.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id, revision)

        prefix = ""
        model = FlashLlamaForCausalLM(prefix, config, weights)
        torch.distributed.barrier(group=self.process_group)
        super(FlashinferLlama, self).__init__(
            model=model,
            tokenizer=tokenizer,
            config=config,
            dtype=dtype,
            device=device,
            lora_ids=lora_ids,
        )
