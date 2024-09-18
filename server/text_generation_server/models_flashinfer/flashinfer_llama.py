import torch
import torch.distributed

from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.models.llama import LlamaTokenizer
from typing import Optional, List

from text_generation_server.models_flashinfer.flashinfer_causal_lm import FlashinferLM
from text_generation_server.models_flashinfer.custom_modeling.flashinfer_llama_modeling import (
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
        weights: Optional[Weights] = None,
    ):
        dtype = dtype or torch.float16
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
        else:
            raise NotImplementedError("Flashinfer Llama is only available on Cuda")

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
        
        if not hasattr(config, "rms_norm_eps"):
            config.rms_norm_eps = 1e-05
        
        if not hasattr(config, "intermediate_size"):
            config.intermediate_size = 11008
        
        if not hasattr(config, "hidden_act"):
            config.hidden_act = "silu"
            
        if not hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = 32
            
        if not hasattr(config, "hidden_size"):
            config.hidden_size = 4096
        
        if not hasattr(config, "num_attention_heads"):
            config.num_attention_heads = 32
            
        if not hasattr(config, "num_key_value_heads"):
            config.num_key_value_heads = config.num_attention_heads

        if not hasattr(config, "rope_theta"):
            config.rope_theta = 1.0e4

        torch.distributed.barrier(group=self.process_group)
        if not weights:
            filenames = weight_files(model_id, revision=revision, extension=".safetensors")
            weights = Weights(filenames, device, dtype, process_group=self.process_group)
            if config.quantize in ["gptq", "awq"]:
                weights._set_gptq_params(model_id, revision)

        prefix = "language_model" if 'llava' in model_id else ""
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
