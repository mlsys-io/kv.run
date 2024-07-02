import torch
import torch.distributed
from typing import Optional, List
from transformers import AutoTokenizer, AutoConfig
from text_generation_server.models_flashinfer.flashinfer_causal_lm import FlashinferLM
from text_generation_server.models_flashinfer.custom_modeling.flashinfer_chatglm_modeling import (
    ChatGLMConfig, FlashChatGLMForCausalLM
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)


class FlashinferChatGLM(FlashinferLM):
    def __init__(
        self,
        model_id: str,
        lora_ids: List[str] = None,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.bfloat16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashChatGLM3 is only available on GPU")

        chatglmConfig = ChatGLMConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        
        chatglmConfig.quantize = quantize
        chatglmConfig.speculator = speculator

        # deal with chatglm special config names
        chatglmConfig.num_hidden_layers = chatglmConfig.num_layers
        chatglmConfig.intermediate_size = chatglmConfig.ffn_hidden_size
        chatglmConfig.num_key_value_heads = chatglmConfig.multi_query_group_num

        torch.distributed.barrier(group=process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=process_group)
        if chatglmConfig.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id, revision)

        model = FlashChatGLMForCausalLM(chatglmConfig, weights)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        super(FlashinferChatGLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            config = chatglmConfig,
            dtype=dtype,
            device=device,
            lora_ids = lora_ids,
        )
