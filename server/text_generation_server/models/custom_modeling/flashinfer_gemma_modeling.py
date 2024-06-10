# Adapted from TGI's flash_gemma_modeling.py
import torch
import torch.distributed
import os
from shutil import copyfile

from text_generation_server.utils.lora_utils import BatchedModelLoraWeight
from text_generation_server.utils.cache_manager_flashinfer import KvCachePool, KvCacheBatchPosition
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List, Tuple
from tokenizers import processors
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging
from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear
)

from text_generation_server.layers.flashinfer_attention import (
    FlashinferAttentionWrapper, AttentionRotaryParams
)
from punica_kernels import (
    rms_norm,
)

class FlashinferBatch:
    def __init__(self, seq_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len):
        self.seq_indptr = seq_indptr
        self.kv_page_indptr = kv_page_indptr
        self.kv_page_indices = kv_page_indices
        self.kv_last_page_len = kv_last_page_len

GemmaTokenizer = None

logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {
    "vocab_file": "tokenizer.model",
    "tokenizer_file": "tokenizer.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on


class GemmaTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    slow_tokenizer_class = GemmaTokenizer
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        add_bos_token=True,
        add_eos_token=False,
        use_default_system_prompt=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            use_default_system_prompt=use_default_system_prompt,
            **kwargs,
        )
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()
        self.use_default_system_prompt = use_default_system_prompt
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def update_post_processor(self):
        """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        eos = self.eos_token
        eos_token_id = self.eos_token_id
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        single = f"{(bos+':0 ') if self.add_bos_token else ''}$A:0{(' '+eos+':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' '+bos+':1') if self.add_bos_token else ''} $B:1{(' '+eos+':1') if self.add_eos_token else ''}"

        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def add_eos_token(self):
        return self._add_eos_token

    @property
    def add_bos_token(self):
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value
        self.update_post_processor()

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    @property
    def default_chat_template(self):
        raise NotImplementedError

    # TODO ArthurZ let's rely on the template processor instead, refactor all fast tokenizers
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output


class GemmaConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=256128,
        hidden_size=3072,
        intermediate_size=24576,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        hidden_act="gelu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        speculator=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.speculator = speculator

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class GemmaRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight") + 1
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps
        
    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states
        return rms_norm(hidden_states, self.weight, self.variance_epsilon), residual

def load_attention(config, prefix, weights):
    if config.num_attention_heads != config.num_key_value_heads:
        return _load_gqa(config, prefix, weights)
    else:
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=False,
        )

def _load_gqa(config, prefix: str, weights):
    assert config.num_attention_heads % weights.process_group.size() == 0

    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        quantize=config.quantize,
        dim=0,
    )

    if config.quantize not in ["gptq", "awq"]:
        weight = weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.head_dim
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()
        assert list(weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.shape)} != {[(num_heads + 2 * config.num_key_value_heads) * head_size, config.hidden_size]}"

    return TensorParallelColumnLinear(
        get_linear(weight, bias=None, quantize=config.quantize)
    )

class FlashGemmaAttention(nn.Module):
    def __init__(self, prefix: str, flashinferWrapper: FlashinferAttentionWrapper, config: GemmaConfig, weights, layer_idx: int):
        super().__init__()
        
        self.flashinferWrapper = flashinferWrapper 
        self.rotaryParams = AttentionRotaryParams(rope_scale=config.rope_scaling, rope_theta=config.rope_theta)
       
        self.layer_idx = layer_idx
        self.qkv_proj = load_attention(config, prefix, weights)
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight,
    ) -> torch.Tensor:
        q_dim = self.flashinferWrapper.num_attention_heads * self.flashinferWrapper.head_dim
        kv_dim = self.flashinferWrapper.num_key_value_heads * self.flashinferWrapper.head_dim
        qkv = self.qkv_proj(hidden_states)
        q_proj, k_proj, v_proj = qkv.split(
            [
                q_dim,
                kv_dim,
                kv_dim
            ],
            dim=1,
        )
        q = q_proj.contiguous()
        k = k_proj.contiguous()
        v = v_proj.contiguous()
        loraWeight.apply_lora_weight_kvq(q, k, v, hidden_states, self.layer_idx)
        attn_outputs_raw = self.flashinferWrapper.computeAttention(q, k, v, kvCachePool.cache_data[self.layer_idx], 
                                kvCachePool.page_len, prefillBatchPosition, decodeBatchPosition, self.rotaryParams)
        attn_outputs = self.o_proj(attn_outputs_raw)
        loraWeight.apply_lora_weight_attn(attn_outputs, attn_outputs_raw, self.layer_idx)
        return attn_outputs


class GemmaMLP(nn.Module):
    def __init__(self, prefix, config, weights, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
                ),
            )
        )
        # Fuse gate and up proj
        self.gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=False,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

    def forward(self, hidden_states: torch.Tensor, loraWeight: BatchedModelLoraWeight):
        gate_up_states = self.gate_up_proj(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        gate = gate_up_states[:, 0].contiguous()
        loraWeight.apply_lora_weight_gate(gate, hidden_states, self.layer_idx)
        gate = self.act(gate)
        up = gate_up_states[:, 1].contiguous()
        loraWeight.apply_lora_weight_up(up, hidden_states, self.layer_idx)     
        t = gate * up
        down = self.down_proj(t)
        loraWeight.apply_lora_weight_down(down, t, self.layer_idx)      
        return down

class FlashGemmaLayer(nn.Module):
    def __init__(self, flashinferWrapper: FlashinferAttentionWrapper, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashGemmaAttention(
            prefix=f"{prefix}.self_attn", flashinferWrapper=flashinferWrapper, config=config, weights=weights, layer_idx=layer_id
        )
        self.mlp = GemmaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights, layer_idx=layer_id)

        self.input_layernorm = GemmaRMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)
        attn_output = self.self_attn(
            normed_hidden_states,
            kvCachePool,
            prefillBatchPosition,
            decodeBatchPosition,
            loraWeight
        )

        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output, loraWeight)

        return mlp_output, attn_res


class FlashGemmaModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        embed_norm = config.hidden_size**0.5
        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.embed_tokens.weight *= embed_norm
        
        assert config.num_attention_heads % weights.process_group.size() == 0
        assert config.num_key_value_heads % weights.process_group.size() == 0
        num_attention_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()
        
        flashinferWrapper = FlashinferAttentionWrapper(
            num_attention_heads, num_key_value_heads, config.hidden_size
        )

        self.layers = nn.ModuleList(
            [
                FlashGemmaLayer(
                    flashinferWrapper,
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                kvCachePool,
                prefillBatchPosition,
                decodeBatchPosition,
                loraWeight
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashGemmaForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.model = FlashGemmaModel(config, weights)
        self.lm_head = SpeculativeHead.load(
            config,
            prefix="model.embed_tokens" if config.tie_word_embeddings else "lm_head",
            weights=weights,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool,
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.model(
            input_ids,
            kvCachePool,
            prefillBatchPosition,
            decodeBatchPosition,
            loraWeight
        )
        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits
