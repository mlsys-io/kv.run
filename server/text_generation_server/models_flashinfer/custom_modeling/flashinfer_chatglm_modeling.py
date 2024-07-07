# Adapted from HuggingFace Transformers Library
# https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/models/llama/modeling_llama.py

import math

import torch
import flashinfer
from torch import nn
import torch.distributed
import torch.nn.functional as F

from text_generation_server.layers.layernorm import FastRMSNorm
from text_generation_server.utils.lora_utils import BatchedModelLoraWeight
from text_generation_server.utils.cache_manager_flashinfer import KvCachePool, KvCacheBatchPosition
from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear
)

from typing import Optional, List, Tuple
from transformers.tokenization_utils import AddedToken
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.activations import ACT2FN

from text_generation_server.layers.flashinfer_attention import (
    FlashinferAttentionWrapper,
    AttentionRotaryParams,
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


logger = logging.get_logger(__name__)


class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"
    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        classifier_dropout=None,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        super().__init__(**kwargs)


class ChatGLMRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states
        return rms_norm(hidden_states, self.weight, self.variance_epsilon), residual


class ChatGLMMLP(nn.Module):
    def __init__(self, prefix, config, weights, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        self.up_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense_h_to_4h",
            weights=weights,
            bias=False,
        )

        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense_4h_to_h",
            weights=weights,
            bias=False,
        )


    def forward(self, hidden_states, loraWeight: BatchedModelLoraWeight):
        # [s, b, 3hp]
        up = self.up_proj(hidden_states)
        loraWeight.apply_lora_weight_gate(up, hidden_states, self.layer_idx)
        up = self.activation_func(up)
        # [s, b, h]
        down = self.down_proj(up)
        loraWeight.apply_lora_weight_down(down, up, self.layer_idx)
        return down


def load_attention(config, prefix, weights, num_key_value_heads):
    if config.num_attention_heads != num_key_value_heads:
        return _load_gqa(config, prefix, weights)
    else:
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=True,
        )


def _load_gqa(config, prefix: str, weights):
    assert config.num_attention_heads % weights.process_group.size() == 0

    weight = weights.get_weights_col(
        prefix=prefix,
        quantize=config.quantize,
    )

    # manually concatenate qkv project bias
    bias = weights.get_tensor(f"{prefix}.bias")

    if config.quantize not in ["gptq", "awq"]:
        weight = weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.hidden_size // config.num_attention_heads
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.multi_query_group_num // weights.process_group.size()
        
        assert list(weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.shape)} != {[(num_heads + 2 * num_key_value_heads) * head_size, config.hidden_size]}"

    return TensorParallelColumnLinear(
        get_linear(weight, bias=bias, quantize=config.quantize)
    )


class FlashChatGLMAttention(nn.Module):
    def __init__(
        self,
        prefix: str,
        flashinferWrapper: FlashinferAttentionWrapper,
        config: ChatGLMConfig,
        weights,
        layer_idx: int
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_qo_heads = self.num_heads // weights.process_group.size()

        self.num_key_value_heads = config.multi_query_group_num
        self.num_key_value_groups = self.num_qo_heads // self.num_key_value_heads
        
        # self.num_kv_heads = (
        #     config.num_key_value_heads // weights.process_group.size()
        # )
        self.num_kv_heads = self.num_key_value_heads // weights.process_group.size()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.flashinferWrapper = flashinferWrapper
        self.rotaryParams = AttentionRotaryParams(
            rope_scale=None, rope_theta=10000 * config.rope_ratio
        )

        self.layer_idx = layer_idx
        self.qkv_proj = load_attention(config, f'{prefix}.query_key_value', weights, self.num_key_value_heads)

        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense",
            weights=weights,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight | None
    ):
        q_dim = (
            self.flashinferWrapper.num_attention_heads * self.flashinferWrapper.head_dim
        )
        kv_dim = (
            self.flashinferWrapper.num_key_value_heads * self.flashinferWrapper.head_dim
        )
        qkv = self.qkv_proj(hidden_states)
        q_proj, k_proj, v_proj = qkv.split(
            [q_dim, kv_dim, kv_dim],
            dim=1,
        )
        q = q_proj.contiguous()
        k = k_proj.contiguous()
        v = v_proj.contiguous()

        loraWeight.apply_lora_weight_kvq(q, k, v, hidden_states, self.layer_idx)

        attn_outputs_raw = self.flashinferWrapper.computeAttention(
            q,
            k,
            v,
            kvCachePool.cache_data[self.layer_idx],
            kvCachePool.page_len,
            prefillBatchPosition,
            decodeBatchPosition,
            self.rotaryParams,
        )
        attn_outputs = self.o_proj(attn_outputs_raw)
        loraWeight.apply_lora_weight_attn(
            attn_outputs, attn_outputs_raw, self.layer_idx
        )
        return attn_outputs


class FlashChatGLM3Layer(nn.Module):
    def __init__(self,
        flashinferWrapper: FlashinferAttentionWrapper,
        layer_id, config, weights
    ):
        super().__init__()
        self.layer_id = layer_id
        prefix = f"transformer.encoder.layers.{layer_id}"
        self.self_attn = FlashChatGLMAttention(
            prefix=f"{prefix}.self_attention",
            flashinferWrapper=flashinferWrapper,
            config=config,
            weights=weights,
            layer_idx=layer_id
        )
        self.mlp = ChatGLMMLP(prefix=f"{prefix}.mlp", config=config, weights=weights, layer_idx=layer_id)

        self.input_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.layernorm_epsilon
        )
        self.post_attention_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.layernorm_epsilon,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight | None
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


class FlashChatGLM3Model(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix="transformer.embedding.word_embeddings", weights=weights
        )

        assert config.num_attention_heads % weights.process_group.size() == 0
        assert config.multi_query_group_num % weights.process_group.size() == 0
        num_attention_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.multi_query_group_num // weights.process_group.size()

        flashinferWrapper = FlashinferAttentionWrapper(
            num_attention_heads, num_key_value_heads, config.hidden_size
        )

        self.layers = nn.ModuleList(
            [
                FlashChatGLM3Layer(
                    flashinferWrapper,
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix="transformer.encoder.final_layernorm", weights=weights, eps=config.layernorm_epsilon
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_dim
        self.num_heads = self.layers[0].self_attn.num_qo_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_kv_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight | None
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


class FlashChatGLMForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.model = FlashChatGLM3Model(config, weights)
        self.lm_head = SpeculativeHead.load(
            config,
            prefix="transformer.embedding.word_embeddings" if config.tie_word_embeddings else "transformer.output_layer",
            weights=weights,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool,
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight | None
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