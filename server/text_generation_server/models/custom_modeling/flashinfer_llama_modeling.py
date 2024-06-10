import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    ACT2FN,
    LlamaConfig,
    PreTrainedModel,
)

from punica_kernels import (
    rms_norm,
)
from text_generation_server.utils.lora_utils import BatchedModelLoraWeight
from text_generation_server.utils.cache_manager_flashinfer import KvCachePool, KvCacheBatchPosition
from text_generation_server.layers.flashinfer_attention import FlashinferAttentionWrapper, AttentionRotaryParams

class FlashinferBatch:
    def __init__(self, seq_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len):
        self.seq_indptr = seq_indptr
        self.kv_page_indptr = kv_page_indptr
        self.kv_page_indices = kv_page_indices
        self.kv_last_page_len = kv_last_page_len

class LlamaAttention(nn.Module):
    def __init__(self, flashinferWrapper: FlashinferAttentionWrapper, config: LlamaConfig, layer_idx: int):
        super().__init__()

        self.flashinferWrapper = flashinferWrapper
        self.rotaryParams = AttentionRotaryParams(rope_scale=config.rope_scaling, rope_theta=config.rope_theta)
        self.layer_idx = layer_idx
        
        q_dim = flashinferWrapper.num_attention_heads * flashinferWrapper.head_dim
        kv_dim = flashinferWrapper.num_key_value_heads * flashinferWrapper.head_dim
        self.q_proj = nn.Linear(
            flashinferWrapper.hidden_size, q_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            kv_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            kv_dim, bias=False
        )
        self.o_proj = nn.Linear(
            q_dim, flashinferWrapper.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        loraWeight.apply_lora_weight_kvq(q, k, v, hidden_states, self.layer_idx)
        attn_outputs_raw = self.flashinferWrapper.computeAttention(q, k, v, kvCachePool.cache_data[self.layer_idx], 
                                kvCachePool.page_len, prefillBatchPosition, decodeBatchPosition, self.rotaryParams)
        attn_outputs = self.o_proj(attn_outputs_raw)
        loraWeight.apply_lora_weight_attn(attn_outputs, attn_outputs_raw, self.layer_idx)
        return attn_outputs

class LlamaMlp(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        x: torch.Tensor,
        loraWeight: BatchedModelLoraWeight
    ) -> torch.Tensor:
        gate = self.gate_proj(x)
        loraWeight.apply_lora_weight_gate(gate, x, self.layer_idx)
        gate = self.act_fn(gate)
        up = self.up_proj(x)
        loraWeight.apply_lora_weight_gate(up, x, self.layer_idx)
        t = gate * up
        down = self.down_proj(t)
        loraWeight.apply_lora_weight_gate(down, t, self.layer_idx)
        return down

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, flashinferWrapper: FlashinferAttentionWrapper, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(flashinferWrapper, config, layer_idx)
        self.mlp = LlamaMlp(config, layer_idx)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight,
    ) -> torch.Tensor:
        residual = hidden_states

        torch.cuda.nvtx.range_push("input_norm")
        hidden_states = self.input_layernorm(hidden_states)
        torch.cuda.nvtx.range_pop()

        # Self Attention
        torch.cuda.nvtx.range_push("LlamaAttention")
        hidden_states = self.self_attn(hidden_states, kvCachePool, prefillBatchPosition, decodeBatchPosition, loraWeight)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("r")
        hidden_states = residual + hidden_states
        torch.cuda.nvtx.range_pop()

        # Fully Connected
        residual = hidden_states
        torch.cuda.nvtx.range_push("norm")
        hidden_states = self.post_attention_layernorm(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("mlp")
        hidden_states = self.mlp(hidden_states, loraWeight)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("r")
        hidden_states = residual + hidden_states
        torch.cuda.nvtx.range_pop()

        return hidden_states


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LlamaDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.version",
        r"self_attn\.rotary_emb\.inv_freq",
    ]


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        flashinferWrapper = FlashinferAttentionWrapper(
            config.num_attention_heads, config.num_key_value_heads, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(flashinferWrapper, config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight
    ) -> torch.Tensor:
        torch.cuda.nvtx.range_push("embed")
        hidden_states = self.embed_tokens(input_ids)
        torch.cuda.nvtx.range_pop()

        for layer_idx, decoder_layer in enumerate(self.layers):
            torch.cuda.nvtx.range_push(f"layer={layer_idx}")
            hidden_states = decoder_layer(hidden_states, kvCachePool, prefillBatchPosition, decodeBatchPosition, loraWeight)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("lastnorm")
        hidden_states = self.norm(hidden_states)
        torch.cuda.nvtx.range_pop()

        return hidden_states


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.nvtx.range_push("LlamaForCausalLM")
        hidden_states = self.model(input_ids, kvCachePool, prefillBatchPosition, decodeBatchPosition, loraWeight)
        torch.cuda.nvtx.range_push("lm_head")
        logits = self.lm_head(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        return logits, hidden_states
