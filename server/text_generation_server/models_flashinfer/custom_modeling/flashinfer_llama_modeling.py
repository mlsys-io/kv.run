from typing import Tuple
from text_generation_server.layers.rotary import PositionRotaryEmbedding
import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    ACT2FN,
    LlamaConfig,
)

from punica_kernels import (
    rms_norm,
)
from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear,
)
from text_generation_server.utils import Weights
from text_generation_server.utils.lora_utils import BatchedModelLoraWeight
from text_generation_server.utils.cache_manager_flashinfer import (
    KvCachePool,
    KvCacheBatchPosition,
)
from text_generation_server.layers.flashinfer_attention import (
    POS_ENCODING_MODE,
    FlashinferAttentionWrapper,
    AttentionRotaryParams,
)


def _load_attention(config, prefix, weights):
    # Only defined in granite.
    bias = getattr(config, "attention_bias", False)

    # if specific model type, load the correct attention
    if config.model_type == "phi3":
        return TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.qkv_proj",
            weights=weights,
            bias=bias,
        )
    elif config.model_type == "baichuan":
        return TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.W_pack",
            weights=weights,
            bias=bias,
        )

    # otherwise, load the default attention based on the number of heads
    return TensorParallelColumnLinear.load_multi(
        config,
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        dim=0,
        weights=weights,
        bias=bias,
    )


class FlashinferBatch:
    def __init__(self, seq_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len):
        self.seq_indptr = seq_indptr
        self.kv_page_indptr = kv_page_indptr
        self.kv_page_indices = kv_page_indices
        self.kv_last_page_len = kv_last_page_len


class FlashLlamaAttention(nn.Module):
    def __init__(
        self,
        prefix: str,
        flashinferWrapper: FlashinferAttentionWrapper,
        config: LlamaConfig,
        weights,
        layer_idx: int,
    ):
        super().__init__()
        self.flashinferWrapper = flashinferWrapper
        self.rotaryParams = AttentionRotaryParams(
            pos_encoding_mode=POS_ENCODING_MODE.NONE
        )
        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=flashinferWrapper.head_dim,
            base=config.rope_theta,
            device=weights.device,
        )
        self.layer_idx = layer_idx
        self.qkv_proj = _load_attention(config, prefix, weights)
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
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        cos: torch.Tensor,
        sin: torch.Tensor,
        loraWeight: BatchedModelLoraWeight,
    ) -> torch.Tensor:
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
        if loraWeight:
            loraWeight.apply_lora_weight_kvq(q, k, v, hidden_states, self.layer_idx)

        self.rotary_emb(
            q.view(
                -1,
                self.flashinferWrapper.num_attention_heads,
                self.flashinferWrapper.head_dim,
            ),
            k.view(
                -1,
                self.flashinferWrapper.num_key_value_heads,
                self.flashinferWrapper.head_dim,
            ),
            cos,
            sin,
        )

        attn_outputs_raw = self.flashinferWrapper.computeAttention(
            q,
            k,
            v,
            kvCachePool.cache_data[self.layer_idx],
            is_prefill,
            batch_position,
            self.rotaryParams,
        )
        attn_outputs = self.o_proj(attn_outputs_raw)
        if loraWeight:
            loraWeight.apply_lora_weight_attn(
                attn_outputs, attn_outputs_raw, self.layer_idx
            )
        return attn_outputs


class LlamaMLP(nn.Module):
    def __init__(
        self, prefix: str, config: LlamaConfig, weights: Weights, layer_idx: int
    ):
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
        if config.model_type == "phi3":
            self.gate_up_proj = TensorParallelColumnLinear.load_gate_up(
                config,
                prefix=f"{prefix}.gate_up_proj",
                weights=weights,
                bias=False,
            )
        else:
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

    def forward(
        self, hidden_states: torch.Tensor, loraWeight: BatchedModelLoraWeight
    ) -> torch.Tensor:
        gate_up_states = self.gate_up_proj(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        gate = gate_up_states[:, 0].contiguous()
        if loraWeight:
            loraWeight.apply_lora_weight_gate(gate, hidden_states, self.layer_idx)
        gate = self.act(gate)
        up = gate_up_states[:, 1].contiguous()
        if loraWeight:
            loraWeight.apply_lora_weight_up(up, hidden_states, self.layer_idx)
        t = gate * up
        down = self.down_proj(t)
        if loraWeight:
            loraWeight.apply_lora_weight_down(down, t, self.layer_idx)
        return down


class RMSNorm(nn.Module):
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


class FlashLlamaLayer(nn.Module):
    def __init__(
        self,
        prefix: str,
        flashinferWrapper: FlashinferAttentionWrapper,
        config: LlamaConfig,
        weights: Weights,
        layer_idx: int,
    ):
        super().__init__()
        self.self_attn = FlashLlamaAttention(
            prefix=f"{prefix}.self_attn",
            flashinferWrapper=flashinferWrapper,
            config=config,
            weights=weights,
            layer_idx=layer_idx,
        )
        self.mlp = LlamaMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights, layer_idx=layer_idx
        )
        self.input_layernorm = RMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )

        self.post_attention_layernorm = RMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        kvCachePool: KvCachePool,
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        cos: torch.Tensor,
        sin: torch.Tensor,
        loraWeight: BatchedModelLoraWeight,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)
        attn_output = self.self_attn(
            normed_hidden_states,
            kvCachePool,
            is_prefill,
            batch_position,
            cos,
            sin,
            loraWeight,
        )

        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output, loraWeight)
        return mlp_output, attn_res


class FlashLlamaModel(torch.nn.Module):
    def __init__(self, prefix: str, config: LlamaConfig, weights: Weights):
        super().__init__()
        assert config.num_attention_heads % weights.process_group.size() == 0
        assert config.num_key_value_heads % weights.process_group.size() == 0
        num_attention_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()

        self.flashinferWrapper = FlashinferAttentionWrapper(
            num_attention_heads, num_key_value_heads, config.hidden_size
        )

        self.layers = nn.ModuleList(
            [
                FlashLlamaLayer(
                    prefix=(
                        f"model.layers.{layer_id}"
                        if not prefix
                        else f"{prefix}.model.layers.{layer_id}"
                    ),
                    flashinferWrapper=self.flashinferWrapper,
                    config=config,
                    weights=weights,
                    layer_idx=layer_id,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        kvCachePool: KvCachePool,
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        position_ids, max_seq_len = (
            self._getPositionIdsAndMaxSeqLenForPrefill(
                batch_position.seq_lens, hidden_states.device
            )
            if is_prefill
            else self._getPositionIdsAndMaxSeqLenForDecode(
                batch_position.seq_lens, hidden_states.device
            )
        )

        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, max_seq_len, hidden_states.dtype
        )
        residual = None

        self.flashinferWrapper.prepareAttention(
            is_prefill,
            batch_position,
            kvCachePool.page_len,
            POS_ENCODING_MODE.NONE,
            kvCachePool.cache_data[0].dtype,
        )
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                kvCachePool,
                is_prefill,
                batch_position,
                cos,
                sin,
                loraWeight,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        self.flashinferWrapper.endBatchAttention(is_prefill)
        return hidden_states

    def _getPositionIdsAndMaxSeqLenForPrefill(
        self, seq_lens: torch.Tensor, device
    ) -> Tuple[torch.Tensor, int]:
        if seq_lens.numel() == 0:
            return torch.tensor([], dtype=torch.int32, device=device), 0
        position_ids = torch.cat(
            [
                torch.arange(seq_len, dtype=torch.int32, device=device)
                for seq_len in seq_lens
            ]
        )
        max_seq_len = torch.max(seq_lens).item()
        return position_ids, max_seq_len

    def _getPositionIdsAndMaxSeqLenForDecode(
        self, seq_lens: torch.Tensor, device
    ) -> Tuple[torch.Tensor, int]:
        if seq_lens.numel() == 0:
            return torch.tensor([], dtype=torch.int32, device=device), 0
        position_ids = torch.cat(
            [
                torch.tensor([seq_len - 1], dtype=torch.int32, device=device)
                for seq_len in seq_lens
            ]
        )
        max_seq_len = torch.max(seq_lens).item()
        return position_ids, max_seq_len


class FlashLlamaForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config: LlamaConfig, weights: Weights):
        super().__init__()

        self.embed_tokens = TensorParallelEmbedding(
            prefix=(
                "model.embed_tokens" if not prefix else f"{prefix}.model.embed_tokens"
            ),
            weights=weights,
        )
        self.model = FlashLlamaModel(prefix, config, weights)
        self.lm_head = SpeculativeHead.load(
            config,
            prefix="lm_head" if not prefix else f"{prefix}.lm_head",
            weights=weights,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool,
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.model(
            inputs_embeds,
            kvCachePool,
            is_prefill,
            batch_position,
            loraWeight,
        )
        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits
