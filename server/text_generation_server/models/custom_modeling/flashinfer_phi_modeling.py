import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List, Tuple

from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear,
)
from text_generation_server.layers.rotary import (
    PositionRotaryEmbedding,
)
import flashinfer
from punica_kernels import (
    add_lora_sgmv_custom_cutlass as add_lora,
    rms_norm,
)
from text_generation_server.utils.lora_utils import BatchedModelLoraWeight
from text_generation_server.utils.cache_manager_flashinfer import (
    KvCachePool,
    KvCacheBatchPosition,
)


class FlashinferBatch:
    def __init__(self, seq_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len):
        self.seq_indptr = seq_indptr
        self.kv_page_indptr = kv_page_indptr
        self.kv_page_indices = kv_page_indices
        self.kv_last_page_len = kv_last_page_len


class PhiConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=51200,
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="gelu_fast",  # llama uses silu
        layer_norm_eps=1e-05,  # rms in llama,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        resid_pdrop=0.1,  # llama doesn't have this
        partial_rotary_factor=0.5,  # important difference between llama and phi
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.resid_pdrop = resid_pdrop
        self.partial_rotary_factor = partial_rotary_factor

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# this is the same as llama except for Phi uses bias=True
def load_attention(config, prefix, weights):
    if config.num_attention_heads != config.num_key_value_heads:
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
    assert config.hidden_size % config.num_attention_heads == 0
    assert config.num_attention_heads % weights.process_group.size() == 0

    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        quantize=config.quantize,
        dim=0,
    )

    if config.quantize not in ["gptq", "awq"]:
        weight = weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.hidden_size // config.num_attention_heads
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()
        assert list(weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.shape)} != {[(num_heads + 2 * config.num_key_value_heads) * head_size, config.hidden_size]}"

    # this is the same as llama except for Phi uses bias=True
    return TensorParallelColumnLinear(
        get_linear(weight, bias=True, quantize=config.quantize)
    )


class PhiLayerNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        bias = weights.get_tensor(f"{prefix}.bias")
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states
        return rms_norm(hidden_states, self.weight, self.variance_epsilon), residual


class FlashPhiAttention(torch.nn.Module):
    def __init__(self, prefix: str, config: PhiConfig, weights, layer_idx: int):
        super().__init__()
        self.num_qo_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_qo_heads
        self.head_padded_dim = self._find_padded_head_dim(self.head_dim)
        self.softmax_scale = self.head_dim**-0.5
        self.rotary_dim = int(config.partial_rotary_factor * self.head_dim)
        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.rotary_dim,
            base=config.rope_theta,
            device=weights.device,
        )
        if self.num_qo_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )

        self.num_qo_heads = self.num_qo_heads // weights.process_group.size()
        self.num_kv_heads = config.num_key_value_heads // weights.process_group.size()

        self.query_key_value = load_attention(config, prefix, weights)

        self.dense = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense",
            weights=weights,
            bias=True,
        )
        self.layer_idx = layer_idx
        self.num_groups = self.num_qo_heads // self.num_kv_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        kvCachePool: KvCachePool,
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        cos: torch.Tensor,
        sin: torch.Tensor,
        lora: BatchedModelLoraWeight | None,
    ) -> torch.Tensor:
        qkv = self.query_key_value(hidden_states)
        q_proj, k_proj, v_proj = qkv.split(
            [
                self.head_dim * self.num_qo_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=1,
        )

        q_proj = q_proj.contiguous()
        k_proj = k_proj.contiguous()
        v_proj = v_proj.contiguous()

        if lora:
            add_lora(
                q_proj,
                hidden_states,
                lora.q.wa_ptr,
                lora.q.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )
            add_lora(
                k_proj,
                hidden_states,
                lora.k.wa_ptr,
                lora.k.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )
            add_lora(
                v_proj,
                hidden_states,
                lora.v.wa_ptr,
                lora.v.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )

        self.rotary_emb(
            q_proj.view(-1, self.num_qo_heads, self.head_dim)[:, :, : self.rotary_dim],
            k_proj.view(-1, self.num_kv_heads, self.head_dim)[:, :, : self.rotary_dim],
            cos,
            sin,
        )

        stack_attn_output = []
        workspace_buffer = torch.empty(
            32 * 1024 * 1024, dtype=torch.int8, device=kvCachePool.device
        )
        prefillTotalSeqLen = prefillBatchPosition.total_seq_len
        if prefillTotalSeqLen > 0:
            # need to revisit if contiguous conversion is the best way
            q_raw = q_proj[:prefillTotalSeqLen].view(
                prefillTotalSeqLen, self.num_qo_heads, self.head_dim
            )
            k_raw = k_proj[:prefillTotalSeqLen].view(
                prefillTotalSeqLen, self.num_kv_heads, self.head_dim
            )
            v_raw = v_proj[:prefillTotalSeqLen].view(
                prefillTotalSeqLen, self.num_kv_heads, self.head_dim
            )
            q = torch.nn.functional.pad(
                q_raw, (0, self.head_padded_dim - self.head_dim)
            )
            k = torch.nn.functional.pad(
                k_raw, (0, self.head_padded_dim - self.head_dim)
            )
            v = torch.nn.functional.pad(
                v_raw, (0, self.head_padded_dim - self.head_dim)
            )

            seq_indptr = prefillBatchPosition.seq_indptr.clone()
            kv_page_indices = prefillBatchPosition.kv_page_indices.clone()
            kv_page_indptr = prefillBatchPosition.kv_page_indptr.clone()
            kv_last_page_len = prefillBatchPosition.kv_last_page_len.clone()

            flashinfer.append_paged_kv_cache(
                k,
                v,
                seq_indptr,
                kvCachePool.cache_data[self.layer_idx],
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_len,
            )

            prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer, "NHD"
            )

            prefill_wrapper.begin_forward(
                seq_indptr,
                kv_page_indptr,
                kv_page_indices,
                kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_padded_dim,
            )

            attn_output_prefill = prefill_wrapper.forward(
                q,
                kvCachePool.cache_data[self.layer_idx],
                causal=True,
                pos_encoding_mode="NONE",
            )[:, :, : self.head_dim].reshape(prefillTotalSeqLen, self.hidden_size)
            prefill_wrapper.end_forward()
            stack_attn_output.append(attn_output_prefill)

        decodeTotalSeqLen = decodeBatchPosition.total_seq_len
        if decodeTotalSeqLen > 0:
            q_raw = (
                q_proj[prefillTotalSeqLen:]
                .view(decodeTotalSeqLen, self.num_qo_heads, self.head_dim)
                .contiguous()
            )
            k_raw = (
                k_proj[prefillTotalSeqLen:]
                .view(decodeTotalSeqLen, self.num_kv_heads, self.head_dim)
                .contiguous()
            )
            v_raw = (
                v_proj[prefillTotalSeqLen:]
                .view(decodeTotalSeqLen, self.num_kv_heads, self.head_dim)
                .contiguous()
            )

            q = torch.nn.functional.pad(
                q_raw, (0, self.head_padded_dim - self.head_dim)
            )
            k = torch.nn.functional.pad(
                k_raw, (0, self.head_padded_dim - self.head_dim)
            )
            v = torch.nn.functional.pad(
                v_raw, (0, self.head_padded_dim - self.head_dim)
            )

            flashinfer.append_paged_kv_cache(
                k,
                v,
                decodeBatchPosition.seq_indptr,
                kvCachePool.cache_data[self.layer_idx],
                decodeBatchPosition.kv_page_indices,
                decodeBatchPosition.kv_page_indptr,
                decodeBatchPosition.kv_last_page_len,
            )

            decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer, "NHD"
            )
            decode_wrapper.begin_forward(
                decodeBatchPosition.kv_page_indptr,
                decodeBatchPosition.kv_page_indices,
                decodeBatchPosition.kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_padded_dim,
                kvCachePool.page_len,
                pos_encoding_mode="NONE",
            )

            attn_output_decode = decode_wrapper.forward(
                q, kvCachePool.cache_data[self.layer_idx], pos_encoding_mode="NONE"
            )[:, :, : self.head_dim].reshape(decodeTotalSeqLen, self.hidden_size)

            decode_wrapper.end_forward()
            stack_attn_output.append(attn_output_decode)

        if len(stack_attn_output) == 1:
            attn_outputs = stack_attn_output[0]
        else:
            attn_outputs = torch.cat(stack_attn_output, dim=0)

        # output projection
        o = self.dense(attn_outputs)
        return o

    def _find_padded_head_dim(self, head_dim):
        flashInferDimensions = [64, 128, 256]
        for dim in flashInferDimensions:
            if head_dim <= dim:
                return dim
        raise ValueError("The head dimension is too large for FlashInfer")


class PhiMLP(nn.Module):
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

        # llama weights are up_proj and down_proj and bias=False
        self.up_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.fc1",
            weights=weights,
            bias=True,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.fc2",
            weights=weights,
            bias=True,
        )

    def forward(self, hidden_states, lora: BatchedModelLoraWeight | None):
        gate_up_states = self.up_proj(hidden_states)
        if lora:
            add_lora(
                gate_up_states,
                hidden_states,
                lora.gate.wa_ptr,
                lora.gate.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )

        gate_up_acted = self.act(gate_up_states)
        if lora:
            add_lora(
                gate_up_acted,
                hidden_states,
                lora.gate.wa_ptr,
                lora.gate.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )

        gate_down_states = self.down_proj(gate_up_acted)
        if lora:
            add_lora(
                gate_down_states,
                hidden_states,
                lora.gate.wa_ptr,
                lora.gate.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )

        # NOTE: Llama requires the gate up states to an intermediate size
        # Phi does not and we can avoid the `view` operation
        return gate_down_states


class FlashPhiLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashPhiAttention(
            prefix=f"{prefix}.self_attn",
            config=config,
            weights=weights,
            layer_idx=layer_id,
        )
        self.mlp = PhiMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights, layer_idx=layer_id
        )
        self.input_layernorm = PhiLayerNorm(
            prefix=f"{prefix}.input_layernorm",
            weights=weights,
            eps=config.layer_norm_eps,
        )
        self.resid_dropout = torch.nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        kvCachePool: KvCachePool,
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        cos: torch.Tensor,
        sin: torch.Tensor,
        lora: BatchedModelLoraWeight | None,
    ):

        hidden_states, res = self.input_layernorm(hidden_states, residual)
        # Self Attention
        attn_output = self.self_attn(
            hidden_states,
            kvCachePool,
            prefillBatchPosition,
            decodeBatchPosition,
            cos,
            sin,
            lora,
        )

        hidden_states = self.resid_dropout(attn_output).add(
            self.resid_dropout(self.mlp(hidden_states, lora))
        )

        return hidden_states, res


class FlashPhiModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashPhiLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_dim
        self.num_heads = self.layers[0].self_attn.num_qo_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_kv_heads

        self.norm = PhiLayerNorm(
            prefix="model.final_layernorm",
            weights=weights,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool,
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        lora: BatchedModelLoraWeight | None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        position_ids_prefill, max_seq_len_prefill = self._getPositionIdsAndMaxSeqLen(
            prefillBatchPosition.seq_indptr, hidden_states.device
        )
        position_ids_decode, max_seq_len_decode = self._getPositionIdsAndMaxSeqLen(
            decodeBatchPosition.seq_indptr, hidden_states.device
        )
        position_ids = torch.cat([position_ids_prefill, position_ids_decode])
        max_seq_len = max(max_seq_len_prefill, max_seq_len_decode)

        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, max_seq_len, hidden_states.dtype
        )
        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                kvCachePool,
                prefillBatchPosition,
                decodeBatchPosition,
                cos,
                sin,
                lora,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def _getPositionIdsAndMaxSeqLen(
        self, seq_indptr: torch.Tensor, device
    ) -> Tuple[torch.Tensor, int]:
        seq_lens = torch.diff(seq_indptr)
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


class FlashPhiForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.model = FlashPhiModel(config, weights)
        self.lm_head = SpeculativeHead.load(
            config,
            prefix="lm_head",
            weights=weights,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool,
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        lora: BatchedModelLoraWeight | None,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, kvCachePool, prefillBatchPosition, decodeBatchPosition, lora
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]

        return self.lm_head(hidden_states)
