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
from punica_kernels import (
    rms_norm,
)
from text_generation_server.utils.lora_utils import BatchedModelLoraWeightPhi
from text_generation_server.utils.cache_manager_flashinfer import (
    KvCachePool,
    KvCacheBatchPosition,
)
from text_generation_server.layers.flashinfer_attention import (
    POS_ENCODING_MODE,
    FlashinferAttentionWrapper,
    AttentionRotaryParams,
)
from text_generation_server.layers.layernorm import (
    FastLayerNorm,
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
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states
        return rms_norm(hidden_states, self.weight, self.variance_epsilon), residual


class FlashPhiAttention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        flashinferWrapper: FlashinferAttentionWrapper,
        config: PhiConfig,
        weights,
        layer_idx: int,
    ):
        super().__init__()
        self.flashinferWrapper = flashinferWrapper
        self.rotaryParams = AttentionRotaryParams(
            pos_encoding_mode=POS_ENCODING_MODE.NONE
        )
        self.rotary_dim = int(config.partial_rotary_factor * flashinferWrapper.head_dim)
        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.rotary_dim,
            base=config.rope_theta,
            device=weights.device,
        )
        self.layer_idx = layer_idx
        self.qkv_proj = load_attention(config, prefix, weights)
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense",
            weights=weights,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kvCachePool: KvCachePool,
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        cos: torch.Tensor,
        sin: torch.Tensor,
        loraWeight: BatchedModelLoraWeightPhi,
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
            loraWeight.apply_lora_weight_Wkvq(q, k, v, hidden_states, self.layer_idx)

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
            # Note: Bug here, need to fix
            # loraWeight.apply_lora_weight_attn(
            #     attn_outputs, attn_outputs_raw, self.layer_idx
            # )
            pass
        return attn_outputs


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

    # TODO: add lora adapter
    def forward(self, hidden_states, loraWeight: BatchedModelLoraWeightPhi):
        gate_up_states = self.up_proj(hidden_states)
        gate_up_acted = self.act(gate_up_states)
        gate_down_states = self.down_proj(gate_up_acted)

        # NOTE: Llama requires the gate up states to an intermediate size
        # Phi does not and we can avoid the `view` operation
        return gate_down_states


class FlashPhiLayer(nn.Module):
    def __init__(
        self,
        layer_id: str,
        flashinferWrapper: FlashinferAttentionWrapper,
        config: PhiConfig,
        weights,
    ):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashPhiAttention(
            prefix=f"{prefix}.self_attn",
            flashinferWrapper=flashinferWrapper,
            config=config,
            weights=weights,
            layer_idx=layer_id,
        )
        self.mlp = PhiMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights, layer_idx=layer_id
        )

        self.input_layernorm = FastLayerNorm.load(
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
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        cos: torch.Tensor,
        sin: torch.Tensor,
        loraWeight: BatchedModelLoraWeightPhi,
    ):

        hidden_states, res = self.input_layernorm(hidden_states, residual)
        # Self Attention
        attn_output = self.self_attn(
            hidden_states,
            kvCachePool,
            is_prefill,
            batch_position,
            cos,
            sin,
            loraWeight,
        )

        hidden_states = self.resid_dropout(attn_output).add(
            self.resid_dropout(self.mlp(hidden_states, loraWeight))
        )

        return hidden_states, res


class FlashPhiModel(torch.nn.Module):
    def __init__(self, config: PhiConfig, weights):
        super().__init__()

        assert config.num_attention_heads % weights.process_group.size() == 0
        assert config.num_key_value_heads % weights.process_group.size() == 0
        num_attention_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()

        self.flashinferWrapper = FlashinferAttentionWrapper(
            num_attention_heads, num_key_value_heads, config.hidden_size
        )

        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashPhiLayer(
                    layer_id,
                    flashinferWrapper=self.flashinferWrapper,
                    config=config,
                    weights=weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        self.norm = FastLayerNorm.load(
            prefix="model.final_layernorm",
            weights=weights,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool,
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeightPhi,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
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
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeightPhi,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            kvCachePool,
            is_prefill,
            batch_position,
            loraWeight,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]

        return self.lm_head(hidden_states)
