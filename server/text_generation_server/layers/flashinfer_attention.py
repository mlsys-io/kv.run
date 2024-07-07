from dataclasses import dataclass
from enum import Enum
import math
import torch
import flashinfer
from text_generation_server.utils.cache_manager_flashinfer import KvCacheBatchPosition

FLASH_INFER_SUPPORTED_DIMS = [64, 128, 256]


class POS_ENCODING_MODE(Enum):
    ROPE_LLAMA = "ROPE_LLAMA"
    ALIBI = "ALIBI"
    NONE = "NONE"


@dataclass(frozen=True)
class AttentionRotaryParams:
    causal: bool = True
    pos_encoding_mode: POS_ENCODING_MODE = POS_ENCODING_MODE.ROPE_LLAMA
    rope_scale: float = 1.0
    rope_theta: float = 1.0e4


def find_padded_head_dim(head_dim):
    for dim in FLASH_INFER_SUPPORTED_DIMS:
        if head_dim <= dim:
            return dim
    raise ValueError("The head dimension is too large for FlashInfer")


class FlashinferAttentionWrapper:
    def __init__(
        self, num_attention_heads: int, num_key_value_heads: int, hidden_size: int
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self._head_padded_dim = find_padded_head_dim(self.head_dim)
        self.page_size = 16

        self.group_size = self.num_attention_heads // self.num_key_value_heads
        _workspace_buffer = torch.empty(
            32 * 1024 * 1024, dtype=torch.int8, device=torch.cuda.current_device()
        )
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer=_workspace_buffer, kv_layout="NHD"
        )
        _use_tensor_cores = self.group_size in [7, 16]
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer=_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=_use_tensor_cores,
        )

    def computeAttention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cacheData: torch.Tensor,
        page_len: int,
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        rotaryParams: AttentionRotaryParams,
    ):
        stack_attn_output = []
        prefillTotalSeqLen = prefillBatchPosition.total_seq_len
        if prefillTotalSeqLen > 0:
            q = q[:prefillTotalSeqLen].view(
                prefillTotalSeqLen, self.num_attention_heads, self.head_dim
            )
            k = k[:prefillTotalSeqLen].view(
                prefillTotalSeqLen, self.num_key_value_heads, self.head_dim
            )
            v = v[:prefillTotalSeqLen].view(
                prefillTotalSeqLen, self.num_key_value_heads, self.head_dim
            )
            q, k, v = self._pad_qkv(q, k, v)
            attn_output_prefill = self._batchPrefill(
                q, k, v, cacheData, prefillBatchPosition, rotaryParams
            )
            attn_output_prefill = self._unpad_attention(
                attn_output_prefill, prefillTotalSeqLen
            )
            stack_attn_output.append(attn_output_prefill)

        decodeTotalSeqLen = decodeBatchPosition.total_seq_len
        if decodeTotalSeqLen > 0:
            q = q[prefillTotalSeqLen:].view(
                decodeTotalSeqLen, self.num_attention_heads, self.head_dim
            )
            k = k[prefillTotalSeqLen:].view(
                decodeTotalSeqLen, self.num_key_value_heads, self.head_dim
            )
            v = v[prefillTotalSeqLen:].view(
                decodeTotalSeqLen, self.num_key_value_heads, self.head_dim
            )
            q, k, v = self._pad_qkv(q, k, v)
            attn_output_decode = self._batchDecode(
                q, k, v, cacheData, page_len, decodeBatchPosition, rotaryParams
            )
            attn_output_decode = self._unpad_attention(
                attn_output_decode, decodeTotalSeqLen
            )
            stack_attn_output.append(attn_output_decode)
        return (
            stack_attn_output[0]
            if len(stack_attn_output) == 1
            else torch.cat(stack_attn_output, dim=0)
        )

    def _unpad_attention(self, attn_output, seqLen):
        if self._head_padded_dim > self.head_dim:
            return attn_output[:, :, : self.head_dim].reshape(seqLen, self.hidden_size)
        else:
            return attn_output.view(seqLen, self.hidden_size)

    def _pad_qkv(self, q, k, v):
        if self._head_padded_dim > self.head_dim:
            q = torch.nn.functional.pad(q, (0, self._head_padded_dim - self.head_dim))
            k = torch.nn.functional.pad(k, (0, self._head_padded_dim - self.head_dim))
            v = torch.nn.functional.pad(v, (0, self._head_padded_dim - self.head_dim))
        return q, k, v

    def _batchPrefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cacheData: torch.Tensor,
        prefillBatchPosition: KvCacheBatchPosition,
        rotaryParams: AttentionRotaryParams,
    ):

        flashinfer.append_paged_kv_cache(
            k,
            v,
            prefillBatchPosition.seq_indptr,
            cacheData,
            prefillBatchPosition.kv_page_indices,
            prefillBatchPosition.kv_page_indptr,
            prefillBatchPosition.kv_last_page_len,
        )

        self.prefill_wrapper.begin_forward(
            prefillBatchPosition.seq_indptr,
            prefillBatchPosition.kv_page_indptr,
            prefillBatchPosition.kv_page_indices,
            prefillBatchPosition.kv_last_page_len,
            self.num_attention_heads,
            self.num_key_value_heads,
            self._head_padded_dim,
            self.page_size,
        )

        attn_output_prefill = self.prefill_wrapper.forward(
            q,
            cacheData,
            causal=rotaryParams.causal,
            pos_encoding_mode=rotaryParams.pos_encoding_mode.value,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            rope_scale=rotaryParams.rope_scale,
            rope_theta=rotaryParams.rope_theta,
        )

        self.prefill_wrapper.end_forward()
        return attn_output_prefill

    def _batchDecode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cacheData: torch.Tensor,
        page_len: int,
        decodeBatchPosition: KvCacheBatchPosition,
        rotaryParams: AttentionRotaryParams,
    ):

        flashinfer.append_paged_kv_cache(
            k,
            v,
            decodeBatchPosition.seq_indptr,
            cacheData,
            decodeBatchPosition.kv_page_indices,
            decodeBatchPosition.kv_page_indptr,
            decodeBatchPosition.kv_last_page_len,
        )

        self.decode_wrapper.begin_forward(
            decodeBatchPosition.kv_page_indptr,
            decodeBatchPosition.kv_page_indices,
            decodeBatchPosition.kv_last_page_len,
            self.num_attention_heads,
            self.num_key_value_heads,
            self._head_padded_dim,
            page_len,
            pos_encoding_mode=rotaryParams.pos_encoding_mode.value,
            data_type=cacheData.dtype,
        )

        attn_output_decode = self.decode_wrapper.forward(
            q,
            cacheData,
            pos_encoding_mode=rotaryParams.pos_encoding_mode.value,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            rope_scale=rotaryParams.rope_scale,
            rope_theta=rotaryParams.rope_theta,
        )

        self.decode_wrapper.end_forward()
        return attn_output_decode
