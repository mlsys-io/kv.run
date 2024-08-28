from typing import Tuple
import rotary_emb  # flash attention's rotary implementation
import torch


# cos: seq_len x 1 x rotary_dim
# sin: seq_len x 1 x rotary_dim
# query: seq_len x num_attention_heads x head_dim
# key: seq_len x num_kv_heads x head_dim
# is_neox: whether to use GPT-NeoX's rotary implementation OR GPT-J's rotary implementation
def rotate_query_key_in_place(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool,
):
    rotate_in_place(query, cos, sin, is_neox)
    rotate_in_place(key, cos, sin, is_neox)


def rotate_in_place(x, cos, sin, is_neox):
    rotary_dim = cos.shape[-1] * 2
    if is_neox:
        x1 = x[..., :rotary_dim]
        x2 = x[..., rotary_dim : 2 * rotary_dim]
        rotary_emb.apply_rotary(x1, x2, cos, sin, x1, x2, False)
    else:
        even_positions = list(range(0, rotary_dim, 2))
        odd_positions = list(range(1, rotary_dim, 2))
        x1 = x[..., even_positions]
        x2 = x[..., odd_positions]
        rotary_emb.apply_rotary(x1, x2, cos, sin, x1, x2, False)
        x[..., :rotary_dim] = torch.stack((x1, x2), dim=-1).flatten(start_dim=-2)


def getPositionIdsAndMaxSeqLenForPrefill(
    seq_lens: torch.Tensor, device
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


def getPositionIdsAndMaxSeqLenForDecode(
    seq_lens: torch.Tensor, device
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


def get_cos_sin(rotary_emb, seq_lens: torch.Tensor, device, dtype, is_prefill):
    position_ids, max_seq_len = (
        getPositionIdsAndMaxSeqLenForPrefill(seq_lens, device)
        if is_prefill
        else getPositionIdsAndMaxSeqLenForDecode(seq_lens, device)
    )

    return rotary_emb.get_cos_sin(position_ids, max_seq_len, dtype)
