# Editor: Alfred Gui

import pytest
import torch
import math
from torch.nn import functional as F
from typing import List

numHead = 2
headDim = 128
page_len = 16
device = torch.device("cuda:0")
dtype = torch.bfloat16

def assert_close(a, b):
    rtol, atol = {
        torch.float16: (1e-3, 5e-4),
        torch.float32: (1e-5, 5e-6),
        torch.bfloat16: (8e-3, 8e-3),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

## q: (seqLen, numHead, headDim)
## k: (seqLen, numHead, headDim)
## v: (seqLen, numHead, headDim)
def batch_prefill_baseline(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqLens: List[int]):
    attns = []
    startingIndex = 0
    for seqLen in seqLens:
        seqSlice = slice(startingIndex, startingIndex + seqLen)
        qs = q[seqSlice]
        ks = k[seqSlice]
        vs = v[seqSlice]

        qs = rotary_embed(qs.transpose(0, 1), 0).transpose(0, 1)
        ks = rotary_embed(ks.transpose(0, 1), 0).transpose(0, 1)
        attns.append(prefill_single_seq_attn(qs, ks, vs, seqLen))
        startingIndex += seqLen
    return torch.cat(attns, dim=0) 

## q: (seqLen, numHead, headDim)
## k: (seqLen, numHead, headDim)
## v: (seqLen, numHead, headDim)
def prefill_single_seq_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqLen: int):
    qt = q.transpose(1, 0) # (numHead, seqLen, headDim)
    kt = k.transpose(1, 0) # (numHead, seqLen, headDim)
    vt = v.transpose(1, 0) # (numHead, seqLen, headDim)
    scale = math.sqrt(kt.size(-1))

    qkProduct = (qt @ kt.transpose(-2, -1)) * (1.0 / scale) # (numHead, seqLen, seqLen)
    causalMask = torch.triu(torch.full((seqLen, seqLen), float('-inf'), dtype=dtype, device=device), diagonal=1) # lower triangular matrix
    softmax = F.softmax(qkProduct + causalMask, dim=-1)
    attn = softmax @ vt # (numHead, seqLen, seqLen) x (numHead, seqLen, headDim) -> (numHead, seqLen, headDim)
    return attn.transpose(0,1) # (seqLen, numHead, headDim)

## q: (numSeqs, numHead, headDim)
## k: (seqLen, numHead, headDim)
## v: (seqLen, numHead, headDim)
def batch_decode_baseline(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqLens: List[int]):
    attns = []
    startingIndex = 0
    for i, seqLen in enumerate(seqLens):
        seqSlice = slice(startingIndex, startingIndex + seqLen)
        qi = rotary_embed(q[i], seqLen - 1) # (numHead, headDim)
        kSlice = rotary_embed(k[seqSlice].transpose(0,1), 0).transpose(0,1) # (seqLen, numHead, headDim)
        attns.append(decode_single_seq_attn(qi, kSlice, v[seqSlice], seqLen))
        startingIndex += seqLen
    return torch.cat(attns, dim=0)

def decode_single_seq_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqLen):
    qt = q.view(numHead, 1, headDim)  # (numHead, 1, headDim)
    kt = k.transpose(1, 0) # (numHead, seqLen, headDim)
    vt = v.transpose(1, 0) # (numHead, seqLen, headDim)
    scale = math.sqrt(headDim)

    qkProduct = (qt @ kt.transpose(-2, -1)) * (1.0 / scale) # (numHead, 1, seqLen)
    # the causal mask is not needed for decoding
    softmax = F.softmax(qkProduct, dim=-1)
    attn = softmax @ vt # (numHead, 1, seqLen) x (numHead, seqLen, headDim) -> (numHead, 1, headDim)
    return attn.transpose(0,1) # (1, numHead, headDim)

def rotary_embed(q, beg):
    device = q.device
    dtype = q.dtype
    dim = q.size(-1)
    l = q.size(-2) if q.dim() == 3 else 1

    base = 1e4
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim)
    )
    t = torch.arange(beg, beg + l, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)