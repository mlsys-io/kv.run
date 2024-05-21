import torch

import punica.ops._kernels as _kernels

def bgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    w_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
    """
    Semantics:
      y[i] += (
          x[i].unsqueeze(0)
          @ w_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          * scale
        ).squeeze(0)

    Args:
      y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
      x: Shape: `[B, H1]`. Input vectors.
      w_T_all: Shape: `[None, L, H2, H1]`. All of the transposed weight matrices.
      indicies: Shape: `[B]`. Indices of the weight matrices.
      layer_idx: Layer index of the weight matrices.
      scale: Scaling factor.
    """
    f = _kernels.dispatch_bgmv
    f(y, x, w_T_all, indicies, layer_idx, scale)


def add_lora_bgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_T_all: torch.Tensor,
    wb_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
    """
    Semantics:
      y[i] += (
          x[i].unsqueeze(0)
          @ wa_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          @ wb_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          * scale
        ).squeeze(0)

    Args:
      y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
      x: Shape: `[B, H1]`. Input vectors.
      wa_T_all: Shape: `[None, L, R, H1]`. All of the transposed LoRA A matrices.
      wb_T_all: Shape: `[None, L, H2, R]`. All of the transposed LoRA B matrices.
      indicies: Shape: `[B]`. Indices of the LoRA weights.
      layer_idx: Layer index of LoRA weights.
      scale: Scaling factor.
    """
    f = _kernels.dispatch_bgmv
    device = x.device
    dtype = x.dtype

    r = wb_T_all.size(-1)
    tmp = torch.zeros((x.size(0), r), dtype=dtype, device=device)
    f(tmp, x, wa_T_all, indicies, layer_idx, 1.0)
    f(y, tmp, wb_T_all, indicies, layer_idx, scale)


def sgmv_cutlass(
    y: torch.Tensor,
    x: torch.Tensor,
    w_ptr: torch.Tensor,
    s: torch.Tensor,
    layer_idx: int,
):
    """
  Semantics:
    y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(w_ptr[i])

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    w_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, H1, H2]`.
    s: Shape: `[S+1]`, DType: torch.int32. Indptr of the weight matrices.\
      `s[0] == 0`, `s[-1] == B`.
    layer_idx: Layer index of the weight matrices.
  """
    tmp_size = _kernels.sgmv_cutlass_tmp_size(w_ptr.size(0))
    tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
    _kernels.sgmv_cutlass(y, x, w_ptr, s, tmp, layer_idx)


def add_lora_sgmv_cutlass(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s: torch.Tensor,
    layer_idx: int,
    lora_rank: int,
):
    """
  Semantics:
    y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(wa_ptr[i]) @ deref(wb_ptr[i])

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    wa_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, H1, R]`.
    wb_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, R, H2]`.
    s: Shape: `[S+1]`, DType: torch.int32. Indptr of the weight matrices.\
      `s[0] == 0`, `s[-1] == B`.
    layer_idx: Layer index of the weight matrices.
  """
    tmp_size = _kernels.sgmv_cutlass_tmp_size(wa_ptr.size(0))
    tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    _kernels.sgmv_cutlass(v, x, wa_ptr, s, tmp, layer_idx)
    _kernels.sgmv_cutlass(y, v, wb_ptr, s, tmp, layer_idx)


def add_lora_sgmv_custom_cutlass(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s: torch.Tensor,
    layer_idx: int,
    lora_rank: int,
):
    """
  Semantics:
    y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(wa_ptr[i]).T @ deref(wb_ptr[i])

  Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    x: Shape: `[B, H1]`. Input vectors.
    wa_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, R, H1]`.
    wb_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
      Weight matrix shape: `[num_layers, R, H2]`.
    s: Shape: `[S+1]`, DType: torch.int32. Indptr of the weight matrices.\
      `s[0] == 0`, `s[-1] == B`.
    layer_idx: Layer index of the weight matrices.
  """
    tmp1 = torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=x.device)
    tmp2_size = _kernels.sgmv_cutlass_tmp_size(wa_ptr.size(0))
    tmp2 = torch.empty((tmp2_size,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    _kernels.sgmv_shrink(v, x, wa_ptr, s, tmp1, layer_idx)
    _kernels.sgmv_cutlass(y, v, wb_ptr, s, tmp2, layer_idx)


def sgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    w_ptr: torch.Tensor,
    s: torch.Tensor,
    layer_idx: int,
):
    if x.size(1) < y.size(1):
        raise NotImplementedError("TODO: sgmv_expand")
    tmp = torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=x.device)
    _kernels.sgmv_shrink(y, x, w_ptr, s, tmp, layer_idx)


def add_lora_sgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s: torch.Tensor,
    layer_idx: int,
    lora_rank: int,
):
    tmp = torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    _kernels.sgmv_shrink(v, x, wa_ptr, s, tmp, layer_idx)
    raise NotImplementedError("TODO: sgmv_expand")


def rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
):
    o = torch.empty_like(x)
    _kernels.rms_norm(o, x, w, eps)
    return o
