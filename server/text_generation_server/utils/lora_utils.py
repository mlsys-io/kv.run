from collections.abc import Sequence

import torch
import re

class LoraWeight:
    def __init__(
        self,
        num_layers: int,
        in_features: int,
        out_features: int,
        lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        # SGMV-Shrink custom CUDA kernel uses column-major.
        self.wa = torch.zeros(
            (num_layers, lora_rank, in_features), dtype=dtype, device=device
        )
        # SGMV-Expand cutlass kernel uses row-major.
        self.wb = torch.zeros(
            (num_layers, lora_rank, out_features), dtype=dtype, device=device
        )

    def copy_from_tensor(self, a: torch.Tensor, b: torch.Tensor):
        """
        Copy from column-major weight tensors.

        Args:
          a: Shape: `[num_layers, lora_rank, in_features]`.
          b: Shape: `[num_layers, out_features, lora_rank]`.
        """
        self.wa.copy_(a.to(self.wa.device).to(self.wa.dtype))
        self.wb.copy_(b.to(self.wb.device).to(self.wb.dtype).transpose(1, 2))

    @property
    def device(self) -> torch.device:
        return self.wa.device

    @property
    def dtype(self) -> torch.dtype:
        return self.wa.dtype

    @property
    def num_layers(self) -> int:
        return self.wa.size(0)

    @property
    def in_features(self) -> int:
        return self.wa.size(2)

    @property
    def out_features(self) -> int:
        return self.wb.size(2)

    @property
    def lora_rank(self) -> int:
        return self.wa.size(1)


class BatchedLoraWeight:
    def __init__(self, weights: Sequence[LoraWeight]):
        assert len(weights) > 0
        device = weights[0].device
        self.wa_ptr = torch.tensor(
            [w.wa.data_ptr() for w in weights], dtype=torch.int64, device=device
        )
        self.wb_ptr = torch.tensor(
            [w.wb.data_ptr() for w in weights], dtype=torch.int64, device=device
        )

def convert_lora_weight(peft_weight_path):
    weights = torch.load(
        peft_weight_path, map_location=torch.device("cpu"), weights_only=True
    )
    projs = set()
    num_layers = 0
    rank = 0
    tmp = {}
    for key, value in weights.items():
        layer, proj, ab = re.findall(
            r"\.(\d+)\..*\.(\w+)_proj\.lora_(A|B)\.weight$", key
        )[0]
        ab = ab.upper()
        layer = int(layer)
        projs.add(proj)
        # PyTorch Linear layer is column-major
        if ab == "A":
            assert value.size(0) < value.size(1)
            r = value.size(0)
        elif ab == "B":
            assert value.size(0) > value.size(1)
            r = value.size(1)
        else:
            raise KeyError(f"Unknown weight key: {key}")
        if rank != 0:
            assert r == rank
        else:
            rank = r
        num_layers = max(num_layers, layer + 1)
        tmp[(layer, proj, ab)] = value

    out = {}
    for proj in projs:
        for ab in "AB":
            tensors = []
            for layer in range(num_layers):
                tensors.append(tmp[(layer, proj, ab)])
            out[f"{proj}.{ab}"] = torch.stack(tensors)

    return out