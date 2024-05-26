from collections.abc import Sequence
from typing import Dict, List

import peft
import torch
from huggingface_hub import hf_hub_download
from loguru import logger

class ModelConfigForLora:
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads 


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

class ModelLoraWeight:
    def __init__(
        self,
        modelConfig: ModelConfigForLora,
        lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        num_kv_group = modelConfig.num_qo_heads // modelConfig.num_kv_heads
        self.q = LoraWeight(
            modelConfig.num_hidden_layers,
            modelConfig.hidden_size,
            modelConfig.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.k = LoraWeight(
            modelConfig.num_hidden_layers,
            modelConfig.hidden_size,
            modelConfig.hidden_size // num_kv_group,
            lora_rank,
            dtype,
            device,
        )
        self.v = LoraWeight(
            modelConfig.num_hidden_layers,
            modelConfig.hidden_size,
            modelConfig.hidden_size // num_kv_group,
            lora_rank,
            dtype,
            device,
        )
        self.o = LoraWeight(
            modelConfig.num_hidden_layers,
            modelConfig.hidden_size,
            modelConfig.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.gate = LoraWeight(
            modelConfig.num_hidden_layers,
            modelConfig.hidden_size,
            modelConfig.intermediate_size,
            lora_rank,
            dtype,
            device,
        )
        self.up = LoraWeight(
            modelConfig.num_hidden_layers,
            modelConfig.hidden_size,
            modelConfig.intermediate_size,
            lora_rank,
            dtype,
            device,
        )
        self.down = LoraWeight(
            modelConfig.num_hidden_layers,
            modelConfig.intermediate_size,
            modelConfig.hidden_size,
            lora_rank,
            dtype,
            device,
        )

    def copy_from_tensors(self, ts: dict[str, torch.Tensor]):
        self.q.copy_from_tensor(ts["q.A"], ts["q.B"])
        self.k.copy_from_tensor(ts["k.A"], ts["k.B"])
        self.v.copy_from_tensor(ts["v.A"], ts["v.B"])
        self.o.copy_from_tensor(ts["o.A"], ts["o.B"])
        self.gate.copy_from_tensor(ts["gate.A"], ts["gate.B"])
        self.up.copy_from_tensor(ts["up.A"], ts["up.B"])
        self.down.copy_from_tensor(ts["down.A"], ts["down.B"])
  
class BatchedModelLoraWeight:
    def __init__(self, weights: List[ModelLoraWeight], lens: List[int]):
        assert len(weights) == len(lens)
        device = weights[0].q.wa.device
        self.q = BatchedLoraWeight([w.q for w in weights])
        self.k = BatchedLoraWeight([w.k for w in weights])
        self.v = BatchedLoraWeight([w.v for w in weights])
        self.o = BatchedLoraWeight([w.o for w in weights])
        self.gate = BatchedLoraWeight([w.gate for w in weights])
        self.up = BatchedLoraWeight([w.up for w in weights])
        self.down = BatchedLoraWeight([w.down for w in weights])
        self.segment = torch.cumsum(
            torch.tensor([0] + lens, dtype=torch.int32, device=device),
            dim=0,
            dtype=torch.int32,
        )
        self.rank = weights[0].q.lora_rank
        
class ModelLoraManager:
    def __init__(self, model_config: ModelConfigForLora, dtype, device: torch.device):
        self.lora_weights: Dict[str, ModelLoraWeight] = {}
        self.defalut_rank = 16
        self.lora_weights["empty"] = ModelLoraWeight(
                model_config, self.defalut_rank, dtype, device
            )
        
    def set_lora_weights(
            self,
            lora_id_path_dict: Dict[str, str],
            model_config: ModelConfigForLora,
            device: torch.device,
            dtype=torch.float16,
            ):
        for lora_id, lora_path in lora_id_path_dict.items():
            if lora_id not in self.lora_weights:
                try:
                    model_path = hf_hub_download(lora_path, filename='adapter_model.bin')
                except:
                    from safetensors.torch import load_file
                    model_path = hf_hub_download(lora_path, filename='adapter_model.safetensors')
                    tmp = load_file(model_path, device="cpu")
                    model_path = model_path.replace('.safetensors', '.bin')
                    torch.save(tmp, model_path)
                raw_weights = torch.load(model_path, map_location=device, weights_only=True)
                config_path = hf_hub_download(lora_path, filename='adapter_config.json')
                lora_rank = peft.config.PeftConfigMixin.from_json_file(config_path)['r']
                lora_weight = ModelLoraWeight(model_config, lora_rank*2, dtype, device) \
                    if lora_rank < 16 \
                    else ModelLoraWeight(model_config, lora_rank, dtype, device)
                converted_weights = self.__convert_weight(raw_weights, lora_rank)
                lora_weight.copy_from_tensors(converted_weights)
                del converted_weights
                self.lora_weights[lora_id] = lora_weight
                logger.info(f'{lora_id} loaded!')
                
    def remove_lora_weights(self, lora_ids: List[str] = None):
        if (not lora_ids) or (lora_ids == '') or (lora_ids == 'all'):
            lora_ids = list(self.lora_weights.keys())
        for lora_id in lora_ids:
            if lora_id != 'empty' and lora_id in self.lora_weights:
                del self.lora_weights[lora_id]
                logger.info(f'{lora_id} removed!')
                
    def get_lora_batched_weights(self, lora_ids: List[str], lora_lens: List[int]) -> BatchedModelLoraWeight:
        return BatchedModelLoraWeight([self.lora_weights[lora_id] for lora_id in lora_ids], lora_lens)
                
    def __convert_weight(self, weights, rank):
        qA, qB, kA, kB, vA, vB, oA, oB = [], [], [], [], [], [], [], []
        gateA, gateB, upA, upB, downA, downB = [], [], [], [], [], []
        for key in weights.keys():
            if 'q_proj' in key:
                if 'A' in key:
                    qA.append(weights[key].unsqueeze(0))
                if 'B' in key:
                    qB.append(weights[key].unsqueeze(0))
            if 'k_proj' in key:
                if 'A' in key:
                    kA.append(weights[key].unsqueeze(0))
                if 'B' in key:
                    kB.append(weights[key].unsqueeze(0))
            if 'v_proj' in key:
                if 'A' in key:
                    vA.append(weights[key].unsqueeze(0))
                if 'B' in key:
                    vB.append(weights[key].unsqueeze(0))
            if 'o_proj' in key:
                if 'A' in key:
                    oA.append(weights[key].unsqueeze(0))
                if 'B' in key:
                    oB.append(weights[key].unsqueeze(0))
            if 'gate_proj' in key:
                if 'A' in key:
                    gateA.append(weights[key].unsqueeze(0))
                if 'B' in key:
                    gateB.append(weights[key].unsqueeze(0))
            if 'up_proj' in key:
                if 'A' in key:
                    upA.append(weights[key].unsqueeze(0))
                if 'B' in key:
                    upB.append(weights[key].unsqueeze(0))
            if 'down_proj' in key:
                if 'A' in key:
                    downA.append(weights[key].unsqueeze(0))
                if 'B' in key:
                    downB.append(weights[key].unsqueeze(0))
        weights = {
            'q.A': torch.cat(qA, dim=0) if qA else None,
            'q.B': torch.cat(qB, dim=0) if qB else None,
            'k.A': torch.cat(kA, dim=0) if kA else None,
            'k.B': torch.cat(kB, dim=0) if kB else None,
            'v.A': torch.cat(vA, dim=0) if vA else None,
            'v.B': torch.cat(vB, dim=0) if vB else None,
            'o.A': torch.cat(oA, dim=0) if oA else None,
            'o.B': torch.cat(oB, dim=0) if oB else None,
            'gate.A': torch.cat(gateA, dim=0) if gateA else None,
            'gate.B': torch.cat(gateB, dim=0) if gateB else None,
            'up.A': torch.cat(upA, dim=0) if upA else None,
            'up.B': torch.cat(upB, dim=0) if upB else None,
            'down.A': torch.cat(downA, dim=0) if downA else None,
            'down.B': torch.cat(downB, dim=0) if downB else None,
        }
        if rank == 8:
            for key in weights.keys():
                if weights[key] is not None:
                    if 'A' in key:
                        complement = torch.zeros_like(weights[key])
                        weights[key] = torch.cat([weights[key], complement], dim=1)
                    if 'B' in key:
                        complement = torch.zeros_like(weights[key])
                        weights[key] = torch.cat([weights[key], complement], dim=2)
        return weights