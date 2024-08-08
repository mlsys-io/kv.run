from collections.abc import Sequence
from typing import Dict, List

import peft
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from punica_kernels import (
    add_lora_sgmv_custom_cutlass as add_lora,
)


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

    def to_gpu(self):
        # todo: multi-gpu
        self.wa = self.wa.cuda()
        self.wb = self.wb.cuda()

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

    def to_gpu(self):
        self.q.to_gpu()
        self.k.to_gpu()
        self.v.to_gpu()
        self.o.to_gpu()
        self.gate.to_gpu()
        self.up.to_gpu()
        self.down.to_gpu()

    def copy_from_tensors(self, ts: dict[str, torch.Tensor]):
        if ts.get("q.A") is not None:
            self.q.copy_from_tensor(ts["q.A"], ts["q.B"])
        if ts.get("k.A") is not None:
            self.k.copy_from_tensor(ts["k.A"], ts["k.B"])
        if ts.get("v.A") is not None:
            self.v.copy_from_tensor(ts["v.A"], ts["v.B"])
        if ts.get("o.A") is not None:
            self.o.copy_from_tensor(ts["o.A"], ts["o.B"])
        if ts.get("gate.A") is not None:
            self.gate.copy_from_tensor(ts["gate.A"], ts["gate.B"])
        if ts.get("up.A") is not None:
            self.up.copy_from_tensor(ts["up.A"], ts["up.B"])
        if ts.get("down.A") is not None:
            self.down.copy_from_tensor(ts["down.A"], ts["down.B"])

class ModelLoraWeightPhi:
    def __init__(
        self,
        modelConfig: ModelConfigForLora,
        lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        # num_kv_group = modelConfig.num_qo_heads // modelConfig.num_kv_heads
        self.out_proj = LoraWeight(
            modelConfig.num_hidden_layers,
            modelConfig.hidden_size,
            modelConfig.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        # self.Wqkv = LoraWeight(
        #     modelConfig.num_hidden_layers,
        #     modelConfig.hidden_size,
        #     modelConfig.hidden_size * 3,    # check why it's hidden_size * 3
        #     lora_rank,
        #     dtype,
        #     device,
        # )
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
            modelConfig.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.v = LoraWeight(
            modelConfig.num_hidden_layers,
            modelConfig.hidden_size,
            modelConfig.hidden_size,
            lora_rank,
            dtype,
            device,
        )

    def to_gpu(self):
        self.out_proj.to_gpu()
        # self.Wqkv.to_gpu()
        self.q.to_gpu()
        self.k.to_gpu()
        self.v.to_gpu()

    def copy_from_tensors(self, ts: dict[str, torch.Tensor]):
        self.out_proj.copy_from_tensor(ts["out_proj.A"], ts["out_proj.B"])
        # self.Wqkv.copy_from_tensor(ts["Wqkv.A"], ts["Wqkv.B"])
        self.q.copy_from_tensor(ts["Wqkv.A"], ts["Wqkv.B"][:, :2560])
        self.k.copy_from_tensor(ts["Wqkv.A"], ts["Wqkv.B"][:, 2560:5120])
        self.v.copy_from_tensor(ts["Wqkv.A"], ts["Wqkv.B"][:, 5120:])


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

    def apply_lora_weight_attn(
        self, attn_projected: torch.Tensor, attn_raw: torch.Tensor, layer_idx: int
    ):
        add_lora(
            attn_projected,
            attn_raw,
            self.o.wa_ptr,
            self.o.wb_ptr,
            self.segment,
            layer_idx,
            self.rank,
        )

    def apply_lora_weight_kvq(
        self,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ):
        add_lora(
            q_proj,
            hidden_states,
            self.q.wa_ptr,
            self.q.wb_ptr,
            self.segment,
            layer_idx,
            self.rank,
        )
        add_lora(
            k_proj,
            hidden_states,
            self.k.wa_ptr,
            self.k.wb_ptr,
            self.segment,
            layer_idx,
            self.rank,
        )
        add_lora(
            v_proj,
            hidden_states,
            self.v.wa_ptr,
            self.v.wb_ptr,
            self.segment,
            layer_idx,
            self.rank,
        )

    def apply_lora_weight_gate(
        self, gate: torch.Tensor, x: torch.Tensor, layer_idx: int
    ):
        add_lora(
            gate,
            x,
            self.gate.wa_ptr,
            self.gate.wb_ptr,
            self.segment,
            layer_idx,
            self.rank,
        )

    def apply_lora_weight_up(self, up: torch.Tensor, x: torch.Tensor, layer_idx: int):
        add_lora(
            up,
            x,
            self.up.wa_ptr,
            self.up.wb_ptr,
            self.segment,
            layer_idx,
            self.rank,
        )

    def apply_lora_weight_down(
        self, down: torch.Tensor, x: torch.Tensor, layer_idx: int
    ):
        add_lora(
            down,
            x,
            self.down.wa_ptr,
            self.down.wb_ptr,
            self.segment,
            layer_idx,
            self.rank,
        )

class BatchedModelLoraWeightPhi:
    def __init__(self, weights: List[ModelLoraWeightPhi], lens: List[int]):
        assert len(weights) == len(lens)
        device = weights[0].out_proj.wa.device
        self.out_proj = BatchedLoraWeight([w.out_proj for w in weights])
        # self.Wqkv = BatchedLoraWeight([w.Wqkv for w in weights])
        self.q = BatchedLoraWeight([w.q for w in weights])
        self.k = BatchedLoraWeight([w.k for w in weights])
        self.v = BatchedLoraWeight([w.v for w in weights])
        self.segment = torch.cumsum(
            torch.tensor([0] + lens, dtype=torch.int32, device=device),
            dim=0,
            dtype=torch.int32,
        )
        self.rank = weights[0].out_proj.lora_rank

    # change this
    # def apply_lora_weight_out_proj(
    #     self, out_proj: torch.Tensor, hidden_states: torch.Tensor, layer_idx: int
    # ):
    #     add_lora(
    #         out_proj,
    #         hidden_states,
    #         self.out_proj.wa_ptr,
    #         self.out_proj.wb_ptr,
    #         self.segment,
    #         layer_idx,
    #         self.rank,
    #     )

    def apply_lora_weight_attn(
        self, attn_projected: torch.Tensor, attn_raw: torch.Tensor, layer_idx: int
    ):
        add_lora(
            attn_projected,
            attn_raw,
            self.out_proj.wa_ptr,
            self.out_proj.wb_ptr,
            self.segment,
            layer_idx,
            self.rank,
        )

    def apply_lora_weight_Wkvq(
        self,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ):
        add_lora(
            q_proj,
            hidden_states,
            self.q.wa_ptr,
            self.q.wb_ptr,
            self.segment,
            layer_idx,
            self.rank,
        )
        add_lora(
            k_proj,
            hidden_states,
            self.k.wa_ptr,
            self.k.wb_ptr,
            self.segment,
            layer_idx,
            self.rank,
        )
        add_lora(
            v_proj,
            hidden_states,
            self.v.wa_ptr,
            self.v.wb_ptr,
            self.segment,
            layer_idx,
            self.rank,
        )

def load_lora_weights(lora_id):
    try:
        model_path = hf_hub_download(lora_id, filename="adapter_model.bin")
    except:
        from safetensors.torch import load_file

        model_path = hf_hub_download(lora_id, filename="adapter_model.safetensors")
        tmp = load_file(model_path, device="cpu")
        model_path = model_path.replace(".safetensors", ".bin")
        torch.save(tmp, model_path)
    config_path = hf_hub_download(lora_id, filename="adapter_config.json")
    return model_path, config_path

def load_lora_weights_local(lora_id):
    # load lora weights from local
    try:
        model_path = lora_id + '/adapter_model.bin'
        # check if this file exists
        with open(model_path, 'r') as f:
            pass
    except:
        from safetensors.torch import load_file
        model_path = lora_id + '/adapter_model.safetensors'
        tmp = load_file(model_path, device="cpu")
        model_path = model_path.replace('.safetensors', '.bin')
        torch.save(tmp, model_path)
    config_path = lora_id + '/adapter_config.json'
    print(f"model_path: {model_path}, config_path: {config_path}")
    return model_path, config_path


class ModelLoraManager:
    def __init__(self, model_config: ModelConfigForLora, dtype, lora_cap=32, model_type='llama'):
        self.lora_weights_gpu: Dict[str, ModelLoraWeight] = {}
        self.lora_cap = lora_cap + 1  # one for empty
        self.defalut_rank = 16
        self.lora_weights_cpu = {}
        self.model_type = model_type
        if model_type == 'phi':
            self.lora_weights_cpu["empty"] = ModelLoraWeightPhi(
            model_config, self.defalut_rank, dtype, "cpu"
        )
        else:
            self.lora_weights_cpu["empty"] = ModelLoraWeight(
                model_config, self.defalut_rank, dtype, "cpu"
            )
        

    def set_lora_weights(
        self,
        lora_ids: List[str],
        model_config: ModelConfigForLora,
        dtype=torch.float16,
    ):
        for lora_id in lora_ids:
            if lora_id not in self.lora_weights_cpu:
                try:
                    model_path, config_path = load_lora_weights(lora_id)
                except:
                    model_path, config_path = load_lora_weights_local(lora_id)
                raw_weights = torch.load(
                    model_path, map_location="cpu", weights_only=True
                )
                lora_rank = peft.config.PeftConfigMixin.from_json_file(config_path)["r"]
                if self.model_type == 'phi':
                    lora_weight = ModelLoraWeightPhi(model_config, lora_rank, dtype, "cpu")
                    converted_weights = self.__convert_weight_phi(raw_weights, lora_rank)   # lora_rank = 32 for the example model                   
                else: 
                    lora_weight = (
                            ModelLoraWeight(model_config, lora_rank * 2, dtype, "cpu")
                            if lora_rank < 16
                            else ModelLoraWeight(model_config, lora_rank, dtype, "cpu")
                        )
                    converted_weights = self.__convert_weight(raw_weights, lora_rank)
                lora_weight.copy_from_tensors(converted_weights)
                del converted_weights
                self.lora_weights_cpu[lora_id] = lora_weight
                logger.info(f"{lora_id} loaded in cpu memory!")

    def remove_lora_weights(self, lora_ids: List[str] = None):
        if (not lora_ids) or (lora_ids == "") or (lora_ids == "all"):
            lora_ids = list(self.lora_weights_gpu.keys())
        for lora_id in lora_ids:
            if lora_id != "empty" and lora_id in self.lora_weights_gpu:
                del self.lora_weights_gpu[lora_id]
                logger.info(f"{lora_id} removed from gpu memory!")

    def get_lora_batched_weights(
        self, lora_ids: List[str], lora_lens: List[int]
    ) -> BatchedModelLoraWeight:
        assert len(lora_ids) <= self.lora_cap
        # for lora_id in lora_ids:
        #    assert lora_id in self.lora_weights_cpu
        loraweights = []
        for lora_id in lora_ids:
            if lora_id and lora_id not in self.lora_weights_gpu:
                self.lora_weights_gpu[lora_id] = self.lora_weights_cpu[lora_id]
                self.lora_weights_gpu[lora_id].to_gpu()
        while len(self.lora_weights_gpu) > self.lora_cap:
            # eviction policy : kick out the first adapter that is not in the current batch
            # todo: use LRU to evict
            candidate = list(
                set(list(self.lora_weights_gpu)) - set(lora_ids) - set(["empty"])
            )
            self.remove_lora_weights([candidate[0]])
        for lora_id in lora_ids:
            loraweights.append(self.lora_weights_gpu[lora_id])

        if self.model_type == 'phi':
            return BatchedModelLoraWeightPhi(loraweights, lora_lens)
        else:
            return BatchedModelLoraWeight(loraweights, lora_lens)

    def __convert_weight(self, weights, rank):
        qA, qB, kA, kB, vA, vB, oA, oB = [], [], [], [], [], [], [], []
        gateA, gateB, upA, upB, downA, downB = [], [], [], [], [], []
        for key in weights.keys():
            if "q_proj" in key:
                if "A" in key:
                    qA.append(weights[key].unsqueeze(0))
                if "B" in key:
                    qB.append(weights[key].unsqueeze(0))
            if "k_proj" in key:
                if "A" in key:
                    kA.append(weights[key].unsqueeze(0))
                if "B" in key:
                    kB.append(weights[key].unsqueeze(0))
            if "v_proj" in key:
                if "A" in key:
                    vA.append(weights[key].unsqueeze(0))
                if "B" in key:
                    vB.append(weights[key].unsqueeze(0))
            if "o_proj" in key:
                if "A" in key:
                    oA.append(weights[key].unsqueeze(0))
                if "B" in key:
                    oB.append(weights[key].unsqueeze(0))
            if "gate_proj" in key:
                if "A" in key:
                    gateA.append(weights[key].unsqueeze(0))
                if "B" in key:
                    gateB.append(weights[key].unsqueeze(0))
            if "up_proj" in key:
                if "A" in key:
                    upA.append(weights[key].unsqueeze(0))
                if "B" in key:
                    upB.append(weights[key].unsqueeze(0))
            if "down_proj" in key:
                if "A" in key:
                    downA.append(weights[key].unsqueeze(0))
                if "B" in key:
                    downB.append(weights[key].unsqueeze(0))
        weights = {
            "q.A": torch.cat(qA, dim=0) if qA else None,
            "q.B": torch.cat(qB, dim=0) if qB else None,
            "k.A": torch.cat(kA, dim=0) if kA else None,
            "k.B": torch.cat(kB, dim=0) if kB else None,
            "v.A": torch.cat(vA, dim=0) if vA else None,
            "v.B": torch.cat(vB, dim=0) if vB else None,
            "o.A": torch.cat(oA, dim=0) if oA else None,
            "o.B": torch.cat(oB, dim=0) if oB else None,
            "gate.A": torch.cat(gateA, dim=0) if gateA else None,
            "gate.B": torch.cat(gateB, dim=0) if gateB else None,
            "up.A": torch.cat(upA, dim=0) if upA else None,
            "up.B": torch.cat(upB, dim=0) if upB else None,
            "down.A": torch.cat(downA, dim=0) if downA else None,
            "down.B": torch.cat(downB, dim=0) if downB else None,
        }
        if rank == 8:
            for key in weights.keys():
                if weights[key] is not None:
                    if "A" in key:
                        complement = torch.zeros_like(weights[key])
                        weights[key] = torch.cat([weights[key], complement], dim=1)
                    if "B" in key:
                        complement = torch.zeros_like(weights[key])
                        weights[key] = torch.cat([weights[key], complement], dim=2)
        return weights

    def __convert_weight_phi(self, weights, rank):
        out_projA, out_projB = [], []
        WqkvA, WqkvB = [], []
        for key in weights.keys():
            if "out_proj" in key:
                if "A" in key:
                    out_projA.append(weights[key].unsqueeze(0))
                if "B" in key:
                    out_projB.append(weights[key].unsqueeze(0))
            if "Wqkv" in key:
                if "A" in key:
                    WqkvA.append(weights[key].unsqueeze(0))
                if "B" in key:
                    WqkvB.append(weights[key].unsqueeze(0))
        weights = {
            "out_proj.A": torch.cat(out_projA, dim=0) if out_projA else None,
            "out_proj.B": torch.cat(out_projB, dim=0) if out_projB else None,
            "Wqkv.A": torch.cat(WqkvA, dim=0) if WqkvA else None,
            "Wqkv.B": torch.cat(WqkvB, dim=0) if WqkvB else None,
        }
        # if rank == 8:
        #     for key in weights.keys():
        #         if weights[key] is not None:
        #             if "A" in key:
        #                 complement = torch.zeros_like(weights[key])
        #                 weights[key] = torch.cat([weights[key], complement], dim=1)
        #             if "B" in key:
        #                 complement = torch.zeros_like(weights[key])
        #                 weights[key] = torch.cat([weights[key], complement], dim=2)
        return weights