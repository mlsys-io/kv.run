# Modified from https://github.com/punica-ai/punica/blob/master/src/punica/models/llama_lora.py
# Editor: Junyi Shen

import math
import torch
from transformers.models.llama.modeling_llama import LlamaConfig
from text_generation_server.utils.punica_utils import BatchedKvCache, BatchedLoraWeight, BatchLenInfo, LoraWeight, KvPool, KvCache, convert_lora_weight
from .custom_modeling.punica_llama_lora import LlamaForCausalLM, LlamaLoraWeight, BatchedLlamaLoraWeight
import peft
from huggingface_hub import hf_hub_download

import os
import time
from opentelemetry import trace
from typing import Optional, Tuple, List, Type, Dict
from text_generation_server.models import Model
from text_generation_server.utils.tokens import batch_top_tokens
from text_generation_server.models.types import (
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.utils import NextTokenChooser, StoppingCriteria, Sampling
from dataclasses import dataclass
from transformers import AutoTokenizer

from loguru import logger
tracer = trace.get_tracer(__name__)

from .causal_lm import CausalLMBatch
from collections import defaultdict

def weight_convert(weights, rank):
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

@dataclass
class PunicaBatch(CausalLMBatch):
    lora_ids = [] #it goes wrong when lora_ids: List[str] = []

class PunicaLM(Model):
    def __init__(
        self,
        model_id: str = None,
        lora_ids: Dict = None,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False
    ):
        if use_medusa:
            raise RuntimeError("Medusa decoding is not enabled for AutoModel")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        self.device = device

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        model = LlamaForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
            #device_map="auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None,
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
        )
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and quantize != "bitsandbytes"
        ):
            model = model.cuda()

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.model_config = model.config
        self.kvpool = KvPool(
            num_layers=self.model_config.num_hidden_layers,
            num_heads=self.model_config.num_attention_heads,
            head_dim=self.model_config.hidden_size // self.model_config.num_attention_heads,
            page_len=16,
            dtype=dtype,
            device=device,
        )
        self.cache_pool = {}

        self.lora_weights = {}
        self.defalut_rank = 16
        self.lora_weights["empty"] = LlamaLoraWeight(
                self.model_config, self.defalut_rank, dtype, device
            )
        
        self.init_lora(
            lora_ids,
            self.model_config,
            device=device,
            )

        super(PunicaLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    def load_lora_adapters(self, lora_ids):
        self.init_lora(
            lora_ids,
            self.model_config,
            device=self.device
            )

    def remove_lora_adapters(self, lora_ids: list[str] = None):
        if (not lora_ids) or (lora_ids == '') or (lora_ids == 'all'):
            lora_ids = list(self.lora_weights)
        for lora_id in lora_ids:
            if lora_id != 'empty' and lora_id in self.lora_weights:
                del self.lora_weights[lora_id]
                logger.info(f'{lora_id} removed!')

    def get_lora_adapters(self):
        return list(self.lora_weights)

    def init_lora(
            self,
            lora_ids: Dict,
            model_config: LlamaConfig,
            device: torch.device,
            dtype=torch.float16,
            ):
        if lora_ids is None:
            return
        for lora_id in lora_ids:
            if lora_id not in self.lora_weights:
                model_path = hf_hub_download(lora_ids[lora_id], filename='adapter_model.bin')
                config_path = hf_hub_download(lora_ids[lora_id], filename='adapter_config.json')
                tmp = torch.load(model_path, map_location=device, weights_only=True)
                lora_rank = peft.config.PeftConfigMixin.from_json_file(config_path)['r']
                if lora_rank < 16:
                    lora_weight = LlamaLoraWeight(model_config, lora_rank*2, dtype, device)
                else:
                    lora_weight = LlamaLoraWeight(model_config, lora_rank, dtype, device)
                tmp = weight_convert(tmp,lora_rank)
                lora_weight.copy_from_tensors(tmp)
                del tmp
                self.lora_weights[lora_id] = lora_weight
                logger.info(f'{lora_id} loaded!')

    @property
    def batch_type(self) -> Type[PunicaBatch]:
        return PunicaBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @tracer.start_as_current_span("generate_token")
    @torch.no_grad()
    def generate_token(
        self, batch: PunicaBatch
    )-> Tuple[List[Generation], Optional[PunicaBatch], Tuple[int, int]]:
        start = time.time_ns()
        prefill_input_ids, prefill_lens, prefill_kv = [], [], []
        decode_input_ids, decode_kv = [], []
        lora_ids, lora_lens = [], []

        batch.lora_ids = [r.lora_id for r in batch.requests] #['empty' for _ in range(len(batch.requests))]
        for i,(request,ids,stopc,lora_id) in enumerate(zip(
            batch.requests,
            batch.input_ids,
            batch.stopping_criterias,
            batch.lora_ids,
            )):
            if stopc.current_tokens == 0:
                prefill_input_ids.extend(ids)
                prefill_lens.append(len(ids))
                kv_cache = KvCache(self.kvpool, len(ids))
                self.cache_pool[str(request.id)] = kv_cache
                prefill_kv.append(kv_cache)
            else:
                decode_input_ids.append(ids)
                kv_cache = self.cache_pool[str(request.id)]
                decode_kv.append(kv_cache)
                kv_cache.acquire_one()
            if lora_ids and lora_ids[-1] == lora_id:
                lora_lens[-1] += 1
            else:
                lora_ids.append(lora_id)
                lora_lens.append(1)

        input_ids = torch.tensor(
                prefill_input_ids + decode_input_ids,
                dtype=torch.long,
                device=self.device,
            )
        blen = BatchLenInfo(prefill_lens, len(decode_input_ids), self.device)
        prefill_kv = BatchedKvCache(prefill_kv) if prefill_kv else None
        decode_kv = BatchedKvCache(decode_kv) if decode_kv else None
        lora = BatchedLlamaLoraWeight(
            [self.lora_weights[id] for id in lora_ids], lora_lens
        )

        # Forward pass
        logits, _ = self.model(input_ids, blen, prefill_kv, decode_kv, lora)

        ptr = 0
        out = []
        for l in prefill_lens:
            out.append(logits[ptr:ptr+l].unsqueeze(0))
            ptr += l

        for l in range(len(decode_input_ids)):
            out.append(logits[ptr:ptr+1].unsqueeze(0))
            ptr += 1

        logits = torch.cat(out,dim=0)

        generations: List[Generation] = []
        stopped = True

        # Speculation is not active for causal
        accepted_ids = torch.ones_like(batch.input_ids)[:, 0]
        batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
            batch.top_n_tokens,
            batch.top_n_tokens_tensor,
            torch.log_softmax(logits[:, -1,:], -1),
            accepted_ids,
        )

        start_decode = time.time_ns()
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.top_n_tokens,
            batch_top_token_ids,
            batch_top_token_logprobs,
        )

        for i, (
            request,
            input_length,
            prefix_offset,
            read_offset,
            logits,
            next_token_chooser,
            stopping_criteria,
            all_input_ids,
            top_n_tokens,
            top_token_ids,
            top_token_logprobs,
        ) in enumerate(iterator):
            # Select next token
            next_token_id, logprobs = next_token_chooser(
                all_input_ids.view(1, -1), logits[-1:, :]
            )
            # Append next token to all tokens
            all_input_ids = torch.cat([all_input_ids, next_token_id])
            new_input_length = input_length + 1
            # Generated token
            next_token_logprob = logprobs[-1, next_token_id]
            next_token_id_squeezed = next_token_id.squeeze()
            next_token_text, prefix_offset, read_offset = self.decode_token(
                all_input_ids[:, 0], prefix_offset, read_offset
            )
            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token_id_squeezed,
                next_token_text,
            )
            if not stop:
                stopped = False
            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    output_text, _, _ = self.decode_token(
                        all_input_ids[:, 0],
                        prefix_offset=len(all_input_ids)
                        - stopping_criteria.current_tokens
                        - 1,
                        read_offset=len(all_input_ids)
                        - stopping_criteria.current_tokens,
                        skip_special_tokens=True,
                    )
                    # Get seed
                    if isinstance(next_token_chooser.choice, Sampling):
                        seed = next_token_chooser.choice.seed
                    else:
                        seed = None

                    generated_text = GeneratedText(
                        output_text, stopping_criteria.current_tokens, reason, seed
                    )

                    # release kv-cache
                    self.cache_pool[str(request.id)].release()
                    del self.cache_pool[str(request.id)]

                else:
                    generated_text = None

                # Prefill
                if stopping_criteria.current_tokens == 1 and request.prefill_logprobs:
                    # Remove generated token to only have prefill and add nan for first prompt token
                    prefill_logprobs = [float("nan")] + torch.log_softmax(
                        logits, -1
                    ).gather(1, all_input_ids[1:]).squeeze(1)[
                        -new_input_length:-1
                    ].tolist()
                    prefill_token_ids = all_input_ids[-new_input_length:-1]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    prefill_tokens = Tokens(
                        prefill_token_ids,
                        prefill_logprobs,
                        prefill_texts,
                        is_special=[],
                    )
                else:
                    prefill_tokens = None

                if top_n_tokens > 0:
                    all_top_tokens = []
                    for top_token_ids, top_token_logprobs in zip(
                        top_token_ids, top_token_logprobs
                    ):
                        toptoken_texts = self.tokenizer.batch_decode(
                            top_token_ids,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=False,
                        )
                        special_toptokens = [
                            token_id in self.all_special_ids
                            for token_id in top_token_ids
                        ]
                        top_tokens = Tokens(
                            top_token_ids,
                            top_token_logprobs,
                            toptoken_texts,
                            special_toptokens,
                        )
                        all_top_tokens.append(top_tokens)
                    top_tokens = all_top_tokens
                else:
                    top_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    Tokens(
                        [next_token_id_squeezed],
                        [next_token_logprob],
                        [next_token_text],
                        [next_token_id_squeezed.item() in self.all_special_ids],
                    ),
                    generated_text,
                    top_tokens,
                )

                generations.append(generation)

            # Update values
            batch.next_token_choosers[i] = batch.next_token_choosers[i].advance_grammar(
                next_token_id_squeezed.item()
            )
            batch.input_ids[i, 0] = next_token_id
            batch.all_input_ids[i] = all_input_ids
            batch.input_lengths[i] = new_input_length
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.max_input_length = max(batch.max_input_length, new_input_length)

        # We finished all generations in the batch; there is no next batch
        if stopped:
            forward_ns = start_decode - start
            decode_ns = time.time_ns() - start_decode
            return generations, None, (forward_ns, decode_ns)

        # Slice unused values from prefill
        batch.input_ids = batch.input_ids[:, :1]

        # Update attention_mask as we added a new token to input_ids
        batch.attention_mask[:, -batch.padding_right_offset] = 1
        # Decrease right offset
        batch.padding_right_offset -= 1

        # Update position_ids
        batch.position_ids = batch.position_ids[:, -1:] + 1

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        return generations, batch, (forward_ns, decode_ns)