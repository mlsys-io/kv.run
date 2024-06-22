# Modified from https://github.com/punica-ai/punica/blob/master/src/punica/models/llama_lora.py
# Editor: Junyi Shen

import torch
from text_generation_server.models_flashinfer.custom_modeling.embedding_llama import (
    LlamaForCausalLM,
)

import time
import json
from opentelemetry import trace
from typing import Optional, Tuple, List, Type, Dict
from text_generation_server.models import Model
from text_generation_server.utils.tokens import batch_top_tokens
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.utils import Sampling
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoConfig, LlavaConfig
from huggingface_hub import hf_hub_download

from loguru import logger
from PIL import Image
from io import BytesIO
import base64

tracer = trace.get_tracer(__name__)

from text_generation_server.models.causal_lm import CausalLMBatch


@dataclass
class LlavaBatch(CausalLMBatch):
    imgs = []


class LlavaLM(Model):
    def __init__(
        self,
        model_id: str = None,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        if use_medusa:
            raise RuntimeError("Medusa decoding is not enabled for ThisModel")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        self.device = device
        self.model_id = model_id
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
            # device_map="auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None,
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
        )
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and quantize != "bitsandbytes"
        ):
            model = model.cuda()
        self.model = model

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.model_config = AutoConfig.from_pretrained(model_id)
        self.kvpool = KvPool(
            num_layers=self.model_config.num_hidden_layers,
            num_heads=self.model_config.num_attention_heads,
            head_dim=self.model_config.hidden_size
            // self.model_config.num_attention_heads,
            page_len=16,
            dtype=dtype,
            device=device,
        )
        self.cache_pool = {}

        with open(hf_hub_download(self.model_id, filename="config.json")) as f:
            mm_config = json.loads(f.read())

        self.vision_model = self.build_vision_model(mm_config)
        self.vision_model.to(self.device).eval()
        self.projector = self.build_projector(mm_config)
        self.projector.to(self.device).eval()
        self.id_embedder = self.model.model.embed_tokens
        self.additional_init_length = 576  # 512 + 64 I guess

        super(LlavaLM, self).__init__(
            model=self.model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )
        logger.info(f"Initialized LlavaLM with model_id: {model_id}")

    def build_vision_model(self, model_config, **kwargs):
        from .llava_models.encoder.encoder import CLIPVisionTower

        mm_vision_tower = "openai/clip-vit-large-patch14-336"
        return CLIPVisionTower(mm_vision_tower, args=model_config, **kwargs)

    def build_projector(self, model_config, **kwargs):
        from .llava_models.projector.builder import build_vision_projector

        projector = build_vision_projector(model_config, **kwargs)
        model_path = hf_hub_download(self.model_id, filename="mm_projector.bin")
        state_dict = torch.load(model_path)
        new_state_dict = {
            "0.weight": state_dict["model.mm_projector.0.weight"],
            "0.bias": state_dict["model.mm_projector.0.bias"],
            "2.weight": state_dict["model.mm_projector.2.weight"],
            "2.bias": state_dict["model.mm_projector.2.bias"],
        }
        projector.load_state_dict(new_state_dict)
        return projector

    @property
    def batch_type(self) -> Type[LlavaBatch]:
        return LlavaBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @torch.no_grad()
    def prefill_token(self, batch: LlavaBatch):
        img_features = []
        for r in batch.requests:
            img = Image.open(r.inputb).convert("RGB")
            img = self.vision_model.image_processor(img, return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)
            img_features.append(self.vision_model(img))
        img_features = torch.stack(img_features, dim=0)
        if self.projector:
            img_features = self.projector(img_features)

        input_ids = torch.tensor(batch.input_ids, dtype=torch.long, device=self.device)
        input_embeddings = self.id_embedder(input_ids).unsqueeze(0)
        input_embeddings = torch.cat([img_features, input_embeddings], dim=1)
        lens = batch.input_lengths + self.additional_init_length
        blen = BatchLenInfo(lens, 0, self.device)

        for r, l in zip(batch.requests, lens):
            kv_cache = KvCache(self.kvpool, l)
            self.cache_pool[str(r.id)] = kv_cache

        prefill_kv = BatchedKvCache(
            [self.cache_pool[str(r.id)] for r in batch.requests]
        )

        logits, _ = self.model(
            input_ids=None,
            blen=blen,
            prefill_kv=prefill_kv,
            decode_kv=None,
            input_embeddings=input_embeddings,
        )

        logits = logits[blen.indptr[1:] - 1]
        logits = logits.unsqueeze(1)
        return logits

    @torch.no_grad()
    def generate_token(self, batch: LlavaBatch):
        input_ids, decode_kv = [], []

        for i, (request, ids) in enumerate(zip(batch.requests, batch.input_ids)):
            input_ids.append(ids)
            kv_cache = self.cache_pool[str(request.id)]
            decode_kv.append(kv_cache)
            kv_cache.acquire_one()

        blen = BatchLenInfo([], len(input_ids), self.device)
        decode_kv = BatchedKvCache(decode_kv) if decode_kv else None

        # Forward pass
        logits, _ = self.model(input_ids, blen, None, decode_kv, None)
        logits = logits.unsqueeze(1)
        return logits

    def generate(
        self, batch: LlavaBatch
    ) -> Tuple[List[Generation], Optional[LlavaBatch], Tuple[int, int]]:
        start = time.time_ns()
        logits = (
            self.prefill_token(batch)
            if batch.stopping_criterias[0].current_tokens == 0
            else self.generate_token(batch)
        )
        generations: List[Generation] = []
        stopped = True

        # Speculation is not active for causal
        accepted_ids = torch.ones_like(batch.input_ids)[:, 0]
        batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
            batch.top_n_tokens,
            batch.top_n_tokens_tensor,
            torch.log_softmax(logits[:, -1, :], -1),
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


if __name__ == "__main__":
    model = LlavaLM(model_id="liuhaotian/llava-v1.5-7b")
    print(model)
