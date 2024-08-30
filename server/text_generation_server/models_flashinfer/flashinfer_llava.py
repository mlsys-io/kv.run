# Modified from transformers/models/llava_next/modeling_llava_next.py
# Editor: Junyi Shen

import torch
import math
import time
from opentelemetry import trace
from typing import Optional, Tuple, List, Type, Iterable
from text_generation_server.models import Model
from text_generation_server.models.types import (
    Tokens,
    Generation,
    GeneratedText,
)
from dataclasses import dataclass
from transformers import AutoConfig, AutoProcessor
from loguru import logger
from text_generation_server.pb import generate_pb2
tracer = trace.get_tracer(__name__)

from text_generation_server.models_flashinfer.flashinfer_causal_lm import (
    FlashinferBatch,
    RequestContext, 
    RequestKvCache,
    getKvCacheBatchPosition,
    KvCacheBatchPosition,
    find_padded_head_dim,
    MEMORY_FRACTION,
    PAGE_LEN,
    KvCachePool,
)
from text_generation_server.models_flashinfer.flashinfer_llama import (
    FlashinferLlama,
    weight_files,
    Weights,
    initialize_torch_distributed,
)
from text_generation_server.models.vlm_causal_lm import (
    image_text_replacement,
    load_data_uri,
    split,
)
from text_generation_server.models.custom_modeling.llava_next import (
    get_anyres_image_grid_shape,
    unpad_image,
    load_vision_model,
    LlavaNextMultiModalProjector,
)

@dataclass
class LlavaBatch(FlashinferBatch):
    pixel_values: Optional[List[torch.Tensor]]
    pixel_attention_mask: Optional[List[torch.Tensor]]
    image_sizes: Optional[List[Tuple[int, int]]]

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches):
        batch = super(LlavaBatch, cls).concatenate(batches)
        batch.pixel_values = None
        batch.pixel_attention_mask = None
        batch.image_sizes = None
        return batch

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]):
        batch = super().filter(request_ids)
        batch.pixel_values = None
        batch.pixel_attention_mask = None
        batch.image_sizes = None
        return batch
        
class LlavaLM(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False
    ):  
        # Initialize LlavaLM
        self.config = AutoConfig.from_pretrained(model_id)
        self.config.quantize = quantize
        self.config.vision_config.quantize = quantize
        self.process_group, rank, world_size = initialize_torch_distributed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, self.device, dtype, process_group=self.process_group)
        
        self.vision_tower = load_vision_model(
            prefix="vision_tower",
            config=self.config.vision_config,
            weights=weights,
        )

        self.multi_modal_projector = LlavaNextMultiModalProjector(
            prefix="multi_modal_projector", config=self.config, weights=weights
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        self.vocab_size = self.config.text_config.vocab_size
        
        self.language_model = FlashinferLlama(
            model_id= model_id,
            lora_ids= None, 
            revision= revision,
            quantize= quantize,
            dtype= dtype,
            trust_remote_code= trust_remote_code, 
        )
        
        self.image_newline = self.image_newline = weights.get_tensor("image_newline")
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.id_embedder = self.language_model.model.embed_tokens
        self.vision_feature_select_strategy = self.config.vision_feature_select_strategy
        # Initialize KvCachePool
        if not self.language_model.kvCachePool:
            head_dim_padded = find_padded_head_dim(
                self.language_model.model_config.hidden_size // self.language_model.model_config.num_attention_heads
            )
            dtype_size = torch.tensor([], dtype=self.language_model.dtype).element_size()
            cache_page_size = (
                2
                * PAGE_LEN
                * self.language_model.model_config.num_hidden_layers
                * self.language_model.model_config.num_attention_heads
                * head_dim_padded
                * dtype_size
            )

            currentDevice = torch.cuda.current_device()
            total_free_memory, _ = torch.cuda.mem_get_info(currentDevice)
            total_gpu_memory = torch.cuda.get_device_properties(
                currentDevice
            ).total_memory
            free_memory = max(
                0, total_free_memory - (1 - MEMORY_FRACTION) * total_gpu_memory
            )
            num_pages_to_allocate = int(free_memory * 0.80 / cache_page_size)
            print(
                f"Cache allocation:\n"
                f"  Cache Page Size: {cache_page_size / 1024 / 1024} MB\n"
                f"  Dtype Size: {dtype_size}\n"
                f"  Free Memory: {free_memory / 1024 / 1024 / 1024} GB\n"
                f"  Total GPU Memory: {total_gpu_memory / 1024 / 1024 / 1024} GB\n"
                f"  Number of Pages to Allocate: {num_pages_to_allocate}"
            )

            self.language_model.kvCachePool = KvCachePool(
                max_pages=num_pages_to_allocate,
                num_layers=self.language_model.model_config.num_hidden_layers,
                num_heads=self.language_model.model_config.num_key_value_heads,
                head_dim=head_dim_padded,
                page_len=PAGE_LEN,
                dtype=self.language_model.dtype,
                device=self.language_model.device,
            )

        num_free_pages = self.language_model.kvCachePool.num_free_pages()
        logger.info(f"Initialized LlavaLM with model_id: {model_id}")
        
    @property
    def batch_type(self) -> Type[LlavaBatch]:
        return LlavaBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.language_model.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
    @tracer.start_as_current_span("generate_token")
    @torch.no_grad()
    def generate_token(
        self, 
        batch: LlavaBatch,
        embeddings: torch.Tensor = None,
    ) -> Tuple[List[Generation], Optional[LlavaBatch], Tuple[int, int]]:
        start = time.time_ns()
        input_ids, lora_ids, lora_lens = [], [], []
        request_kv_caches = []
        all_input_ids_stacked: List[List[int]] = []
        for request_context in batch.request_contexts:
            if not request_context.is_stopped:
                all_input_ids_stacked.append(request_context.output_ids)
                if batch.is_prefill:
                    input_ids.extend(request_context.output_ids)
                else:
                    input_ids.append(request_context.output_ids[-1])
                request_kv_caches.append(request_context.request_kv_cache)
                if not batch.is_prefill:
                    request_context.request_kv_cache.increment()

                if lora_ids and lora_ids[-1] == request_context.lora_id:
                    lora_lens[-1] += 1
                elif request_context.lora_id:
                    lora_ids.append(request_context.lora_id)
                    lora_lens.append(1)

        all_input_ids_tensor = self.language_model._get_all_input_ids_tensor(
            all_input_ids_stacked, batch.request_contexts
        )
        input_ids_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.language_model.device,
        )

        batch_position: KvCacheBatchPosition = getKvCacheBatchPosition(
            request_kv_caches, isPrefill=batch.is_prefill, device=self.language_model.device
        )

        loraWeights = (
            self.language_model.loraManager.get_lora_batched_weights(lora_ids, lora_lens)
            if lora_ids
            else None
        )
        
        raw_logits, _ = self.language_model.model(
            input_ids = input_ids_tensor if embeddings is None else None,
            kvCachePool = self.language_model.kvCachePool,
            is_prefill = batch.is_prefill,
            batch_position = batch_position,
            loraWeight = loraWeights,
            input_embeddings = embeddings,
        )

        start_decode = time.time_ns()
        logits = (
            raw_logits[batch_position.seq_indptr[1:] - 1]
            if batch.is_prefill
            else raw_logits
        )

        all_stop = True
        generations: List[Generation] = []
        num_stopped_requests = 0
        start_next_token_id = time.time_ns()

        next_token_ids, next_token_logprobs, logprobs, _, _ = (
            self.language_model._get_next_batch_token_id_heterogeneous(
                batch.request_contexts, all_input_ids_tensor, logits
            )
        )
        next_token_id_ns = time.time_ns() - start_next_token_id

        for i, request_context in enumerate(batch.request_contexts):
            if request_context.is_stopped:
                num_stopped_requests += 1
                continue
            next_token_id = next_token_ids[i - num_stopped_requests]
            request_context.append_token(next_token_id)
            text = self.processor.decode(
                next_token_id,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )

            stop_reason = request_context.get_stop_reason()
            if stop_reason != None:
                output_text = self.processor.decode(
                    request_context.output_ids[request_context.prompt_len :],
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=True,
                )
                generated_text = GeneratedText(
                    output_text,
                    len(request_context.output_ids) - request_context.prompt_len + 1,
                    stop_reason,
                    None,
                )
                request_context.is_stopped = True
                request_context.request_kv_cache.release()
            else:
                generated_text = None
                all_stop = False

            request_context.prefill_tokens = None

            generation = Generation(
                request_context.request_id,
                request_context.prefill_tokens,
                Tokens(
                    [next_token_id],
                    [0],  # prob
                    [text],
                    [next_token_id in self.language_model.all_special_ids],
                ),
                generated_text,
                # top_tokens
                None,
            )
            generations.append(generation)

        forward_ns = start_decode - start
        decode_ns = next_token_id_ns
        # The router stops generation only when batch=None
        if all_stop:
            return generations, None, (forward_ns, decode_ns)
        else:
            return generations, batch, (forward_ns, decode_ns)
        
    def decode_batch(
        self, cachedBatchesPb: Iterable[generate_pb2.CachedBatch]
    ) -> Tuple[List[Generation], Optional[LlavaBatch], Tuple[int, int], int]:
        start_concat = time.time_ns()
        batch = self.language_model._convertCachedBatch(cachedBatchesPb)
        concat_ns = time.time_ns() - start_concat
        generations, next_batch, timings = self.generate_token(batch)
        if next_batch:
            self.language_model.batch_cache.set(next_batch)
        return generations, next_batch, timings, concat_ns

    def prefill_batch(
        self, 
        pb_batch: generate_pb2.Batch,
    ) -> Tuple[List[Generation], Optional[LlavaBatch], Tuple[int, int]]:
        batch_tokenized_inputs, image_inputs = self.batch_tokenized_inputs(
            pb_batch.requests, 
            self.language_model.tokenizer, 
            self.processor, 
            self.config
        )

        input_ids = [torch.tensor(_tokenized, device=self.language_model.device) for _tokenized in batch_tokenized_inputs]
        
        if image_inputs is not None:
            pixel_values = image_inputs["pixel_values"].to(device=self.language_model.device)
            if "pixel_attention_mask" in image_inputs:
                pixel_attention_mask = image_inputs["pixel_attention_mask"].to(
                    device=self.vision_tower.device
                )
            else:
                pixel_attention_mask = None
            if "image_sizes" in image_inputs:
                image_sizes = image_inputs["image_sizes"].to(device=self.language_model.device)
            else:
                image_sizes = None
        else:
            pixel_values = None
            pixel_attention_mask = None
            image_sizes = None
            
        # Embed the input
        inputs_embeds = [self.id_embedder(_input_ids) for _input_ids in input_ids]
        
        if pixel_values is not None and len(pixel_values) > 0:
            num_images, num_patches, channels, height, width = pixel_values.shape
            print(f"num_images: {num_images}, num_patches: {num_patches}, channels: {channels}, height: {height}, width: {width}")
            pixel_values = pixel_values.view(
                num_images * num_patches, channels, height, width
            )
        
            image_outputs = self.vision_tower(pixel_values)
            
            selected_image_feature = image_outputs.last_hidden_state

            if self.vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif self.vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

            image_features = self.multi_modal_projector(selected_image_feature)
            split_sizes = [num_patches] * num_images
            image_features = torch.split(image_features, split_sizes, dim=0)
            
            height = width = (
                self.config.vision_config.image_size
                // self.config.vision_config.patch_size
            )
            
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]

                    if height * width != base_image_feature.shape[0]:
                        raise ValueError(
                            "The number of patches is not consistent with the image size."
                        )
                    num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                        image_sizes[image_idx],
                        self.config.image_grid_pinpoints,
                        self.config.vision_config.image_size,
                    )
                    image_feature = image_feature.view(
                        num_patch_height, num_patch_width, height, width, -1
                    )
                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                    image_feature = unpad_image(image_feature, image_sizes[image_idx])
                    image_feature = torch.cat(
                        (
                            image_feature,
                            self.image_newline[:, None, None].expand(
                                *image_feature.shape[:-1], 1
                            ),
                        ),
                        dim=-1,
                    )
                    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    image_feature = torch.cat(
                        (base_image_feature, image_feature), dim=0
                    )
                else:
                    image_feature = image_feature[0]
                    image_feature = torch.cat(
                        (image_feature, self.image_newline[None]), dim=0
                    )
                new_image_features.append(image_feature)

            model_input = []
            for _input_ids, _input_embeds, _image_features in zip(input_ids, inputs_embeds, new_image_features):
                mask = _input_ids == self.config.image_token_index
                _input_embeds[mask] = _image_features.view(-1, _image_features.shape[-1])
                model_input.append(_input_embeds.view(-1, _input_embeds.shape[-1]))
            model_input = torch.concat(model_input, dim=0)
                
        batch = self.batch_convert(pb_batch, input_ids)
        
        generations, next_batch, timings = self.generate_token(batch, model_input)
        if next_batch:
            self.language_model.batch_cache.set(next_batch)
        return generations, batch, timings
    
    @classmethod
    def batch_tokenized_inputs(self, requests, tokenizer, processor, config):
        batch_inputs = []
        image_inputs = []
        max_truncation = 0
        for r in requests:
            chunks = split(r.inputs)
            full_text = ""
            image_id = 0
            for chunk in chunks:
                if chunk["type"] == "text":
                    full_text += chunk["content"]
                elif chunk["type"] == "image":
                    image = chunk["content"]
                    if image.startswith("https://") or image.startswith("http://"):
                        image = processor.image_processor.fetch_images(image)
                    elif image.startswith("data:"):
                        image = load_data_uri(image)
                    else:
                        raise RuntimeError(
                            "Cannot process input image not starting with data:"
                        )
                    image_input = processor.image_processor(image, return_tensors="pt")
                    full_text += image_text_replacement(image_input, config, image_id)
                    image_inputs.append(image_input)
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk['type']}")
                
            batch_inputs.append(full_text)
            max_truncation = max(max_truncation, r.truncate)

        batch_tokenized_inputs = tokenizer(
            batch_inputs,
            truncation=True,
            max_length=max_truncation,
            add_special_tokens=not config.model_type == "paligemma",
        )["input_ids"]
        
        if image_inputs:
            image_input = image_inputs[0]
            new_image_inputs = {
                "pixel_values": torch.cat(
                    [img["pixel_values"] for img in image_inputs], dim=0
                ),
            }
            if "pixel_attention_mask" in image_input:
                new_image_inputs["pixel_attention_mask"] = torch.cat(
                    [img["pixel_attention_mask"] for img in image_inputs], dim=0
                )
            if "image_sizes" in image_input:
                new_image_inputs["image_sizes"] = torch.cat(
                    [img["image_sizes"] for img in image_inputs], dim=0
                )
            image_inputs = new_image_inputs
        else:
            image_inputs = None
        return batch_tokenized_inputs, image_inputs

    def batch_convert(
        self, 
        batchPb: generate_pb2.Batch,
        input_ids: torch.Tensor,
        pixel_values: Optional[List[torch.Tensor]] = None,
        pixel_attention_mask: Optional[List[torch.Tensor]] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
        ) -> LlavaBatch:
        request_contexts = []

        for i,request in enumerate(batchPb.requests):
            _input_ids = input_ids[i]
            
            parameters = request.parameters
            request_context = RequestContext(
                request.id,
                _input_ids,
                next_token_chooser_parameter=parameters,
                maxlen=min(request.stopping_parameters.max_new_tokens, 4096),
                stop_token_id=self.language_model.tokenizer.eos_token_id,
                is_stopped=False,
                request_kv_cache=RequestKvCache(
                    self.language_model.kvCachePool,
                    self.language_model.kvCachePool.page_len,
                    len(_input_ids),
                ),
                prefill_logprobs=request.prefill_logprobs,
                lora_id=request.lora_id,
            )

            request_contexts.append(request_context)

        return LlavaBatch(
            batch_id=batchPb.id, 
            is_prefill=True, 
            request_contexts=request_contexts,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_sizes=image_sizes,
        )
    
    def _merge_input_ids_with_image_features(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ):
        """In place merges in vision_embeddings with inputs_embeds."""
        mask = input_ids == self.config.image_token_index
        # Let's pray we have enabled enough slots !
        try:
            inputs_embeds[mask] = image_features.view(-1, image_features.shape[-1])
        except Exception as e:
            raise RuntimeError(
                f"Cannot fill images right now. If error happens at warmup, make sure you have enough `--max-input-tokens`  to handle images. If error happens at regular runtime, please fill in an issue: {e}"
            )
        return inputs_embeds

if __name__ == '__main__':
    model = LlavaLM(model_id='llava-hf/llava-v1.6-vicuna-7b-hf')
    print(model)