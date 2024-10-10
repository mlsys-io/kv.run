# Modified from transformers/models/llava_next/modeling_llava_next.py
# Editor: Junyi Shen

import torch
import time
from opentelemetry import trace
from typing import Optional, Tuple, List, Type, Iterable
from transformers import AutoTokenizer, AutoConfig, AutoProcessor, GenerationConfig

from text_generation_server.models.types import (
    Tokens,
    Generation,
    GeneratedText,
)

from text_generation_server.pb import generate_pb2
tracer = trace.get_tracer(__name__)

from text_generation_server.models_flashinfer.flashinfer_causal_lm import (
    FlashinferBatch,
    FlashinferLM,
    RequestContext, 
    RequestKvCache,
    DebugInfo,
    KvCachePool,
    find_padded_head_dim,
    PAGE_LEN,
    MEMORY_FRACTION,
)

from text_generation_server.models_flashinfer.custom_modeling.flashinfer_llama_modeling import FlashLlamaForCausalLM

from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
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
       
class FlashinferLlava(FlashinferLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        weights: Optional[Weights] = None,
    ):  
        # Initialize LlavaLM
        config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
        config.quantize = quantize
        config.vision_config.quantize = quantize
        config.speculator = None
        if not hasattr(config, "rms_norm_eps"):
            config.rms_norm_eps = 1e-05
        
        if not hasattr(config, "intermediate_size"):
            config.intermediate_size = 11008
        
        if not hasattr(config, "hidden_act"):
            config.hidden_act = "silu"
            
        if not hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = 32
            
        if not hasattr(config, "hidden_size"):
            config.hidden_size = 4096
        
        if not hasattr(config, "num_attention_heads"):
            config.num_attention_heads = 32
            
        if not hasattr(config, "num_key_value_heads"):
            config.num_key_value_heads = config.num_attention_heads

        if not hasattr(config, "rope_theta"):
            config.rope_theta = 1.0e4
            
        dtype = dtype or torch.float16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.process_group, rank, world_size = initialize_torch_distributed()
        torch.distributed.barrier(group=self.process_group)
        
        if not weights:
            filenames = weight_files(model_id, revision=revision, extension=".safetensors")
            weights = Weights(filenames, device, dtype, process_group=self.process_group)
            if config.quantize in ["gptq", "awq"]:
                weights._set_gptq_params(model_id, revision)
                
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id, revision=revision, trust_remote_code=trust_remote_code
            )
            if isinstance(generation_config.eos_token_id, (list, set)):
                # TODO Huge hack
                tokenizer._eos_token_ids = set(generation_config.eos_token_id)
        except Exception:
            pass
        
        model = FlashLlamaForCausalLM("language_model", config, weights)
        
        self.vision_tower = load_vision_model(
            prefix="vision_tower",
            config=config.vision_config,
            weights=weights,
        )

        self.multi_modal_projector = LlavaNextMultiModalProjector(
            prefix="multi_modal_projector", config=config, weights=weights
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        
        self.image_newline = weights.get_tensor("image_newline")
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
        self.id_embedder = model.embed_tokens
        self.vision_feature_select_strategy = config.vision_feature_select_strategy
        self.config = config

        super(FlashinferLlava, self).__init__(
            model=model,
            tokenizer=tokenizer,
            config=config,
            dtype=dtype,
            device=device,
            lora_ids=None,
        )

    def _convertPbBatch(self, batchPb: generate_pb2.Batch) -> FlashinferBatch:
        batch_tokenized_inputs, image_inputs = self.batch_tokenize_inputs(
            batchPb.requests, 
            self.tokenizer, 
            self.processor, 
            self.config
        )
        
        if image_inputs is not None:
            pixel_values = image_inputs["pixel_values"].to(device=self.device)
            if "pixel_attention_mask" in image_inputs:
                pixel_attention_mask = image_inputs["pixel_attention_mask"].to(
                    device=self.vision_tower.device
                )
            else:
                pixel_attention_mask = None
            if "image_sizes" in image_inputs:
                image_sizes = image_inputs["image_sizes"].to(device=self.device)
            else:
                image_sizes = None
        else:
            pixel_values = None
            pixel_attention_mask = None
            image_sizes = None
            
        # Embed the input
        input_ids = [torch.tensor(_tokenized, device=self.device) for _tokenized in batch_tokenized_inputs]
        inputs_embeds = [self.id_embedder(_input_ids) for _input_ids in input_ids]
        model_input = None
        
        if pixel_values is not None and len(pixel_values) > 0:
            num_images, num_patches, channels, height, width = pixel_values.shape
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
            
        #Inicialize the kvCache
        request_contexts = []

        for i,request in enumerate(batchPb.requests):
            _input_ids = input_ids[i]
            parameters = request.parameters
            request_context = RequestContext(
                request.id,
                _input_ids,
                next_token_chooser_parameter=parameters,
                maxlen=min(request.stopping_parameters.max_new_tokens, 4096),
                stop_token_id=self.tokenizer.eos_token_id,
                is_stopped=False,
                request_kv_cache=RequestKvCache(
                    self.kvCachePool,
                    self.kvCachePool.page_len,
                    len(_input_ids),
                ),
                prefill_logprobs=request.prefill_logprobs,
                lora_id=request.lora_id,
            )

            request_contexts.append(request_context)

        return FlashinferBatch(
            batch_id=batchPb.id, is_prefill=True, request_contexts=request_contexts
        ), model_input
    
    def prefill_batch(
        self, batchPb: generate_pb2.Batch, debug_mode: bool = False
    ) -> Tuple[List[Generation], Optional[FlashinferBatch], DebugInfo]:
        batch, embeddings = self._convertPbBatch(batchPb)
        generations, next_batch, debug_info = self.generate_token(batch, debug_mode, embeddings)
        if next_batch:
            self.batch_cache.set(next_batch)
        return generations, batch, debug_info
    
    @classmethod
    def batch_tokenize_inputs(self, requests, tokenizer, processor, config):
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
    
    @tracer.start_as_current_span("generate_token")
    @torch.no_grad()
    def generate_token(
        self, batch: FlashinferBatch, debug_mode: bool = False, embeddings: torch.Tensor = None
    ) -> Tuple[List[Generation], Optional[FlashinferBatch], DebugInfo]:
        start = time.time_ns()
        input_ids_tensor, all_input_ids_tensor, batch_position, loraWeights = (
            self._prepare_model_inputs(batch)
        )
        raw_logits, _ = self.model(
            input_ids_tensor if embeddings is None else None,
            self.kvCachePool,
            batch.is_prefill,
            batch_position,
            loraWeights,
            input_embeddings = embeddings,
        )
        torch.cuda.synchronize()
        start_decode_token = time.time_ns()

        logits = (
            raw_logits[batch_position.seq_indptr[1:] - 1]
            if batch.is_prefill
            else raw_logits
        )
        
        next_token_ids, next_token_logprobs, alllogprobs, _, _ = (
            self._get_next_batch_token_id_heterogeneous(
                batch.request_contexts, all_input_ids_tensor, logits
            )
        )

        next_token_ids_list = next_token_ids.tolist()
        next_token_logprobs_list = next_token_logprobs.tolist()
        torch.cuda.synchronize()
        start_decode_text = time.time_ns()
        generations, all_stop = self._decode_text(
            batch, next_token_ids_list, next_token_logprobs_list
        )
        torch.cuda.synchronize()
        end = time.time_ns()

        forward_ns = start_decode_token - start
        decode_ns = start_decode_text - start_decode_token
        total_ns = end - start
        debug_info = (
            DebugInfo(
                forward_ns=forward_ns,
                decode_ns=decode_ns,
                total_ns=total_ns,
                concat_ns=None,
                all_log_probs=alllogprobs,
            )
            if debug_mode
            else DebugInfo(
                forward_ns=forward_ns,
                decode_ns=decode_ns,
                total_ns=total_ns,
                concat_ns=None,
                all_log_probs=None,
            )
        )

        # The router stops generation only when batch=None
        if all_stop:
            return generations, None, debug_info
        else:
            return generations, batch, debug_info

    def warmup(self, batchPb: generate_pb2.Batch):
        if not self.kvCachePool:
            head_dim_padded = find_padded_head_dim(
                self.model_config.hidden_size // self.model_config.num_attention_heads
            )
            dtype_size = torch.tensor([], dtype=self.dtype).element_size()
            cache_page_size = (
                2
                * PAGE_LEN
                * self.model_config.num_hidden_layers
                * self.model_config.num_attention_heads
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
            num_pages_to_allocate = int(free_memory * 0.60 / cache_page_size)
            print(
                f"Cache allocation:\n"
                f"  Cache Page Size: {cache_page_size / 1024 / 1024} MB\n"
                f"  Dtype Size: {dtype_size}\n"
                f"  Free Memory: {free_memory / 1024 / 1024 / 1024} GB\n"
                f"  Total GPU Memory: {total_gpu_memory / 1024 / 1024 / 1024} GB\n"
                f"  Number of Pages to Allocate: {num_pages_to_allocate}"
            )

            self.kvCachePool = KvCachePool(
                max_pages=num_pages_to_allocate,
                num_layers=self.model_config.num_hidden_layers,
                num_heads=self.model_config.num_key_value_heads,
                head_dim=head_dim_padded,
                page_len=PAGE_LEN,
                dtype=self.dtype,
                device=self.device,
            )

        num_free_pages = self.kvCachePool.num_free_pages()
        batch, _ = self._convertPbBatch(batchPb)
        self.generate_token(batch)
        for request_context in batch.request_contexts:
            request_context.request_kv_cache.release()
        return num_free_pages * PAGE_LEN
  
if __name__ == '__main__':
    model = FlashinferLlava(model_id='llava-hf/llava-v1.6-vicuna-7b-hf')
    print(model)