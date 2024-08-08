import torch
import torch.distributed
from typing import Any, Optional
from text_generation_server.utils.lora_utils import ModelLoraManager, ModelConfigForLora
from text_generation_server.utils.cache_manager_flashinfer import (
    getKvCacheBatchPosition,
    KvCacheBatchPosition,
    KvCachePool,
    RequestKvCache,
    PAGE_LEN,
)
from text_generation_server.utils.tokens import (
    FinishReason,
)
from text_generation_server.layers.flashinfer_attention import find_padded_head_dim
from transformers import PreTrainedTokenizerBase, PretrainedConfig
import transformers
from text_generation_server.pb import generate_pb2

import time
from opentelemetry import trace
from typing import Optional, Tuple, List, Type, Dict
from text_generation_server.models import Model
from text_generation_server.models.types import (
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.utils import HeterogeneousNextTokenChooser
from text_generation_server.utils.dist import MEMORY_FRACTION
from dataclasses import dataclass
from collections.abc import Iterable
from text_generation_server.cache import Cache

tracer = trace.get_tracer(__name__)


class RequestContext:
    def __init__(
        self,
        request_id: str,
        input_ids: list[int],
        *,
        next_token_chooser_parameter: generate_pb2.NextTokenChooserParameters,
        maxlen: int,
        stop_token_id: int,
        is_stopped: bool,
        request_kv_cache: RequestKvCache,
        prefill_logprobs: bool = True,
        lora_id: str = "empty",
    ):
        self.request_id = request_id
        self.maxlen = maxlen
        self.stop_token_id = stop_token_id
        self.prefill_logprobs = prefill_logprobs
        self.next_token_chooser_parameter = next_token_chooser_parameter
        self.output_ids = [int(x) for x in input_ids]
        self.prompt_len = len(self.output_ids)
        self.lora_id = lora_id
        self.prefix_offset = 0
        self.read_offset = 0
        self.is_stopped = is_stopped
        self.prefill_tokens: Optional[Tokens] = None
        self.request_kv_cache = request_kv_cache

    def append_token(self, token_id: int):
        self.output_ids.append(token_id)

    def get_stop_reason(self) -> FinishReason:
        if len(self.output_ids) - self.prompt_len >= self.maxlen:
            return FinishReason.FINISH_REASON_LENGTH
        if self.output_ids[-1] == self.stop_token_id:
            return FinishReason.FINISH_REASON_EOS_TOKEN
        return None


@dataclass(frozen=True)
class FlashinferBatch:
    batch_id: int
    is_prefill: bool
    request_contexts: List[RequestContext]

    def to_pb(self) -> generate_pb2.CachedBatch:

        max_input_length = max([r.prompt_len for r in self.request_contexts])
        max_decode_tokens = max([r.maxlen for r in self.request_contexts])
        max_tokens = len(self.request_contexts) * (max_input_length + max_decode_tokens)

        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[
                request_context.request_id for request_context in self.request_contexts
            ],
            size=len(self.request_contexts),
            max_tokens=max_tokens,
        )


class FlashinferLM(Model):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: PretrainedConfig,
        dtype: torch.dtype,
        device: torch.device,
        lora_ids: List[str],
        model_type: str = "llama",
    ):
        self.device = device
        self.dtype = dtype
        self.model_config = config
        self.batch_cache = Cache()

        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and config.quantize != "bitsandbytes"
        ):
            model = model.cuda()

        if tokenizer.pad_token_id is None:
            if config.pad_token_id is not None:
                tokenizer.pad_token_id = config.pad_token_id
            elif config.eos_token_id is not None:
                tokenizer.pad_token_id = config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.model_config_for_lora = ModelConfigForLora(
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
        )

        self.loraManager = ModelLoraManager(self.model_config_for_lora, dtype, model_type=model_type)
        if lora_ids:
            self.loraManager.set_lora_weights(
                lora_ids, self.model_config_for_lora, dtype
            )

        self.kvCachePool = None

        super(FlashinferLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    def load_lora_adapters(self, lora_ids: List[str]):
        self.loraManager.set_lora_weights(
            lora_ids,
            self.model_config_for_lora,
            dtype=self.dtype,
        )

    def remove_lora_adapters(self, lora_ids: list[str] = None):
        self.loraManager.remove_lora_weights(lora_ids)

    def get_lora_adapters(self):
        return list(self.loraManager.lora_weights_cpu)

    def decode_batch(
        self, cachedBatchesPb: Iterable[generate_pb2.CachedBatch]
    ) -> Tuple[List[Generation], Optional[FlashinferBatch], Tuple[int, int], int]:
        start_concat = time.time_ns()
        batch = self._convertCachedBatch(cachedBatchesPb)
        concat_ns = time.time_ns() - start_concat
        generations, next_batch, timings = self.generate_token(batch)
        if next_batch:
            self.batch_cache.set(next_batch)
        return generations, next_batch, timings, concat_ns

    def prefill_batch(
        self, batchPb: generate_pb2.Batch
    ) -> Tuple[List[Generation], Optional[FlashinferBatch], Tuple[int, int]]:
        batch = self._convertPbBatch(batchPb)
        generations, next_batch, timings = self.generate_token(batch)
        if next_batch:
            self.batch_cache.set(next_batch)
        return generations, batch, timings

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
            num_pages_to_allocate = int(free_memory * 0.80 / cache_page_size)
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
        batch = self._convertPbBatch(batchPb)
        self.generate_token(batch)
        for request_context in batch.request_contexts:
            request_context.request_kv_cache.release()
        return num_free_pages * PAGE_LEN

    def filter_batch(self, batch_id: int) -> Optional[FlashinferBatch]:
        batch = self.batch_cache.pop(batch_id)
        if batch is None:
            raise ValueError(f"Batch ID {batch_id} not found in cache.")
        self.batch_cache.set(batch)
        return batch

    def clear_cache(self):
        all_batches: List[FlashinferBatch] = self.batch_cache.get_all_values()
        for batch in all_batches:
            for request_context in batch.request_contexts:
                request_context.request_kv_cache.release()

        self.batch_cache.clear()

    def _find_padded_head_dim(self, head_dim):
        flashInferDimensions = [64, 128, 256]
        for dim in flashInferDimensions:
            if head_dim <= dim:
                return dim
        raise ValueError("The head dimension is too large for FlashInfer")

    def _convertPbBatch(self, batchPb: generate_pb2.Batch) -> FlashinferBatch:
        request_contexts = []

        for request in batchPb.requests:
            prompt = request.inputs
            input_ids = self.tokenizer.encode(prompt)
            parameters = request.parameters
            request_context = RequestContext(
                request.id,
                input_ids,
                next_token_chooser_parameter=parameters,
                maxlen=min(request.stopping_parameters.max_new_tokens, 4096),
                stop_token_id=self.tokenizer.eos_token_id,
                is_stopped=False,
                request_kv_cache=RequestKvCache(
                    self.kvCachePool,
                    self.kvCachePool.page_len,
                    len(input_ids),
                ),
                prefill_logprobs=request.prefill_logprobs,
                lora_id=request.lora_id,
            )

            request_contexts.append(request_context)

        return FlashinferBatch(
            batch_id=batchPb.id, is_prefill=True, request_contexts=request_contexts
        )

    def _convertCachedBatch(
        self, cachedBatchesPb: Iterable[generate_pb2.CachedBatch]
    ) -> FlashinferBatch:
        batches: List[FlashinferBatch] = []
        for batch_pb in cachedBatchesPb:
            batch = self.batch_cache.pop(batch_pb.id)
            if batch is None:
                raise ValueError(f"Batch ID {batch_pb.id} not found in cache.")
            batches.append(batch)

        if len(batches) == 0:
            raise ValueError("All batches are empty")

        request_contexts_combined: List[RequestContext] = []
        for batch in batches:
            request_contexts_combined.extend(batch.request_contexts)

        return FlashinferBatch(
            batch_id=batches[0].batch_id,
            is_prefill=False,
            request_contexts=request_contexts_combined,
        )

    def _get_all_input_ids_tensor(
        self,
        all_input_ids_stacked: List[List[int]],
        request_contexts: List[RequestContext],
    ):
        max_input_length = max(
            [
                (request_context.maxlen + request_context.prompt_len)
                for request_context in request_contexts
            ]
        )
        all_input_ids_padded = [
            input_ids + [0] * (max_input_length - len(input_ids))
            for input_ids in all_input_ids_stacked
        ]
        return torch.tensor(
            all_input_ids_padded,
            dtype=torch.long,
            device=self.device,
        )

    def _get_next_batch_token_id_heterogeneous(
        self,
        request_contexts: List[RequestContext],
        all_input_ids_tensor: torch.Tensor,
        logits: torch.Tensor,
    ) -> List[int]:
        next_token_chooser_parameters = [
            request_context.next_token_chooser_parameter
            for request_context in request_contexts
            if not request_context.is_stopped
        ]
        next_token_chooser = HeterogeneousNextTokenChooser(
            watermark=[
                parameter.watermark for parameter in next_token_chooser_parameters
            ],
            temperature=[
                parameter.temperature for parameter in next_token_chooser_parameters
            ],
            repetition_penalty=[
                parameter.repetition_penalty
                for parameter in next_token_chooser_parameters
            ],
            frequency_penalty=[
                parameter.frequency_penalty
                for parameter in next_token_chooser_parameters
            ],
            top_k=[parameter.top_k for parameter in next_token_chooser_parameters],
            top_p=[parameter.top_p for parameter in next_token_chooser_parameters],
            typical_p=[
                parameter.typical_p for parameter in next_token_chooser_parameters
            ],
            do_sample=[
                parameter.do_sample for parameter in next_token_chooser_parameters
            ],
            seeds=[parameter.seed for parameter in next_token_chooser_parameters],
            device=self.device,
            dtype=self.dtype,
            tokenizer=self.tokenizer,
            grammars=[parameter.grammar for parameter in next_token_chooser_parameters],
            grammar_types=[
                parameter.grammar_type for parameter in next_token_chooser_parameters
            ],
            fsm_grammar_states=[0] * len(next_token_chooser_parameters),
        )

        return next_token_chooser(
            all_input_ids_tensor,
            logits,
            0,
            None,
            None,
        )

    def batch_type(self):
        return FlashinferBatch

    @tracer.start_as_current_span("generate_token")
    @torch.no_grad()
    def generate_token(
        self, batch: FlashinferBatch
    ) -> Tuple[List[Generation], Optional[FlashinferBatch], Tuple[int, int]]:
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

        all_input_ids_tensor = self._get_all_input_ids_tensor(
            all_input_ids_stacked, batch.request_contexts
        )
        input_ids_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        )

        batch_position: KvCacheBatchPosition = getKvCacheBatchPosition(
            request_kv_caches, isPrefill=batch.is_prefill, device=self.device
        )

        loraWeights = (
            self.loraManager.get_lora_batched_weights(lora_ids, lora_lens)
            if lora_ids
            else None
        )
        raw_logits, _ = self.model(
            input_ids_tensor,
            self.kvCachePool,
            batch.is_prefill,
            batch_position,
            loraWeights,
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
            self._get_next_batch_token_id_heterogeneous(
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
            text = self.tokenizer.decode(
                next_token_id,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )

            stop_reason = request_context.get_stop_reason()
            if stop_reason != None:
                output_text = self.tokenizer.decode(
                    request_context.output_ids[request_context.prompt_len :],
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
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
                    [next_token_id in self.all_special_ids],
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
