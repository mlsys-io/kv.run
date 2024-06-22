import torch
import torch.distributed
from typing import Any, TypedDict, Optional
from text_generation_server.utils.lora_utils import ModelLoraManager, ModelConfigForLora
from text_generation_server.utils.cache_manager_flashinfer import (
    ModelKvCache,
    KvCachePool,
)
from text_generation_server.utils.tokens import (
    StopSequenceCriteria,
    StoppingCriteria,
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
from text_generation_server.models.causal_lm import CausalLMBatch
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.utils import (
    NextTokenChooser,
    StoppingCriteria,
)
from text_generation_server.utils.dist import MEMORY_FRACTION
from dataclasses import dataclass

tracer = trace.get_tracer(__name__)


class TextGenerationChunk(TypedDict):
    index: int
    token_id: int
    text: str
    is_stop: bool


@dataclass
class FlashinferBatch(CausalLMBatch):
    @classmethod
    def Empty(cls, batch_id):
        return cls(
            batch_id=batch_id,
            requests=None,
            prefix_offsets=None,
            read_offsets=None,
            next_token_choosers=None,
            stopping_criterias=None,
            top_n_tokens=None,
            top_n_tokens_tensor=None,
            input_ids=None,
            requests_idx_mapping=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            all_input_ids=None,
            input_lengths=None,
            max_input_length=None,
            padding_right_offset=None,
            max_tokens=None,
        )

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase = None,
        dtype: torch.dtype = None,
        device: torch.device = "cuda",
    ) -> "CausalLMBatch":
        input_ids = []
        next_token_choosers = []
        stopping_criterias = []
        top_n_tokens = []
        prefix_offsets = []
        read_offsets = []

        # Parse batch
        for i, r in enumerate(pb.requests):
            prompt = r.inputs

            next_token_choosers.append(
                NextTokenChooser.from_pb(r.parameters, device, tokenizer)
            )
            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(r.top_n_tokens)
            tokenized_inputs = tokenizer.encode(prompt)
            input_len = len(tokenized_inputs)
            prefix_offsets.append(input_len - 5)
            read_offsets.append(input_len)
            input_ids.append(tokenized_inputs)

        top_n_tokens_tensor = torch.tensor(
            top_n_tokens, device=device, dtype=torch.int64
        )

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=None,
            input_ids=input_ids,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            all_input_ids=None,
            input_lengths=None,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            max_input_length=None,
            padding_right_offset=None,
            max_tokens=None,
        )


class RequestContext:
    def __init__(
        self,
        input_ids: list[int],
        lora_id: str,
        tokenizer,
        *,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        maxlen: int,
        stop_token_id: int,
        prefill_logprobs: bool = True,
    ):
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.maxlen = maxlen
        self.stop_token_id = stop_token_id
        self.prefill_logprobs = prefill_logprobs

        # Logits processing adapted from: https://github.com/lm-sys/FastChat/blob/bb7ca37c2bfad629ba4751dec188bdcdc2cf0c81/fastchat/serve/inference.py
        self.logits_processor = transformers.LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            self.logits_processor.append(
                transformers.TemperatureLogitsWarper(temperature)
            )
        if repetition_penalty > 1.0:
            self.logits_processor.append(
                transformers.RepetitionPenaltyLogitsProcessor(repetition_penalty)
            )
        if 0 < top_p < 1.0:
            self.logits_processor.append(transformers.TopPLogitsWarper(top_p))
        if top_k > 0:
            self.logits_processor.append(transformers.TopKLogitsWarper(top_k))

        self.output_ids = [int(x) for x in input_ids]
        self.prompt_len = len(self.output_ids)
        self.lora_id = lora_id
        self.tokenizer = tokenizer
        self.prefix_offset = 0
        self.read_offset = 0

    def get_next_token_id(self, logits: torch.Tensor) -> int:
        if self.logits_processor:
            if self.repetition_penalty > 1.0:
                t = torch.as_tensor([self.output_ids], device=logits.device)
            else:
                t = None
            last_token_logits = self.logits_processor(t, logits[-1].unsqueeze(0))[0]
        else:
            last_token_logits = logits[-1, :]

        if self.temperature <= 0 or self.top_p <= 0:
            _, indices = torch.topk(last_token_logits, 2)
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
        token = int(indices.tolist()[0])
        return token

    def append_token(self, token_id: int):
        self.output_ids.append(token_id)

    def is_stop(self) -> FinishReason:
        if len(self.output_ids) - self.prompt_len >= self.maxlen:
            return FinishReason.FINISH_REASON_LENGTH
        if self.output_ids[-1] == self.stop_token_id:
            return FinishReason.FINISH_REASON_EOS_TOKEN
        return None

    def is_prefill(self) -> bool:
        return len(self.output_ids) == self.prompt_len


class FlashinferLM(Model):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: PretrainedConfig,
        dtype: torch.dtype,
        device: torch.device,
        lora_ids: List[str] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.model_config = config

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

        # TODO: consider moving it into cache manager
        PAGE_LEN = 16
        head_dim_padded = find_padded_head_dim(
            config.hidden_size // config.num_attention_heads
        )
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        cache_page_size = (
            2
            * PAGE_LEN
            * config.num_hidden_layers
            * config.num_attention_heads
            * head_dim_padded
            * dtype_size
        )

        currentDevice = torch.cuda.current_device()
        total_free_memory, _ = torch.cuda.mem_get_info(currentDevice)
        total_gpu_memory = torch.cuda.get_device_properties(currentDevice).total_memory
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

        kvCachePool = KvCachePool(
            max_pages=num_pages_to_allocate,
            num_layers=self.model_config.num_hidden_layers,
            num_heads=self.model_config.num_key_value_heads,
            head_dim=head_dim_padded,
            page_len=PAGE_LEN,
            dtype=dtype,
            device=device,
        )

        self.modelKvCache = ModelKvCache(kvCachePool)
        self.model_config_for_lora = ModelConfigForLora(
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
        )

        self.loraManager = ModelLoraManager(self.model_config_for_lora, dtype)
        if lora_ids:
            self.loraManager.set_lora_weights(
                lora_ids, self.model_config_for_lora or {}, dtype
            )
        self.reqctx: dict[int, RequestContext] = {}

        super(FlashinferLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    def _find_padded_head_dim(self, head_dim):
        flashInferDimensions = [64, 128, 256]
        for dim in flashInferDimensions:
            if head_dim <= dim:
                return dim
        raise ValueError("The head dimension is too large for FlashInfer")

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

    def has_request(self):
        return len(self.reqctx) > 0

    @property
    def batch_type(self) -> Type[FlashinferBatch]:
        return FlashinferBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def add_request(self, batch: FlashinferBatch):
        ids = []
        for r in range(len(batch.requests)):
            id = batch.requests[r].id
            # Router sends initial request in each iteration
            if id not in self.reqctx:
                lora_id = batch.requests[r].lora_id or "empty"
                input = batch.input_ids[r]
                parameters = batch.requests[r].parameters
                stop = batch.requests[r].stopping_parameters
                prefill_logprobs = batch.requests[r].prefill_logprobs

                if lora_id not in self.loraManager.lora_weights_cpu:
                    raise ValueError("Cannot find lora weights", lora_id)

                self.reqctx[id] = RequestContext(
                    input,
                    lora_id,
                    self.tokenizer,
                    temperature=parameters.temperature,
                    repetition_penalty=parameters.repetition_penalty,
                    top_p=parameters.top_p,
                    top_k=parameters.top_k,
                    maxlen=min(stop.max_new_tokens, 4096),
                    stop_token_id=self.tokenizer.eos_token_id,
                    prefill_logprobs=prefill_logprobs,
                )
                ids.append(id)
        return ids

    def warmup(self, batch: FlashinferBatch):
        pass

    @tracer.start_as_current_span("generate_token")
    @torch.no_grad()
    def generate_token(
        self, batch: FlashinferBatch
    ) -> Tuple[List[Generation], Optional[FlashinferBatch], Tuple[int, int]]:
        start = time.time_ns()

        if hasattr(batch, "requests") and batch.requests:
            ids = self.add_request(batch)

        if not self.reqctx:
            return None, batch, (0, 0)

        reqs = sorted(
            self.reqctx.items(),
            key=lambda req: (not req[1].is_prefill(), req[1].lora_id),
        )

        input_ids = []
        lora_ids, lora_lens = [], []
        batchKvCache = self.modelKvCache.getOrCreate(batch.batch_id)
        prefill_reqIds = []
        decode_reqIds = []

        for requestId, req in reqs:
            req.prefill = req.is_prefill()
            if req.prefill:
                input_ids.extend(req.output_ids)
                prefill_reqIds.append(requestId)
                batchKvCache.create(requestId, req.prompt_len)
            else:
                input_ids.append(req.output_ids[-1])
                decode_reqIds.append(requestId)
                batchKvCache.get(requestId).increment()
            if lora_ids and lora_ids[-1] == req.lora_id:
                lora_lens[-1] += 1
            else:
                lora_ids.append(req.lora_id)
                lora_lens.append(1)

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        )

        prefillBatchPosition = batchKvCache.getKvCacheBatchPosition(
            prefill_reqIds, isPrefill=True
        )
        decodeBatchPosition = batchKvCache.getKvCacheBatchPosition(
            decode_reqIds, isPrefill=False
        )

        # Forward pass
        raw_logits, _ = self.model(
            input_ids,
            self.modelKvCache.kvCachePool,
            prefillBatchPosition,
            decodeBatchPosition,
            self.loraManager.get_lora_batched_weights(lora_ids, lora_lens),
        )

        start_decode = time.time_ns()

        prefill_logits = (
            raw_logits[prefillBatchPosition.seq_indptr[1:] - 1]
            if prefillBatchPosition.total_seq_len > 0
            else torch.tensor([], device=self.device)
        )
        decode_logits = raw_logits[prefillBatchPosition.total_seq_len :]
        logits = torch.cat([prefill_logits, decode_logits])

        all_stop = True
        generations: List[Generation] = []
        for i, (reqid, reqctx) in enumerate(reqs):
            next_token_id = reqctx.get_next_token_id(logits[i].unsqueeze(0))
            reqctx.append_token(next_token_id)
            # text = reqctx.decode_tokens() # todo: ??
            text = self.tokenizer.decode(
                next_token_id,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )

            is_stop = reqctx.is_stop()
            if is_stop != None:
                output_text = self.tokenizer.decode(
                    reqctx.output_ids[reqctx.prompt_len :],
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
                generated_text = GeneratedText(
                    output_text,
                    len(reqctx.output_ids) - reqctx.prompt_len + 1,
                    is_stop,
                    None,
                )
                self.reqctx.pop(reqid)
                batchKvCache.release(reqid)
            else:
                generated_text = None
                all_stop = False

            # Prefill
            if reqctx.prefill:  # and reqctx.prefill_logprobs:
                # Remove generated token to only have prefill and add nan for first prompt token
                prefill_logprobs = []  # todo
                prefill_token_ids = reqctx.output_ids[: reqctx.prompt_len]
                prefill_texts = self.tokenizer.batch_decode(
                    prefill_token_ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
                reqctx.prefill_tokens = Tokens(
                    prefill_token_ids,
                    prefill_logprobs,
                    prefill_texts,
                    is_special=[],
                )
                reqctx.prefix_offset = reqctx.prompt_len
            else:
                reqctx.prefill_tokens = None

            generation = Generation(
                reqid,
                reqctx.prefill_tokens,
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
        decode_ns = time.time_ns() - start_decode
        # The router stops generation only when batch=None
        if all_stop:
            batch = None
        return generations, batch, (forward_ns, decode_ns)
