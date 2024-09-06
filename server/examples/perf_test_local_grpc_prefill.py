import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
import random, json

import torch
from test_cases import DEMO, LoraSpec
import numpy as np
import scipy.stats as stats

lora_specs = {}
for name, spec in DEMO.items():
    lora_prompts, base_prompts = spec.generate_prompts()
    lora_specs[name] = LoraSpec(lora_prompts, base_prompts)


def make_input(lora_id, lora_or_base, id=0, promptOverride=None):
    if lora_or_base == "lora":
        prompts = lora_specs[lora_id].lora_prompts
    elif lora_or_base == "base" or lora_or_base == "empty":
        prompts = lora_specs[lora_id].base_prompts
        lora_id = None
    else:
        raise ValueError(f"Unknown lora_or_base={lora_or_base}")
    prompt = random.choice(prompts) if not promptOverride else promptOverride
    inputs = prompt

    request = generate_pb2.Request(
        id=id,
        inputs=inputs,
        # truncate=256,
        prefill_logprobs=False,
        # top_n_tokens=20,
        parameters=generate_pb2.NextTokenChooserParameters(
            temperature=0.9,
            # top_k=10,
            # top_p=0.9,
            # typical_p=0.9,
            # repetition_penalty=1.1,
        ),
        stopping_parameters=generate_pb2.StoppingCriteriaParameters(
            max_new_tokens=8, stop_sequences=[], ignore_eos_token=True
        ),
        lora_id=lora_id,
    )  # to match the input in benchmark
    return request


promptOverride = "unt mollit anim id est laborum."  # to match the input in benchmark
global_request_id = 0
global_batch_id = 0


def generateBatch(batch_size: int):
    global global_request_id
    global global_batch_id
    requests = []
    for i in range(batch_size):
        requests.append(
            make_input(
                "tjluyao/gemma-2b-it-math",
                "base",
                id=global_request_id,
                promptOverride=promptOverride,
            )
        )
        global_request_id = global_request_id + 1
    batch_pb2 = generate_pb2.Batch(
        id=global_batch_id, requests=requests, size=len(requests)
    )
    global_batch_id = global_batch_id + 1
    return batch_pb2


num_tests = 100
batch_size = 32

forward_ms_all = []
decode_token_ms_all = []
decode_text_ms_all = []
total_ms_all = []

with grpc.insecure_channel("unix:///tmp/text-generation-server-0") as channel:
    stub = generate_pb2_grpc.TextGenerationServiceStub(channel)
    print(stub.Info(generate_pb2.InfoRequest()))
    warmupRequest = generate_pb2.WarmupRequest(
        batch=generateBatch(2),
        max_total_tokens=64,
        max_prefill_tokens=32,
        max_input_length=1024,
    )
    stub.Warmup(warmupRequest)
    for i in range(num_tests):
        batch = generateBatch(batch_size)
        prefillRequest = generate_pb2.PrefillRequest(batch=batch)
        response = stub.Prefill(prefillRequest)
        # print(response)
        decode_text_ms = (
            response.total_ns - response.forward_ns - response.decode_ns
        ) / 1e6
        forward_ms_all.append(response.forward_ns / 1e6)
        decode_token_ms_all.append(response.decode_ns / 1e6)
        decode_text_ms_all.append(decode_text_ms)
        total_ms_all.append(response.total_ns / 1e6)

        clearCacheRequest = generate_pb2.ClearCacheRequest(id=batch.id)
        stub.ClearCache(clearCacheRequest)
        torch.cuda.empty_cache()


# Calculate the 95% confidence interval for forward_ms_all
def calculate_confidence_interval(data, confidence_level=0.95):
    if data:
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Sample standard deviation
        n = len(data)
        degrees_freedom = n - 1
        return (
            stats.t.interval(
                confidence_level, degrees_freedom, loc=mean, scale=std / np.sqrt(n)
            ),
            mean,
            std,
        )


forward_confidence_interval, forward_mean, forward_std = calculate_confidence_interval(
    forward_ms_all
)
decode_token_confidence_interval, decode_token_mean, decode_token_std = (
    calculate_confidence_interval(decode_token_ms_all)
)
decode_text_confidence_interval, decode_text_mean, decode_text_std = (
    calculate_confidence_interval(decode_text_ms_all)
)
total_confidence_interval, total_mean, total_std = calculate_confidence_interval(
    total_ms_all
)

print(
    f"Forward Time - 95% Confidence Interval: {forward_confidence_interval}, Mean: {forward_mean}, Std: {forward_std}"
)
print(
    f"Decode Token Time - 95% Confidence Interval: {decode_token_confidence_interval}, Mean: {decode_token_mean}, Std: {decode_token_std}"
)
print(
    f"Decode Text Time - 95% Confidence Interval: {decode_text_confidence_interval}, Mean: {decode_text_mean}, Std: {decode_text_std}"
)
print(
    f"Total Time - 95% Confidence Interval: {total_confidence_interval}, Mean: {total_mean}, Std: {total_std}"
)
