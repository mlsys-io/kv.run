import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
import random, json

import torch
from test_cases import DEMO, LoraSpec

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
        truncate=256,
        prefill_logprobs=True,
        top_n_tokens=20,
        parameters=generate_pb2.NextTokenChooserParameters(
            temperature=0.9,
            top_k=10,
            top_p=0.9,
            typical_p=0.9,
            repetition_penalty=1.1,
        ),
        stopping_parameters=generate_pb2.StoppingCriteriaParameters(
            max_new_tokens=32, stop_sequences=[], ignore_eos_token=True
        ),
        lora_id=lora_id,
    )
    return request


promptOverride = "What is deep learning?"
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


print(forward_ms_all)
print(decode_token_ms_all)
print(decode_text_ms_all)
print(total_ms_all)

average_forward_ms = sum(forward_ms_all) / len(forward_ms_all) if forward_ms_all else 0
average_decode_token_ms = (
    sum(decode_token_ms_all) / len(decode_token_ms_all) if decode_token_ms_all else 0
)
average_decode_text_ms = (
    sum(decode_text_ms_all) / len(decode_text_ms_all) if decode_text_ms_all else 0
)
average_total_ms = sum(total_ms_all) / len(total_ms_all) if total_ms_all else 0

print(f"Average Forward Time (ms): {average_forward_ms}")
print(f"Average Decode Token Time (ms): {average_decode_token_ms}")
print(f"Average Decode Text Time (ms): {average_decode_text_ms}")
print(f"Average Total Time (ms): {average_total_ms}")
