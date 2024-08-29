import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
import random, json
from collections import defaultdict
from test_cases import DEMO, LoraSpec

# put this file in ROOT\server, so you don't need to compile TGI

# Start the local server:
# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=1 text_generation_server/cli.py serve meta-llama/Llama-2-7b-hf
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
            max_new_tokens=2048, stop_sequences=[], ignore_eos_token=True
        ),
        lora_id=lora_id,
    )
    return request


requests = [
    make_input(
        "tjluyao/gemma-2b-it-math",
        "base",
        id=0,
        promptOverride="What is deep learning?",
    ),
    make_input(
        "tjluyao/gemma-2b-it-math",
        "base",
        id=1,
        promptOverride="Give me a breif introduction to Byznatine Fault Tolerance and why it is important?",
    ),
]

# Assemble input batch
pb_batch_with_inputs = generate_pb2.Batch(id=0, requests=requests, size=len(requests))
pb_batch_empty = generate_pb2.Batch()

with grpc.insecure_channel("unix:///tmp/text-generation-server-0") as channel:
    stub = generate_pb2_grpc.TextGenerationServiceStub(channel)
    warmupRequest = generate_pb2.WarmupRequest(
        batch=pb_batch_with_inputs,
        max_total_tokens=2048,
        max_prefill_tokens=1024 * 10,
        max_input_length=1024,
    )
    warmupResult = stub.Warmup(warmupRequest)
    print(warmupResult)
    results = defaultdict(lambda: [])
    isPrefill = True
    while True:
        if isPrefill:
            prefill_request = generate_pb2.PrefillRequest(batch=pb_batch_with_inputs)
            prefill_response = stub.Prefill(prefill_request)
            generations, next_batch = (
                prefill_response.generations,
                prefill_response.batch,
            )
            isPrefill = False
        else:
            decode_request = generate_pb2.DecodeRequest(batches=[next_batch])
            decode_response = stub.Decode(decode_request)
            generations, next_batch = decode_response.generations, decode_response.batch

        for gen in generations:
            if gen.generated_text.text:
                results[gen.request_id] += [gen.generated_text.text]

        if all([g.generated_text.text for g in generations]):
            break

    for id in results:
        print(str(id) + "=" * 30)
        print("".join(results[id]))
