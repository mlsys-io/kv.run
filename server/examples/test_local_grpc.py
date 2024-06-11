import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
import random, json
from test_cases import DEMO, LoraSpec

# put this file in ROOT\server, so you don't need to compile TGI

# Start the local server:
# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=1 text_generation_server/cli.py serve meta-llama/Llama-2-7b-hf
lora_specs = {}
for name, spec in DEMO.items():
    lora_prompts, base_prompts = spec.generate_prompts()
    lora_specs[name] = LoraSpec(lora_prompts, base_prompts)


def make_input(lora_id, lora_or_base):
    if lora_or_base == "lora":
        prompts = lora_specs[lora_id].lora_prompts
    elif lora_or_base == "base" or lora_or_base == "empty":
        prompts = lora_specs[lora_id].base_prompts
        lora_id = "empty"
    else:
        raise ValueError(f"Unknown lora_or_base={lora_or_base}")
    prompt = random.choice(prompts)
    inputs = prompt

    request = generate_pb2.Request(
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


req1 = make_input("gsm8k", "base")
req2 = make_input("gsm8k", "lora")
requests = [req1, req2]

# Assemble input batch
pb_batch_with_inputs = generate_pb2.Batch(id=0, requests=requests, size=len(requests))
pb_batch_empty = generate_pb2.Batch()

with grpc.insecure_channel("unix:///tmp/text-generation-server-0") as channel:
    stub = generate_pb2_grpc.TextGenerationServiceStub(channel)

    # Test adapter loading and offloading
    stub.AdapterControl(
        generate_pb2.AdapterControlRequest(lora_ids="all", operation="remove")
    )
    stub.AdapterControl(
        generate_pb2.AdapterControlRequest(
            lora_ids="gsm8k:abcdabcd987/gsm8k-llama2-7b-lora-16,sqlctx:abcdabcd987/sqlctx-llama2-7b-lora-16,viggo:abcdabcd987/viggo-llama2-7b-lora-16",
            operation="load",
        )
    )
    resp = stub.AdapterControl(generate_pb2.AdapterControlRequest(operation="status"))
    print(resp)

    # Info
    print(stub.Info(generate_pb2.InfoRequest()))
    # Warm up
    wr = generate_pb2.WarmupRequest(
        batch=pb_batch_with_inputs,
        max_total_tokens=2048,
        max_prefill_tokens=1024 * 10,
        max_input_length=1024,
    )
    stub.Warmup(wr)
    # Prefill
    pr = generate_pb2.PrefillRequest(batch=pb_batch_empty)
    resp = stub.Prefill(pr)
    gen, cbatch = resp.generations, resp.batch
    # Decode
    dr = generate_pb2.DecodeRequest(batches=[cbatch])
    resp = stub.Decode(dr)
    gen, cbatch = resp.generations, resp.batch

    results = {}
    # Generate token
    pr = generate_pb2.GenerateTokenRequest(batch=pb_batch_empty)
    while True:
        resp = stub.GenerateToken(pr)
        generations, cbatch = resp.generations, resp.batch
        if not generations:
            break
        for gen in generations:
            if gen.request_id in results:
                results[gen.request_id].append(gen.tokens.texts[0])
            else:
                results[gen.request_id] = [gen.tokens.texts[0]]
    for id in results:
        print(str(id) + "=" * 30)
        print("".join(results[id]))
    print("done")
