from text_generation_server.pb import generate_pb2
import torch
from text_generation_server.models.flashinfer_llama import FlashinferLlama
from text_generation_server.models.flashinfer_gemma import FlashinferGemma

try:
    from text_generation_server.models.flashinfer_mistral import FlashinferMistral
    from text_generation_server.models.flashinfer_phi import FlashinferPhi
    from text_generation_server.models.flashinfer_qwen2 import FlashinferQwen2
except:
    print("can't load flashinfer mistral and phi and qwen2 without flash attn")

from text_generation_server.models.flashinfer_causal_lm import FlashinferBatch
import random, json
from test_cases import DEMO, LoraSpec

# Load demo inputs
lora_specs = {}
for name, spec in DEMO.items():
    lora_prompts, base_prompts = spec.generate_prompts()
    lora_specs[name] = LoraSpec(lora_prompts, base_prompts)


# Create input requests
def make_input(lora_id, lora_or_base, id=0, promptOverride=None):
    if lora_or_base == "lora":
        prompts = lora_specs[lora_id].lora_prompts
    elif lora_or_base == "base" or lora_or_base == "empty":
        prompts = lora_specs[lora_id].base_prompts
        lora_id = "empty"
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


test = "gemma"
# test = 'llama-3'
# test = 'llama-3-70'
# test = 'llama-2'
# test = 'mistral'
# test = 'qwen2'
# test = 'qwen2-1.8'
# test = 'qwen2-70'

if test == "llama-2":
    # Load model
    service = FlashinferLlama(
        model_id="meta-llama/Llama-2-7b-hf",
        lora_ids=["abcdabcd987/gsm8k-llama2-7b-lora-16"],
    )
    # Create an input batch of two queries
    requests = [
        make_input(
            "abcdabcd987/gsm8k-llama2-7b-lora-16",
            "base",
            id=0,
            promptOverride="Give me a breif introduction to Byznatine Fault Tolerance and why it is important?",
        ),
        make_input(
            "abcdabcd987/gsm8k-llama2-7b-lora-16",
            "lora",
            id=1,
            promptOverride="Which network interface card is more suitable for distributed systems, Meallanox or Broadcom?",
        ),
    ]
elif test == "llama-3":
    # Load model
    service = FlashinferLlama(
        model_id="tjluyao/llama-3-8b",
        lora_ids=["tjluyao/llama-3-8b-math", "tjluyao/llama-3-8b-zh"],
    )
    # Test load lora adapters
    print(service.get_lora_adapters())
    # Test remove lora adapters
    service.remove_lora_adapters(["llama3-zh"])
    print(service.get_lora_adapters())
    service.remove_lora_adapters()
    print(service.get_lora_adapters())
    service.load_lora_adapters(
        ["tjluyao/llama-3-8b-math", "tjluyao/llama-3-8b-oaast", "tjluyao/llama-3-8b-zh"]
    )
    # Create an input batch of two queries
    requests = [
        make_input("tjluyao/llama-3-8b-zh", "lora", id=0),
        make_input("tjluyao/llama-3-8b-oaast", "lora", id=1),
        make_input("tjluyao/llama-3-8b-zh", "empty", id=2),
    ]
elif test == "llama-3-70":
    # Load model
    service = FlashinferLlama(
        model_id="TechxGenus/Meta-Llama-3-70B-Instruct-AWQ",
        lora_ids=["Dogge/llama-3-70B-instruct-uncensored-lora"],
        quantize="AWQ",
    )
    # service = FlashinferLlama(model_id="TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ",
    #                        lora_ids=['Dogge/llama-3-70B-instruct-uncensored-lora'], quantize='GPTQ')
    # Create an input batch of two queries
    requests = [make_input("Dogge/llama-3-70B-instruct-uncensored-lora", "lora", id=0)]
elif test == "gemma":
    requests = [
        make_input("tjluyao/gemma-2b-it-math", "lora", id=0),
        make_input("tjluyao/gemma-2b-it-math", "lora", id=1),
        make_input("tjluyao/gemma-2b-it-math", "lora", id=2),
    ]
    service = FlashinferGemma(
        model_id="google/gemma-2b-it",
        lora_ids=[
            "tjluyao/gemma-2b-it-math",
            "monsterapi/gemma-2b-lora-maths-orca-200k",
        ],
    )
    # service = FlashinferGemma(model_id="google/gemma-2b", lora_ids=['tjluyao/gemma-2b-math'])
    # service = FlashinferGemma(model_id="google/gemma-2b", lora_ids=[])
    # Quantized version
    # service = FlashinferGemma(model_id="TechxGenus/gemma-2b-GPTQ", quantize='gptq')
elif test == "mistral":
    requests = [
        make_input(
            "abcdabcd987/gsm8k-llama2-7b-lora-16",
            "base",
            id=0,
            promptOverride="why is deep learning so popular these days?",
        ),
        make_input(
            "abcdabcd987/gsm8k-llama2-7b-lora-16",
            "base",
            id=1,
            promptOverride="What are the differences between Manhattan and Brooklyn",
        ),
    ]
    service = FlashinferMistral(model_id="mistralai/Mistral-7B-v0.3")
elif test == "qwen2":
    requests = [
        make_input(
            "REILX/Qwen1.5-7B-Chat-750Mb-lora",
            "base",
            id=0,
            promptOverride="给我讲个故事",
        ),
        make_input(
            "REILX/Qwen1.5-7B-Chat-750Mb-lora",
            "lora",
            id=1,
            promptOverride="什么是深度学习？",
        ),
    ]

    service = FlashinferQwen2(
        model_id="Qwen/Qwen1.5-7B-Chat", lora_ids=["REILX/Qwen1.5-7B-Chat-750Mb-lora"]
    )
elif test == "qwen2-1.8":
    # Todo: Add qwen1.5 1.8b chat lora adapter / Output Repetition Problem
    requests = [
        make_input(
            "REILX/Qwen1.5-7B-Chat-750Mb-lora",
            "base",
            id=0,
            promptOverride="给我讲个故事",
        )
    ]

    service = FlashinferQwen2(
        model_id="Qwen/Qwen1.5-1.8B-Chat", lora_ids=["REILX/Qwen1.5-7B-Chat-750Mb-lora"]
    )
elif test == "qwen2-70":
    # Todo: Add qwen1.5 72b chat lora adapter
    requests = [
        make_input(
            "REILX/Qwen1.5-7B-Chat-750Mb-lora",
            "base",
            id=0,
            promptOverride="给我讲个故事",
        )
    ]

    service = FlashinferQwen2(
        model_id="Qwen/Qwen1.5-72B-Chat-GPTQ-Int4",
        lora_ids=["REILX/Qwen1.5-7B-Chat-750Mb-lora"],
        quantize="gptq",
    )
elif test == "phi":
    requests = [
        make_input(
            "abcdabcd987/gsm8k-llama2-7b-lora-16",
            "base",
            id=0,
            promptOverride="why is deep learning so popular these days?",
        ),
        make_input(
            "abcdabcd987/gsm8k-llama2-7b-lora-16",
            "base",
            id=1,
            promptOverride="What are the differences between Manhattan and Brooklyn",
        ),
    ]
    service = FlashinferPhi(model_id="microsoft/phi-2")
elif test == "phi3":
    requests = [
        make_input(
            "abcdabcd987/gsm8k-llama2-7b-lora-16",
            "base",
            id=0,
            promptOverride="why is deep learning so popular these days?",
        ),
        make_input(
            "abcdabcd987/gsm8k-llama2-7b-lora-16",
            "base",
            id=1,
            promptOverride="What are the differences between Manhattan and Brooklyn",
        ),
    ]
    service = FlashinferLlama(model_id="microsoft/Phi-3-mini-4k-instruct")
elif test == "baichuan":
    requests = [
        make_input(
            "abcdabcd987/gsm8k-llama2-7b-lora-16",
            "base",
            id=0,
            promptOverride="why is deep learning so popular these days?",
        ),
        make_input(
            "abcdabcd987/gsm8k-llama2-7b-lora-16",
            "base",
            id=1,
            promptOverride="What are the differences between Manhattan and Brooklyn",
        ),
    ]
    service = FlashinferLlama(model_id="baichuan-inc/Baichuan2-7B-Chat")

print(service.get_lora_adapters())
tokenizer = service.tokenizer

batch = generate_pb2.Batch(id=0, requests=requests, size=len(requests))
pb_batch = FlashinferBatch.from_pb(
    batch, tokenizer, torch.float16, torch.device("cuda")
)

# Add input batch to model service
ids = service.add_request(pb_batch)
display_results = {}

# Iterative generation: each step generates a token for each input in the batch
while True:
    # When calling iterative text generation, we may add new inputs (use pb_batch like above)
    # or use an empty batch (use EmptyFlashinferBatch)
    generations, _, _ = service.generate_token(FlashinferBatch.Empty(batch.id))
    # Stop if all input generations are done
    if not generations:
        break
    for gen in generations:
        if gen.request_id in display_results:
            display_results[gen.request_id].append(gen.tokens.texts[0])
        else:
            display_results[gen.request_id] = [gen.tokens.texts[0]]

for id in display_results:
    print(str(id) + "=" * 30)
    print("".join(display_results[id]))
