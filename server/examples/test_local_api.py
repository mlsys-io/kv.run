from text_generation_server.pb import generate_pb2
import torch
from text_generation_server.models.flashinfer_causal_lm import FlashinferLM, FlashinferBatch
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
    inputs = json.dumps({"inputs": prompt, "lora_id": lora_id})

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
            max_new_tokens=2048,
            stop_sequences=[],
            ignore_eos_token=True))
    return request

test = 'llama-3'
#test = 'llama-2'

if test == 'llama-2':
    # Load model
    service = FlashinferLM(model_id="meta-llama/Llama-2-7b-hf",
              lora_ids={'llama2-gsm8k':'abcdabcd987/gsm8k-llama2-7b-lora-16'})
    # Create an input batch of two queries
    requests = [make_input('llama2-gsm8k', 'base', id=0), make_input('llama2-gsm8k', 'lora', id=1)]
elif test == 'llama-3':
    # Load model
    service = FlashinferLM(model_id="tjluyao/llama-3-8b",
              lora_ids={'llama3-math':'tjluyao/llama-3-8b-math',
                        'llama3-zh': 'tjluyao/llama-3-8b-zh'})
    # Test load lora adapters
    print(service.get_lora_adapters())
    # Test remove lora adapters
    service.remove_lora_adapters(['llama3-zh'])
    print(service.get_lora_adapters())
    service.remove_lora_adapters()
    print(service.get_lora_adapters())
    service.load_lora_adapters({'llama3-math':'tjluyao/llama-3-8b-math',
                                'llama3-oaast': 'tjluyao/llama-3-8b-oaast',
                                'llama3-zh': 'tjluyao/llama-3-8b-zh'})
    # Create an input batch of two queries
    requests = [make_input('llama3-zh', 'lora', id=0), make_input('llama3-oaast', 'lora', id=1)]


print(service.get_lora_adapters())
tokenizer = service.tokenizer

batch = generate_pb2.Batch(id = 0, requests = requests, size = len(requests))
pb_batch = FlashinferBatch.from_pb(batch, tokenizer, torch.float16, torch.device("cuda"))

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
    print(str(id) + '='*30)
    print(''.join(display_results[id]))
