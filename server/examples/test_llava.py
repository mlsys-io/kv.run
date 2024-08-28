from text_generation_server.pb import generate_pb2
from text_generation_server.models_flashinfer.flashinfer_llava import LlavaLM, LlavaBatch
import random, torch
import base64
from collections import defaultdict

service = LlavaLM(model_id="llava-hf/llava-v1.6-vicuna-7b-hf")
tokenizer = service.language_model.tokenizer

prompts = [
    'How many people are in the image?',
    'What is the main object in the image?',
    'What is the mood of the image?',
    'What is the setting of the image?',
    'What is the image about?',
    'What is this a picture of?',
]

# read image from local file
image_path = "server/examples/test.jpg"
with open(image_path, "rb") as f:
    image = base64.b64encode(f.read()).decode("utf-8")

image = f"data:image/png;base64,{image}"

def make_input(image, id = 0):
    prompt = random.choice(prompts)
    request = generate_pb2.Request(
        id=id,
        inputs=f"![]({image}){prompt}\n\n",
        #inputs=f"{prompt}\n\n",
        truncate=4096,
        prefill_logprobs=True,
        top_n_tokens=20,
        parameters=generate_pb2.NextTokenChooserParameters(
            temperature=0.1,
            top_k=10,
            top_p=0.9,
            typical_p=0.9,
            repetition_penalty=1.1,
        ),
        stopping_parameters=generate_pb2.StoppingCriteriaParameters(
            max_new_tokens=128,
            stop_sequences=[],
            ignore_eos_token=True),
    )
    return request

requests = [make_input(image, i) for i in range(1)]
batch = generate_pb2.Batch(id = 0, requests = requests, size = len(requests))
display_results = defaultdict(lambda: [])

# Iterative generation: each step generates a token for each input in the batch
isPrefill = True
while True:
    if isPrefill:
        generations, next_batch, _ = service.prefill_batch(batch)
        isPrefill = False
        print(generations[0].tokens.texts)
    else:
        generations, next_batch, _, _ = service.decode_batch([next_batch.to_pb()])
        print(generations[0].tokens.texts)

    for gen in generations:
        if gen.prefill_tokens:
            display_results[gen.request_id] = [
                "Prompt:\n"
                + tokenizer.decode(gen.prefill_tokens.token_ids)
                + "\nAnswer:\n"
            ]
        if gen.generated_text:
            display_results[gen.request_id] += [gen.generated_text.text]
    # Stop if all input generations are done
    if all([g.generated_text for g in generations]):
        break

for id in display_results:
    print(str(id) + "=" * 30)
    print("".join(display_results[id]))