from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.models_flashinfer.flashinfer_llava import LlavaLM, LlavaBatch
import random, torch
import base64

model = LlavaLM(model_id="liuhaotian/llava-v1.5-7b")
tokenizer = model.language_model.tokenizer
processor = model.vision_model.image_processor

prompts = [
    'How many people are in the image?',
    'What is the main object in the image?',
    'What is the mood of the image?',
    'What is the setting of the image?',
    'What is the image about?',
    'What is this a picture of?',
]

# read image from local file
image_path = "./build/server/examples/test.jpg"
with open(image_path, "rb") as f:
    image = base64.b64encode(f.read()).decode("utf-8")

image = f"data:image/png;base64,{image}"

def make_input(image, id = 0):
    prompt = random.choice(prompts)
    request = generate_pb2.Request(
        id=id,
        inputs=f"![]({image}){prompt}\n\n",
        truncate=1024,
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
pb_batch = LlavaBatch.from_pb_processor(batch, tokenizer, processor, model.language_model.model_config, torch.float16, torch.device("cuda"))
results = []
while len(pb_batch.requests) > 0:
    generations, pb_batch, _ = model.generate(pb_batch)
    for gen in generations:
        if gen.generated_text is not None:
            results.append(gen.generated_text.text)

for i, result in enumerate(results):
    print(f"Request {i}: {result}\n")