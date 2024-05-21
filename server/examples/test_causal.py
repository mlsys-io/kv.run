from text_generation_server.pb import generate_pb2
import torch
from text_generation_server.models.causal_lm import CausalLM, CausalLMBatch
import random, json
from test_cases import DEMO, LoraSpec

service = CausalLM("tjluyao/llama-3-8b")

tokenizer = service.tokenizer

# Create input requests
def make_input(id=0, promptOverride=None):
    request = generate_pb2.Request(
        id=id,
        inputs=promptOverride,
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
            max_new_tokens=64,
            stop_sequences=[],
            ignore_eos_token=True))
    return request

# Create an input batch of two queries
requests = [make_input(id=1, promptOverride="What is deep learning?")]
batch = generate_pb2.Batch(id = 0, requests = requests, size = len(requests))
pb_batch = CausalLMBatch.from_pb(batch, tokenizer, torch.bfloat16, torch.device("cuda"))

# Iterative generation: each step generates a token for each input in the batch
while True:
    generations, pb_batch, _ = service.generate_token(pb_batch)
    # Stop if all input generations are done
    if generations[0].generated_text:
        print(generations[0].generated_text.text)
        break
