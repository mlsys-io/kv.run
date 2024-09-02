from collections import defaultdict
from typing import Optional
from text_generation_server.cli import download_weights
from text_generation_server.models_flashinfer.flashinfer_gemma import FlashinferGemma
from text_generation_server.pb import generate_pb2


def test_gemma_text_generation():
    download_weights(model_id="google/gemma-2b-it")
    requests = [
        _gen_standard_input(None, id=0, prompt="Deep learning "),
        _gen_standard_input("tjluyao/gemma-2b-it-math", id=1, prompt="What is 1+"),
        _gen_standard_input(
            "tjluyao/gemma-2b-it-math", id=2, prompt="Solve the equation x+1="
        ),
    ]
    batch = generate_pb2.Batch(id=0, requests=requests, size=len(requests))
    service = FlashinferGemma(
        model_id="google/gemma-2b-it",
        lora_ids=["tjluyao/gemma-2b-it-math"],
    )
    service.warmup(batch)

    requests = [
        _gen_standard_input(None, id=0, prompt="Deep learning "),
        _gen_standard_input("tjluyao/gemma-2b-it-math", id=1, prompt="What is 1"),
        _gen_standard_input(
            "tjluyao/gemma-2b-it-math", id=2, prompt="Solve the equation x+1="
        ),
    ]
    batch = generate_pb2.Batch(id=2, requests=requests, size=len(requests))

    isPrefill = True
    text_generation = defaultdict(lambda: [])
    while True:
        if isPrefill:
            generations, next_batch, _ = service.prefill_batch(batch)
            isPrefill = False
        else:
            generations, next_batch, _ = service.decode_batch([next_batch.to_pb()])

        for gen in generations:
            if gen.generated_text:
                text_generation[gen.request_id] = gen.generated_text.text
        if all([g.generated_text for g in generations]):
            break

    expected_text_generation = {
        1: "000 ÷ 100?\n\n1000 ÷ 100 = 10.<eos>",
        0: "\n\n## Deep Learning: A Powerful Tool for Data Analysis\n\nDeep learning is a subfield of machine learning (ML) that allows computers to learn from data without",
        2: "5\n\nStep 1: Subtract 1 from both sides of the equation.\n\nx + 1 - 1 = 5 - 1\nx",
    }

    assert text_generation == expected_text_generation


def _gen_standard_input(lora_id: Optional[str], id: int, prompt: str):
    request = generate_pb2.Request(
        id=id,
        inputs=prompt,
        truncate=256,
        prefill_logprobs=True,
        top_n_tokens=20,
        parameters=generate_pb2.NextTokenChooserParameters(
            temperature=0.9,
            top_k=10,
            top_p=0.9,
            typical_p=0.9,
            repetition_penalty=1.1,
            seed=1000,
        ),
        stopping_parameters=generate_pb2.StoppingCriteriaParameters(
            max_new_tokens=32, stop_sequences=[], ignore_eos_token=True
        ),
        lora_id=lora_id,
    )
    return request
