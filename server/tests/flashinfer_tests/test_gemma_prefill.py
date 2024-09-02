from typing import Optional
from text_generation_server.cli import download_weights
from text_generation_server.models_flashinfer.flashinfer_gemma import FlashinferGemma
from text_generation_server.pb import generate_pb2
import torch


def test_gemma_prefill():
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

    generations, next_batch, debug_info = service.prefill_batch(batch, debug_mode=True)

    softmax = torch.exp(debug_info.all_log_probs.cpu())
    top_10_values, top_10_indices = torch.topk(softmax, k=10, dim=1)

    top_10_indices_correct = torch.tensor(
        [
            [109, 71035, 110, 108, 54550, 7, 3, 8, 1, 4],
            [235276, 963, 235284, 8, 7, 3, 1, 4, 9, 5],
            [235308, 235304, 235310, 235274, 235324, 7, 3, 8, 1, 4],
        ]
    )

    top_10_values_correct = torch.tensor(
        [
            [
                0.4609,
                0.1729,
                0.1523,
                0.1104,
                0.1040,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                0.8672,
                0.0757,
                0.0591,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            [
                0.5078,
                0.1455,
                0.1455,
                0.1069,
                0.0942,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
        ],
        dtype=torch.bfloat16,
    )

    assert torch.all(top_10_indices == top_10_indices_correct)
    torch.testing.assert_close(
        top_10_values, top_10_values_correct, rtol=1e-4, atol=1e-4
    )


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
            max_new_tokens=2048, stop_sequences=[], ignore_eos_token=True
        ),
        lora_id=lora_id,
    )
    return request
