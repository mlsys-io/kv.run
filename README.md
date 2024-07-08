# kv.run
(Limited) comparison of popular model serving solutions

| Solution        | Inference backend | Serving backend      | Advanced kernel support                                                                          | Model support               |
|-----------------|-------------------|----------------------|--------------------------------------------------------------------------------------------------|-----------------------------|
| Huggingface TGI | Pytorch           | HF TGI (Rust)        | Paged + Flash attention                                                                          | Language                    |
| Deepspeed MII   | PyTorch           | Deepspeed (Python)   | [DeepSpeed-Kernels](https://github.com/microsoft/DeepSpeed-Kernels)                              | Language                    |
| TensorRT-LLM    | TensorRT-LLM      | TensorRT-LLM (C++)   | [TensorRT XQA](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/XQA-kernel.md) | Language                    |
| vLLM            | vLLM              | vLLM (Python)        | Paged + Flash attention                                                                          | Language                    |
| kv.run          | PyTorch           | HF TGI + more (Rust) | Paged + Flash attention, [FlashInfer](https://github.com/flashinfer-ai/flashinfer)               | Language, diffusion (soon) |



## Installation

#### Install [Rust](https://rustup.rs/):
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
#### Install Protobuf
```shell
sudo apt-get install libssl-dev gcc -y
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

#### Install Kernel Libraries
```shell
# Install FlashInfer
# For CUDA 12.1 & torch 2.3
pip install flashinfer==0.0.8 -i https://flashinfer.ai/whl/cu121/torch2.3
# For other CUDA & torch versions, please check https://docs.flashinfer.ai/installation.html

# Install Flash and Paged Attention
cd server && make install-flash-attention && make install-vllm-cuda
```

#### Build Code Base
```shell
make install
```

#### Build Docker Image (optional)
`Dockerfile_kvrun` provides a docker image building script. We will provide pre-built docker images shortly.

## Usages
#### Local API tests
```shell
cd server/examples && python test_local_api.py
```
#### Local UI demo
(Inherited from [Punica](https://github.com/punica-ai/punica))
```shell
python server/examples/test_ui.py
```
[demo.mp4](https://github.com/mlsys-io/kv.run/assets/12567967/977b09fb-bd90-4757-85ab-e5fc2a58cd93)

#### Deploy services
```shell
text-generation-launcher --model-id tjluyao/llama-3-8b
```
#### Using quantized models
Add --quantize [Method] to the command above, for example:
```shell
text-generation-launcher --model-id TechxGenus/gemma-2b-GPTQ --lora-ids tjluyao/gemma-2b-it-math --quantize gptq
```
The supported quantization methods include:
- AWQ: 4-bit. Need specific quantized model.
- EETQ: 8-bit. Can work for any model.
- GPTQ: 4-bit. Need specific quantized model.
- Bitandbytes: 8-bit. Can work for any model.

For AWQ and EETQ quantization, you need to build their specific kernels:
```shell
# AWQ
cd server && make install-awq
git clone https://github.com/casper-hansen/AutoAWQ && cd AutoAWQ
pip install -e .
# EETQ
cd server && make install-eetq
# GTPQ
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install -vvv --no-build-isolation -e .
```

#### Multi-LoRA support
- To load LoRA adapters, you may either (1) specify in the laucher argument using `lora-ids`:
```shell
text-generation-launcher --model-id tjluyao/llama-3-8b --lora-ids tjluyao/llama-3-8b-math;tjluyao/llama-3-8b-zh
```
Or, loading dynamically by the client after the model is launched:
```shell
curl 127.0.0.1:3000/download_lora_adapter -X POST -d '{"lora_id":"tjluyao/llama-3-8b-math"}' -H 'Content-Type: application/json'
```
- To query the model, similarly you can use `lora-id` in the parameters (make sure the adapter is loaded):
```shell
curl 127.0.0.1:3000/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"lora_id": "tjluyao/llama-3-8b-math", "max_new_tokens":20}}' -H 'Content-Type: application/json'
```

## Benchmarks
Testing Llama-2-7b on RTX 6000 ada:

| Step    | Batch Size | Average FlashInfer  | Average TGI         |
|---------|------------|---------------------|---------------------|
| Prefill | 1          | 52.16 tokens/secs   | 41.14 tokens/secs   |
|         | 2          | 101.64 tokens/secs  | 78.69 tokens/secs  |
|         | 4          | 191.48 tokens/secs  | 154.11 tokens/secs  |
|         | 8          | 323.21 tokens/secs  | 290.82 tokens/secs  |
|         | 16         | 512.50 tokens/secs  | 538.15 tokens/secs  |
|         | 32         | 697.89 tokens/secs  | 783.61 tokens/secs  |
| Decode  | 1          | 56.55 tokens/secs   | 40.84 tokens/secs   |
|         | 2          | 108.55 tokens/secs  | 77.85 tokens/secs  |
|         | 4          | 207.10 tokens/secs  | 154.27 tokens/secs  |
|         | 8          | 383.92 tokens/secs  | 297.53 tokens/secs  |
|         | 16         | 682.78 tokens/secs  | 562.83 tokens/secs  |
|         | 32         | 1119.92 tokens/secs | 993.33 tokens/secs |

## Model and kernel support matrix
Note: L = Language, I = Image

| Model                                                                        | MOE | Size  | Modality | Flash & Page Attention | FlashInfer |
|------------------------------------------------------------------------------|-----|-------|----------|---------------------|------------|
| [Idefics](https://huggingface.co/HuggingFaceM4/idefics-9b)                   |     | 9B    | L, I ⇒ L | ✔                   |            |
| [Idefics 2](https://huggingface.co/HuggingFaceM4/idefics2-8b)                |     | 8B    | L, I ⇒ L | ✔                   |            |
| [Llava Next (1.6)](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) |     | 13B   | L, I ⇒ L | ✔                   |            |
| [Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-hf)                   |     | 7B    | L ⇒ L   | ✔                   | ✔          |
| [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)        |     | 8B    | L ⇒ L   | ✔                   | ✔          |
| [Phi 1.5](https://huggingface.co/microsoft/phi-2)                            |     | 2.7B  | L ⇒ L   | ✔                   |            |
| [Phi 3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)             |     | 3.8B  | L ⇒ L   | ✔                   | ✔          |
| [Gemma](https://huggingface.co/google/gemma-2b)                              |     | 2B    | L ⇒ L   | ✔                   | ✔          |
| [Cohere](https://huggingface.co/CohereForAI/c4ai-command-r-plus)             |     | 104B  | L ⇒ L   | ✔                   |            |
| [Dbrx](https://huggingface.co/databricks/dbrx-instruct)                      | ✔   | 132B  | L ⇒ L   | ✔                   |            |
| [Mamba](https://huggingface.co/state-spaces/mamba-2.8b-slimpj)               |     | 2.8B  | L ⇒ L   |                     |            |
| [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)         |     | 7B    | L ⇒ L   | ✔                   | ✔          |
| [Mixtral](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)      | ✔   | 8x22B | L ⇒ L   | ✔                   |            |
| [Gpt Bigcode](https://huggingface.co/bigcode/gpt_bigcode-santacoder)         |     | 1.1B  | L ⇒ L   | ✔                   |            |
| [Baichuan](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)            |     | 7B    | L ⇒ L   | ✔                   | ✔          |
| [Falcon](https://huggingface.co/tiiuae/falcon-7b-instruct)                   |     | 7B    | L ⇒ L   | ✔                   |            |
| [StarCoder 2](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)   |     | 15B   | L ⇒ L   | ✔                   |            |
| [Qwen 2](https://huggingface.co/Qwen/Qwen2-7B-Instruct)                      |     | 7B    | L ⇒ L   | ✔                   | ✔          |
| [Qwen 1.5](https://huggingface.co/Qwen/Qwen1.5-7B-Chat)                      |     | 7B    | L ⇒ L   | ✔                   | ✔          |
| [Opt](https://huggingface.co/facebook/opt-6.7b)                              |     | 6.7B  | L ⇒ L   |                     |            |
| [T5](https://huggingface.co/google-t5/t5-11b)                                |     | 11B   | L ⇒ L   |                     |            |
| [Galactica](https://huggingface.co/facebook/galactica-120b)                  |     | 120B  | L ⇒ L   |                     |            |
| [SantaCoder](https://huggingface.co/bigcode/santacoder)                      |     | 1.1B  | L ⇒ L   | ✔                   |            |
| [Bloom](https://huggingface.co/bigscience/bloom-560m)                        |     | 560M  | L ⇒ L   |                     |            |
| [Mpt](https://huggingface.co/mosaicml/mpt-7b-instruct)                       |     | 7B    | L ⇒ L   |                     |            |
| [Gpt2](https://huggingface.co/openai-community/gpt2)                         |     | 124M  | L ⇒ L   | ✔                   |            |
| [Gpt Neox](https://huggingface.co/EleutherAI/gpt-neox-20b)                   |     | 20B   | L ⇒ L   | ✔                   |            |
| [Yi 1.5](https://huggingface.co/01-ai/Yi-1.5-9B-Chat)                        |     | 9B    | L ⇒ L   | ✔                   |         ✔  |
| [ChatGLM 4](https://huggingface.co/THUDM/glm-4-9b-chat)                      |     | 9B    | L ⇒ L   | ✔                   | ✔          |
