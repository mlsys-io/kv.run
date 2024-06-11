# kv.run
(Limited) comparison of popular model serving solutions

| Solution        | Inference backend | Serving backend      | Advanced kernel support                                                                          | Model support              |
|-----------------|-------------------|----------------------|--------------------------------------------------------------------------------------------------|----------------------------|
| Huggingface TGI | Pytorch           | HF TGI (Rust)        | Paged + Flash attention                                                                          | Language                   |
| Deepspeed MII   | PyTorch           | Deepspeed (Python)   | [DeepSpeed-Kernels](https://github.com/microsoft/DeepSpeed-Kernels)                              | Language                   |
| TensorRT-LLM    | TensorRT-LLM      | TensorRT-LLM (C++)   | [TensorRT XQA](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/XQA-kernel.md) | Language                   |
| vLLM            | vLLM              | vLLM (Python)        | Paged + Flash attention                                                                          | Language                   |
| kv.run          | PyTorch           | HF TGI + more (Rust) | Paged + Flash attention, [FlashInfer](https://github.com/flashinfer-ai/flashinfer)               | Language, diffusion (exp.) |



## Installation
#### Sync submodules
```shell
git submodule sync
git submodule update --init
```

#### Install Rust
[Script](https://rustup.rs/):
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

#### Build Code Base
```shell
make install
```

#### Install Kernel Libraries
```shell
# Install FlashInfer
# For CUDA 12.1 & torch 2.3
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3
# For other CUDA & torch versions, please check https://docs.flashinfer.ai/installation.html

# Install Flash and Paged Attention
cd server
make install-flash-attention && make install-vllm-cuda
```
You can debug/edit code in the build folder. When done, use python copy_back.py to copy changes back to the original src folder.

## Usages
#### Local API tests
```shell
cd server/examples && python test_local_api.py
```
#### Local UI demo
(Inherited from [Punica](https://github.com/punica-ai/punica))

[demo.mp4](https://github.com/mlsys-io/kv.run/assets/12567967/977b09fb-bd90-4757-85ab-e5fc2a58cd93)

#### Deploy services
```shell
text-generation-launcher --model-id tjluyao/llama-3-8b --lora-ids tjluyao/llama-3-8b-math;tjluyao/llama-3-8b-zh
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

## Model support matrix
Note: L = Language, I = Image

| Model                                                                        | MOE  | Size  | Modality | Quantization | Tensor Parallelism | FlashInfer | Multi-LoRA |
|------------------------------------------------------------------------------|------|-------|----------|--------------|--------------------|------------|------------|
| [Idefics](https://huggingface.co/HuggingFaceM4/idefics-9b)                   |     | 9B    | L, I ⇒ L |              |                    |            |            |
| [Idefics 2](https://huggingface.co/HuggingFaceM4/idefics2-8b)                |     | 8B    | L, I ⇒ L |              |                    |            |            |
| [Llava Next (1.6)](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) |     | 13B   | L, I ⇒ L |              |                    |            |            |
| [Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-hf)                   |     | 7B    | L ⇒ L    |              |                    | ✔          | ✔          |
| [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)        |     | 8B    | L ⇒ L    |              |                    | ✔          | ✔          |
| [Phi 1.5](https://huggingface.co/microsoft/phi-1_5)                          |     | 1.3B  | L ⇒ L    |              |                    |            |            |
| [Phi 3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)             |     | 3.8B  | L ⇒ L    |              |                    |            |            |
| [Gemma](https://huggingface.co/google/gemma-2b)                              |     | 2B    | L ⇒ L    |              |                    | ✔          | ✔          |
| [Cohere](https://huggingface.co/CohereForAI/c4ai-command-r-plus)             |     | 104B  | L ⇒ L    |              |                    |            |            |
| [Dbrx](https://huggingface.co/databricks/dbrx-instruct)                      | ✔    | 132B  | L ⇒ L   |              |                    |            |            |
| [Mamba](https://huggingface.co/state-spaces/mamba-2.8b-slimpj)               |     | 2.8B  | L ⇒ L    |              |                    |            |            |
| [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)         |     | 7B    | L ⇒ L    |              |                    |            |            |
| [Mixtral](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)      | ✔    | 8x22B | L ⇒ L   |              |                    |            |            |
| [Gpt Bigcode](https://huggingface.co/bigcode/gpt_bigcode-santacoder)         |     | 1.1B  | L ⇒ L    |              |                    |            |            |
| [Baichuan](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)            |     | 7B    | L ⇒ L    |              |                    |            |            |
| [Falcon](https://huggingface.co/tiiuae/falcon-7b-instruct)                   |     | 7B    | L ⇒ L    |              | ✔                  |            |            |
| [StarCoder 2](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)   |     | 15B   | L ⇒ L    |              |                    |            |            |
| [Qwen 2](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)        |     | 15B   | L ⇒ L    |              |                    |            |            |
| [Opt](https://huggingface.co/facebook/opt-6.7b)                              |     | 6.7B  | L ⇒ L    |              |                    |            |            |
| [T5](https://huggingface.co/google-t5/t5-11b)                                |     | 11B   | L ⇒ L    |              |                    |            |            |
| [Galactica](https://huggingface.co/facebook/galactica-120b)                  |     | 120B  | L ⇒ L    |              |                    |            |            |
| [SantaCoder](https://huggingface.co/bigcode/santacoder)                      |     | 1.1B  | L ⇒ L    |              |                    |            |            |
| [Bloom](https://huggingface.co/bigscience/bloom-560m)                        |     | 560M  | L ⇒ L    |              |                    |            |            |
| [Mpt](https://huggingface.co/mosaicml/mpt-7b-instruct)                       |     | 7B    | L ⇒ L    |              |                    |            |            |
| [Gpt2](https://huggingface.co/openai-community/gpt2)                         |     | 124M  | L ⇒ L    |              |                    |            |            |
| [Gpt Neox](https://huggingface.co/EleutherAI/gpt-neox-20b)                   |     | 20B   | L ⇒ L    |              | ✔                  |            |            |
