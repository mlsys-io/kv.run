# Build CPU image (no vLLM)
docker build -f docker/Dockerfile.cpu -t yourrepo/mloc_worker:cpu-latest .

# Build CUDA image (with vLLM)
docker build -f docker/Dockerfile.cuda -t yourrepo/mloc_worker:cuda-latest .

# Run (CPU)
docker run --rm -p 8000:8000 yourrepo/mloc_worker:cpu-latest

# Run (GPU; host must have NVIDIA Container Toolkit)
docker run --rm -p 8000:8000 --gpus all yourrepo/mloc_worker:cuda-latest
