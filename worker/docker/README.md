# Build CPU image (no vLLM)
docker build -f docker/Dockerfile.cpu -t yourrepo/mloc_worker:cpu-latest .

# Build CUDA image (with vLLM)
docker build -f docker/Dockerfile.cuda -t yourrepo/mloc_worker:cuda-latest .

# Run (CPU)
docker run --rm -p 8000:8000 yourrepo/mloc_worker:cpu-latest

# Run (GPU; host must have NVIDIA Container Toolkit)
docker run --rm -p 8000:8000 --gpus all yourrepo/mloc_worker:cuda-latest

## docker-compose with NFS shared results

Edit `worker/docker-compose.yml` and set the following environment variables before
running `docker compose up` (for example via an `.env` file in the same directory):

```
NFS_SERVER=10.0.0.10          # NFS server hostname or IP (defaults to 127.0.0.1)
NFS_EXPORT_PATH=/srv/mloc/results
NFS_VERSION=4                 # optional, defaults to 4
```

The compose file mounts the shared export at `/mnt/mloc-results` inside the
worker container and sets `RESULTS_DIR` accordingly so that all task outputs are
stored on the NFS share.
