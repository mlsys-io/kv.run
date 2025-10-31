# Build CPU image (no vLLM)
docker build -f worker/docker/Dockerfile.cpu -t yourrepo/flowmesh_worker:cpu-latest .

# Build CUDA image (installs vLLM + GPU extras)
docker build -f worker/docker/Dockerfile.cuda -t yourrepo/flowmesh_worker:cuda-latest .

# Run (CPU)
docker run --rm \
  -e REDIS_URL="redis://host.docker.internal:6379/0" \
  -e RESULTS_DIR=/app/results \
  -v "$(pwd)/results_cpu:/app/results" \
  yourrepo/flowmesh_worker:cpu-latest

# Run (GPU; host must have NVIDIA Container Toolkit)
docker run --rm --gpus all \
  -e REDIS_URL="redis://host.docker.internal:6379/0" \
  -e RESULTS_DIR=/app/results \
  -v "$(pwd)/results_gpu:/app/results" \
  yourrepo/flowmesh_worker:cuda-latest

## docker-compose with NFS shared results

Edit `worker/docker-compose.yml` and set the following environment variables before
running `docker compose up` (for example via an `.env` file in the same directory):

```
NFS_SERVER=10.0.0.10          # NFS server hostname or IP (defaults to 127.0.0.1)
NFS_EXPORT_PATH=/srv/flowmesh/results
NFS_VERSION=4                 # optional, defaults to 4
```

The compose file mounts the shared export at `/mnt/flowmesh-results` inside the
worker container and sets `RESULTS_DIR` accordingly so that all task outputs are
stored on the NFS share.
