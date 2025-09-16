1. Build the docker image
```bash
docker build -f Dockerfile.gpu -t mloc_worker:cuda12.6-py312 --build-arg TZ=Asia/Singapore .
```
2. Run the docker container
```bash
# Auto-detect (cuda if GPU available, otherwise cpu), pull and run
./mloc.sh

# Force CPU version
DEVICE=cpu ./mloc.sh

# Pull only (no run)
PULL_ONLY=1 ./mloc.sh

# Pass extra docker args
RUNTIME_ARGS="-p 8000:8000 -e LOG_LEVEL=DEBUG" ./mloc.sh

```
