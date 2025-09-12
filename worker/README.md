1. Build the docker image
```bash
docker build -f Dockerfile.gpu -t mloc_worker:cuda12.6-py312 --build-arg TZ=Asia/Singapore .
```
2. Run the docker container
```bash
docker compose --compatibility up -d
```
