# Worker Guide

Workers subscribe to the orchestrator's Redis channel, execute tasks with the
appropriate executor, and report back success/failure together with optional
artifacts (for example, fine-tuned checkpoints).

## Environment variables
| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | – | Redis connection string (mandatory). |
| `RESULTS_DIR` | `./results_workers` | Local/shared directory where each task writes `responses.json`, checkpoints, and archives. |
| `TASK_TOPIC` | `tasks` | Redis pub/sub topic to listen on. |
| `HEARTBEAT_INTERVAL_SEC` | `30` | Interval between heartbeats. |
| `WORKER_ID` | random | Override to keep a stable worker identifier. |
| `WORKER_TAGS` | empty | Comma-separated tags used by the scheduler. |
| `LOG_LEVEL` | `INFO` | Worker log level. |
| `ORCHESTRATOR_BASE_URL` | empty | Optional base URL (e.g. `http://127.0.0.1:8000`) used to construct artifact download links when templates specify `checkpoint.load` blocks without an explicit URL. |

> When `spec.output.destination.type` is `http`, the worker uploads generated
> model archives back to the orchestrator using `POST /api/v1/results/{task_id}/files`.
> Set `ORCHESTRATOR_BASE_URL` so staged pipelines can later download those files
> through the orchestrator.

## Startup
```bash
export REDIS_URL="redis://localhost:6379/0"
export RESULTS_DIR=./results_workers
export ORCHESTRATOR_BASE_URL="http://127.0.0.1:8000"
python worker/main.py
```

On startup the worker:
1. Collects hardware information and registers with the orchestrator.
2. Sends periodic heartbeats with load metrics.
3. Listens for tasks, selects the correct executor, and writes results to
   `RESULTS_DIR/<task_id>/`.
4. When an HTTP destination is configured, archives `final_model/` or
   `final_lora/` into a `.tar.gz` file and uploads it to the orchestrator, recording
   the publicly retrievable URL in the task result.

> ℹ️ The hardware probe now records per-GPU VRAM by shelling out to
> `nvidia-smi`. This allows templates that specify
> `spec.resources.hardware.gpu.memory` to be matched against the actual device
> memory. Ensure the NVIDIA utilities are on `$PATH` (they are present in the
> CUDA worker image by default).

## Output directories
- Each task receives a dedicated directory inside `RESULTS_DIR`.
- Executors write their JSON summary to `<task_id>/responses.json`.
- Training executors place checkpoints and, when enabled, automatically
  generate `final_model.tar.gz` or `final_lora.tar.gz` next to the directory.

## HTTP artifact workflow
1. Stage 1 runs with `spec.output.destination.type: http` and uploads the archive.
2. The worker stores the archive locally and pushes it to the orchestrator.
3. The orchestrator exposes the archive at
   `GET /api/v1/results/{task_id}/files/<archive>`.
4. Stage 2 templates reference the link via
   `${stage.result.final_model_archive_url}` (or LoRA equivalent) inside
   `checkpoint.load.url`.

If you prefer shared storage, point both the orchestrator and workers at the
same filesystem mount and disable HTTP outputs; the executors still write their
local artifacts, and templates can reference them via absolute paths instead of
`checkpoint.load`.

## Debugging tips
- Check `worker.log` for executor output and upload diagnostics.
- Use `redis-cli` to inspect `worker:<worker_id>` hashes and
  `worker:<worker_id>:hb` TTLs.
- When pipelines stall, ensure the Stage 1 task produced the expected
  `final_*_archive_url` and the orchestrator's results directory contains the
  uploaded `.tar.gz` file.


## Elastic Scaling & Task Merge
- **Elastic Scaling** is governed by `ENABLE_ELASTIC_SCALING` (default `true`). When enabled, the orchestrator may disable idle workers and re-enable them when backlog grows. Workers do not need extra configuration—they continue heartbeating and are redispatched once re-enabled.
- Optional tuning knobs live on the host (`ELASTIC_AUTO_DISABLE_IDLE_SEC`, `ELASTIC_AUTO_ENABLE_QUEUE_THRESHOLD`, `ELASTIC_AUTO_DISABLE_QUEUE_MAX`, `ELASTIC_AUTO_MIN_ACTIVE_WORKERS`, etc.).
- **Task Merge**: the orchestrator can coalesce identical inference/RAG tasks. Workers receive a single parent payload with `merge_children`, and executors (e.g. vLLM) emit per-child results under `result.children` which the runner writes to each child directory. No worker-side toggle is required; ensure custom executors respect the `merge_children` contract if you add your own.
- **Multi-GPU training / inference**: when `CUDA_VISIBLE_DEVICES` (or detected GPUs) exposes more than one device, the vLLM executor automatically sets `tensor_parallel_size`, and the PPO/DPO/SFT executors will launch distributed jobs via `torchrun` unless `training.allow_multi_gpu` is explicitly set to `false`. Override `training.nproc_per_node` to constrain world size.
- **Bandwidth throttling**: set `WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC` to simulate capped HTTP throughput; the worker reports the value back to the host and inserts delays before delivering HTTP callbacks.
