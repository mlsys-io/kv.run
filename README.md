# FlowMesh Worker

FlowMesh workers subscribe to Redis topics published by the orchestrator, execute
the requested task using the appropriate executor (Transformers, TRL, vLLM,
RAG, agent pipelines, and more), and persist results plus optional artifacts to
either shared storage or the orchestrator via HTTP callbacks.

## Quick Start (local)
### 1. Install dependencies with uv
```bash
# Install uv if it is not already available
pip install uv

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# Sync the worker runtime (baseline inference stack)
uv sync --extra inference

# Optional executors:
# uv sync --extra inference --extra training   # enable PPO/DPO/SFT trainers
# uv sync --extra inference --extra rag        # enable RAG executor
# uv sync --extra inference --extra agent      # enable Agent pipelines
# uv sync --all-extras                         # install every optional component
```

### 2. Launch the worker
```bash
export REDIS_URL="redis://localhost:6379/0"
export RESULTS_DIR=./results_workers
export ORCHESTRATOR_BASE_URL="http://127.0.0.1:8000"  # required for HTTP artifact uploads
uv run python worker/main.py
```
At startup the worker:
1. Collects hardware information and registers itself in Redis.
2. Streams heartbeats that include load and power metrics.
3. Listens for tasks, selects the right executor, and writes outputs under
   `RESULTS_DIR/<task_id>/`.
4. Archives `final_model/` or `final_lora/` and uploads the bundle when the task
   requests HTTP artifact delivery.

## Environment variables
| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | – | Redis connection string (required). |
| `RESULTS_DIR` | `./results_workers` | Root directory for task outputs. |
| `TASK_TOPIC` | `tasks` | Redis pub/sub topic the worker subscribes to. |
| `HEARTBEAT_INTERVAL_SEC` | `30` | Interval between heartbeats. |
| `WORKER_ID` | random | Override to pin a stable worker identifier. |
| `WORKER_TAGS` | empty | Comma-separated tags used by the scheduler. |
| `LOG_LEVEL` | `INFO` | Worker log level. |
| `WORKER_COST_PER_HOUR` | `1.0` | Hourly cost in USD; reported with heartbeats. |
| `ORCHESTRATOR_BASE_URL` | empty | Required to build artifact download links when using HTTP results. |
| `MODEL_ARCHIVE_USE_PIGZ` | `1` | Enable multithreaded `pigz` compression (set `0`/`false` to disable). |
| `MODEL_ARCHIVE_COMPRESSION_LEVEL` | `6` | Gzip compression level (`0-9`). |
| `MODEL_ARCHIVE_PIGZ_THREADS` | – | Force a specific thread count for `pigz`; defaults to all CPUs. |
| `MODEL_ARCHIVE_PIGZ_BIN` | `pigz` | Path to the `pigz` binary. |
| `MODEL_ARCHIVE_TAR_BIN` | `tar` | Tar executable used before compression. |
| `WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC` | empty | Throttle HTTP uploads to emulate limited bandwidth. |

> The heartbeat TTL is computed automatically as `max(HEARTBEAT_INTERVAL_SEC * 4, 120)`.

> When a task sets `spec.output.destination.type: http`, configure
> `ORCHESTRATOR_BASE_URL` so the worker can upload artifacts to
> `POST /api/v1/results/{task_id}/files` and expose the generated download link.

## Output directories
- Every task receives a dedicated subdirectory under `RESULTS_DIR`.
- Executors write their JSON summary to `<task_id>/responses.json`.
- Training executors produce checkpoints and, when HTTP uploads are enabled,
  create `final_model.tar.gz` or `final_lora.tar.gz`.

## HTTP artifact workflow
1. Stage 1 declares `spec.output.destination.type: http`.
2. The worker keeps a local copy and uploads the archive to the orchestrator.
3. The orchestrator serves the bundle at
   `GET /api/v1/results/{task_id}/files/<archive>`.
4. Downstream stages reference the URL via `${stage.result.final_model_archive_url}`
   (or the LoRA equivalent).

Prefer shared storage? Point both the orchestrator and workers at the same
mount and disable HTTP uploads—the executors still emit artifacts locally and
templates can reference absolute paths.

## Debugging tips
- Inspect `worker.log` for executor output and upload diagnostics.
- Use `redis-cli` to check `worker:<worker_id>` hashes and heartbeat TTLs.
- If a pipeline stalls, confirm the Stage 1 task produced the expected
  `final_*_archive_url` and that the orchestrator results directory contains the
  uploaded `.tar.gz`.

## Elastic Scaling & Task Merge
- **Elastic scaling** is controlled by `ENABLE_ELASTIC_SCALING` (default `true`).
  The orchestrator can temporarily disable idle workers and reactivate them when
  backlog grows.
- **Task merge** allows the orchestrator to coalesce duplicate inference/RAG
  requests. Workers receive a parent payload with `merge_children`, and
  executors (e.g. vLLM) emit per-child outputs under `result.children`; the
  runner writes each child result to its own directory.
- **Multi-GPU execution**: when multiple GPUs are available, vLLM automatically
  sets `tensor_parallel_size`, and PPO/DPO/SFT executors launch distributed jobs
  via `torchrun`. Override `training.allow_multi_gpu=false` or
  `training.nproc_per_node` to constrain world size.
- **Bandwidth throttling**: set `WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC` to
  simulate limited HTTP throughput; the worker reports the value and delays
  callbacks accordingly.
