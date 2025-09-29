# MLOC - Modular LLM Operations Container

MLOC provides a scalable control plane for Large Language Model workloads. The
system consists of an **Orchestrator** (FastAPI), a fleet of **Workers**, and a
Redis-backed pub/sub bus for dispatch and status tracking. Tasks are described
in YAML and can run spectrum from single-shot inference to multi-stage PPO/DPO
training pipelines.

```
+-------------+    HTTP API    +--------------+    Redis Pub/Sub    +-------------+
|   Client    | -------------> | Orchestrator | -----------------> |   Worker    |
+-------------+                +--------------+                    +-------------+
                                      |                                    |
                                      |                                    |
                                      v                                    v
                              +-------------+                      +-------------+
                              |    Redis    |                      |  Executors  |
                              |  (Broker)   |                      | (vLLM, TRL) |
                              +-------------+                      +-------------+
```

## Components
- **Orchestrator** – Parses YAML, manages dependencies, schedules work, and
  aggregates results/events.
- **Workers** – Subscribe to Redis topics, pick executors (vLLM, Hugging Face
  Transformers, PPO/DPO), and write outputs.
- **Redis** – Message bus plus lightweight state store for worker metadata and
  task status notifications.

## Key Capabilities
- Declarative task definitions with optional `spec.stages` pipelines.
- Resource-aware scheduling with optional data-parallel fan-out when
  `spec.parallel.enabled=true` for inference jobs.
- Flexible artifact delivery: Workers can persist to shared storage or upload
  binaries (for example, fine-tuned checkpoints) back to the orchestrator over
  HTTP after every training run.
- Per-task output directory overrides via `spec.output.destination.path`
  (relative paths resolve under the worker `RESULTS_DIR`).
- Shared storage ready: Docker Compose sample mounts an NFS export so all
  workers write to the same location.

## Quick Start (local)
### 1. Install dependencies
```bash
uv sync                 # core features
uv sync --extra ppo     # PPO / DPO executors
```
Start Redis:
```bash
redis-server
```

### 2. Run the Orchestrator
```bash
export REDIS_URL="redis://localhost:6379/0"
export ORCHESTRATOR_TOKEN="dev-token"  # optional auth
export ORCHESTRATOR_RESULTS_DIR=./results_host
python orchestrator/main.py
# listens on 0.0.0.0:8000 (override with PORT)
```

### 3. Run a Worker
```bash
export REDIS_URL="redis://localhost:6379/0"
export RESULTS_DIR=./results_workers    # or an NFS mount
export ORCHESTRATOR_BASE_URL="http://127.0.0.1:8000"  # enable HTTP artifact uploads
python worker/main.py
```
Workers register with Redis, stream heartbeats, and execute incoming tasks.
If the YAML sets `spec.output.destination.path`, results go there; otherwise
`RESULTS_DIR/<task_id>/responses.json` is used.

### 4. Submit a task
```bash
curl -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/inference_vllm_mistral.yaml
```
Samples under `client/` run similar requests.

## YAML Primer
```yaml
spec:
  taskType: "inference"
  resources: { ... }
  model: { ... }
  data:  { ... }
  output:
    destination:
      type: http
      url: http://127.0.0.1:8000/api/v1/results
      headers:
        Authorization: "Bearer dev-token"
```
- Enable automatic sharding by setting `spec.parallel.enabled: true` with an
  appropriate dataset split.
- Define multi-stage flows using `spec.stages`; the orchestrator chains them via
  dependencies.
- When `spec.output.destination.type` is `http`, the worker will upload the
  final model artifacts to the orchestrator (in addition to local persistence)
  so that downstream stages can fetch them by ID.

## Artifact Handling Overview

- **Worker output root** defaults to `./results_workers`. Each task receives its
  own subdirectory containing `responses.json`, checkpoints, and any archived
  model directories.
- **Orchestrator output root** defaults to `./results_host`. Results ingested by
  `/api/v1/results` are stored under this tree, and the new endpoint
  `POST /api/v1/results/{task_id}/files` accepts extra files (for example
  `final_model.zip`).
- Stage-to-stage pipelines can reference uploaded archives using
  `checkpoint.load.url`, for example
  `url: "${stage1.result.final_model_archive_url}"`.
- If you prefer classic shared storage, point both `RESULTS_DIR` variables at a
  common mount and skip HTTP uploads.

## Shared Storage via NFS (optional)
1. Export an NFS directory on the orchestrator or storage host (instructions in
   `orchestrator/README.md`).
2. Mount the export on every worker and set `RESULTS_DIR` to the mountpoint.
3. Docker users can rely on `worker/docker-compose.yml`, which mounts an NFS
   volume at `/mnt/mloc-results` inside the container.

## Repository Layout
```
README.md                 # Top-level overview
orchestrator/             # Scheduling service + docs
worker/                   # Worker process, executors, docker assets
client/                   # Submission helpers
templates/                # YAML examples
```

See also:
- `orchestrator/README.md` for API and scheduling internals.
- `worker/README.md` for worker configuration and runtime flow.
- `worker/docker/README.md` for container-focused instructions.
