# Orchestrator Service Guide

The orchestrator is the control plane for the platform. It parses task YAML,
tracks dependencies, schedules work to workers via Redis pub/sub, aggregates
child results, and exposes an HTTP API for clients to submit jobs and retrieve
artifacts.

## Running the service
```bash
export REDIS_URL="redis://localhost:6379/0"
export ORCHESTRATOR_TOKEN="dev-token"     # optional bearer auth
export ORCHESTRATOR_RESULTS_DIR=./results_host
python orchestrator/main.py                # listens on 0.0.0.0:8000 by default
```
Set `PORT` if you want to bind to a different TCP port.

### Runtime environment variables
| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | – | Connection string for Redis (mandatory). |
| `ORCHESTRATOR_TOKEN` | empty | If set, the API requires `Authorization: Bearer <token>`. |
| `ORCHESTRATOR_RESULTS_DIR` | `./results_host` | Root directory where results and uploaded artifacts are stored. |
| `PORT` | `8000` | HTTP port. |
| `LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `LOG_FILE` | `orchestrator.log` | Rolling log file path. |
| `LOG_MAX_BYTES` | `5_242_880` | Max size of each log file before rotation. |
| `LOG_BACKUP_COUNT` | `5` | Number of rotated log files to keep. |
| `HEARTBEAT_TTL_SEC` | `120` | TTL for worker heartbeats before they are considered stale. |

## HTTP API
| Method & Path | Description |
|---------------|-------------|
| `GET /healthz` | Liveness probe. |
| `GET /workers` | List registered workers (auth required). |
| `GET /workers/{worker_id}` | Inspect a worker (auth required). |
| `POST /admin/cleanup` | Remove stale workers (auth required). |
| `POST /api/v1/tasks` | Submit YAML task definitions (`Content-Type: text/yaml`). |
| `GET /api/v1/tasks` | List all in-memory task records (auth required). |
| `GET /api/v1/tasks/{task_id}` | Fetch a single task record (auth required). |
| `POST /api/v1/results` | Ingest execution results (called by workers). |
| `GET /api/v1/results/{task_id}` | Retrieve the stored JSON result. |
| `POST /api/v1/results/{task_id}/files` | Upload an additional artifact (e.g., checkpoint archive). |
| `GET /api/v1/results/{task_id}/files/{filename}` | Download a previously uploaded artifact. |

All endpoints that mutate state or expose sensitive information require the
optional bearer token if `ORCHESTRATOR_TOKEN` is configured.

## Scheduling
- Workers publish their hardware info and heartbeats to Redis. The orchestrator
  selects idle workers that satisfy requested resources (CPU, memory, GPU
  requirements).
- When `spec.parallel.enabled: true` on inference jobs, the scheduler can fan
  out shards across multiple workers. Results are aggregated before the parent
  task completes.
- `spec.stages` pipelines are expanded into distinct tasks with automatically
  inferred dependencies.

## Artifact handling
- Every task receives a directory under `ORCHESTRATOR_RESULTS_DIR/<task_id>/`.
  `POST /api/v1/results` writes the canonical `responses.json` file.
- Workers can upload additional binaries—such as `final_model.zip` or
  `final_lora.zip`—by calling `POST /api/v1/results/{task_id}/files`. The new
  `GET /api/v1/results/{task_id}/files/{filename}` endpoint serves those files to
  downstream stages.
- Templates that use staged fine-tuning can reference
  `${stage.result.final_*_archive_url}` to pull the previous stage's archive via
  HTTP rather than relying on shared storage.

## Monitoring and maintenance
- Logs are written to `orchestrator.log` and stdout. Increase `LOG_LEVEL`
  to `DEBUG` for verbose scheduling details.
- Redis pub/sub channels `workers.events` and `tasks.events` include lifecycle
  notifications suitable for dashboards or alerting.
- Use `POST /admin/cleanup` to remove workers that have missed their heartbeat
  TTL.

## Shared storage (optional)
If you prefer to bypass HTTP uploads, share a filesystem between the
orchestrator and workers (for example, via NFS) and point both `RESULTS_DIR`
variables to the same mount. In this configuration the new artifact endpoints
are still available but the worker uploads become optional.
