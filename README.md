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
### 1. Install dependencies (via uv)
```bash
# 安装 uv（如系统已安装可跳过，更多方式参考 https://docs.astral.sh/uv ）
pip install uv

# 创建并激活隔离环境
uv venv .venv
source .venv/bin/activate

# 同步 orchestrator + 默认 worker 所需依赖（含 transformers/torch）
uv sync --extra inference

# 如需其它能力，可追加 extras：
# uv sync --extra inference --extra rag       # 启用 RAG
# uv sync --extra inference --extra agent     # 启用 Agent 执行器
# uv sync --all-extras                        # 安装全部可选组件
```
详见 `docs/executors.md` 获取每个 `taskType` 对应的执行器和可选依赖说明。`uv sync`
会读取仓库内的 `uv.lock`，确保不同机器之间依赖版本一致。

Start Redis:
```bash
redis-server
```

### 2. Run the Orchestrator
```bash
export REDIS_URL="redis://localhost:6379/0"
export ORCHESTRATOR_TOKEN="dev-token"  # optional auth
export ORCHESTRATOR_RESULTS_DIR=./results_host
uv run python orchestrator/main.py
# listens on 0.0.0.0:8000 (override with HOST_APP_PORT/PORT or pass --port)
```

### 3. Run a Worker
```bash
export REDIS_URL="redis://localhost:6379/0"
export RESULTS_DIR=./results_workers    # or an NFS mount
export ORCHESTRATOR_BASE_URL="http://127.0.0.1:8000"  # enable HTTP artifact uploads
uv run python worker/main.py
```
Workers register with Redis, stream heartbeats, and execute incoming tasks.
If the YAML sets `spec.output.destination.path`, results go there; otherwise
`RESULTS_DIR/<task_id>/responses.json` is used.

### 4. Inspect built-in metrics
```bash
curl http://127.0.0.1:8000/metrics
```
The response aggregates event counters (tasks succeeded/failed/requeued, active
workers, heartbeat counts) and is refreshed whenever the orchestrator receives
Redis events.

### 5. Submit a task
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
  own subdirectory containing `responses.json`, `manifest.json`, `logs/` and
  `artifacts/`.
- **Orchestrator output root** defaults to `./results_host`. Results ingested by
  `/api/v1/results` are stored under this tree，上传接口会自动落盘到
  `artifacts/<filename>` 并刷新 manifest。
- Stage-to-stage pipelines can reference uploaded archives using
  `checkpoint.load.url`, for example
  `url: "${stage1.result.final_model_archive_url}"`.
- If you prefer classic shared storage, point both `RESULTS_DIR` variables at a
  common mount and skip HTTP uploads.

每次任务运行后会调用 `orchestrator.manifest_utils.sync_manifest` 同步 `manifest.json`，其中对
在模板中声明的 `spec.output.artifacts` 会标注 `status` 为 `present` 或 `missing`，
方便在验证阶段快速定位缺失工件。

## State & Metrics

- `StateManager` 默认关闭；设置 `ORCHESTRATOR_STATE_ENABLED=1` 后，会定期将
  `TaskStore`、任务记录和父子分片信息写入
  `${ORCHESTRATOR_STATE_DIR:-./state}/task_state.json`，并在重启时自动恢复。
- 指标快照默认写入 `${ORCHESTRATOR_METRICS_DIR:-./metrics}/metrics.json`，同时在
  `/metrics` HTTP 接口中提供实时快照。原始事件以 JSONL 形式写入
  `${ORCHESTRATOR_METRICS_DIR:-./metrics}/events.log`。

### Key environment variables

| 变量 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `ORCHESTRATOR_STATE_ENABLED` | `0` | 是否启用状态快照与恢复 |
| `ORCHESTRATOR_STATE_DIR` | `./state` | 状态快照目录（启用快照时生效） |
| `ORCHESTRATOR_METRICS_DIR` | 取决于 `STATE_ENABLED`（默认为 `./metrics`） | 指标输出目录 |
| `STATE_FLUSH_INTERVAL_SEC` | `5` | 状态写盘周期 |
| `RESULTS_DIR` | `./results_host` | Orchestrator 结果目录 |
| `ORCHESTRATOR_TOKEN` | 无 | 可选的 Bearer Token，用于保护 API |


## Host Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string (fallback when env unset). |
| `RESULTS_DIR` | `./results_host` | Directory where orchestrator writes ingested results. |
| `ORCHESTRATOR_TOKEN` | – | Optional bearer token to protect APIs. |
| `ORCHESTRATOR_DISPATCH_MODE` | `adaptive` | Scheduler flavour (`adaptive`, `fixed_pipeline`, `static_round_robin`). |
| `ORCHESTRATOR_WORKER_SELECTION` | `best_fit` | Worker selection policy (`best_fit`, `first_fit`). |
| `ENABLE_CONTEXT_REUSE` | `true` | Bias toward workers with fresh model/dataset caches (subject to TTL). |
| `WORKER_CACHE_TTL_SEC` | `3600` | Cache metadata expiry before reuse priority is dropped. |
| `ENABLE_TASK_MERGE` | `true` | Enables DAG-level coalescing of identical tasks. |
| `TASK_MERGE_MAX_BATCH_SIZE` | `4` | Max number of siblings merged per dispatch. |
| `ENABLE_ELASTIC_SCALING` | `true` | Global toggle for auto disable/enable logic. |
| `SCHEDULER_LAMBDA_INFERENCE` | `0.4` | λ weight for inference tasks in the convex throughput/cost score. |
| `SCHEDULER_LAMBDA_TRAINING` | `0.8` | λ weight for training-style tasks. |
| `SCHEDULER_LAMBDA_OTHER` | `0.5` | λ weight for uncategorised tasks. |
| `SCHEDULER_SELECTION_JITTER` | `1e-3` | Deterministic jitter magnitude for tie-breaking. |
| `ELASTIC_AUTO_DISABLE_IDLE_SEC` | `60` | Idle duration before auto-disabling an idle worker (when queue ≤ threshold). |
| `ELASTIC_AUTO_DISABLE_QUEUE_MAX` | `0` | Max ready-queue length that still allows idle auto-disable. |
| `ELASTIC_AUTO_ENABLE_QUEUE_THRESHOLD` | `0` | Ready queue length that triggers auto re-enable of disabled workers. |
| `ELASTIC_AUTO_POLL_INTERVAL_SEC` | `30` | Elastic manager poll interval. |
| `ELASTIC_AUTO_TOGGLE_COOLDOWN_SEC` | `>=60` | Cooldown before toggling the same worker again. |
| `ELASTIC_AUTO_MIN_ACTIVE_WORKERS` | `1` | Minimum enabled workers kept alive. |
| `HOST_METRICS_DIR` / `ORCHESTRATOR_METRICS_DIR` | `./metrics` | Metrics snapshot & event log directory. |
| `LOG_LEVEL` | `INFO` | Orchestrator log level. |
| `LOG_FILE` | `host.log` | Rotating log file emitted by the orchestrator. |
| `HOST_APP_PORT` / `PORT` | `8000` | HTTP bind ports. |
| `HOST_APP_HOST` | `0.0.0.0` | HTTP bind address. |

## Worker Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | – | Redis connection string (required). |
| `RESULTS_DIR` | `./results_workers` | Working directory for task outputs. |
| `TASK_TOPIC` | `tasks` | Redis pub/sub channel to subscribe. |
| `HEARTBEAT_INTERVAL_SEC` | `30` | Heartbeat cadence. TTL auto set to `max(120, 4*interval)`. |
| `WORKER_ID` | random | Optional persistent identifier. |
| `WORKER_TAGS` | empty | Comma-separated scheduler hints (e.g. `gpu,a100`). |
| `LOG_LEVEL` | `INFO` | Worker logging level. |
| `ORCHESTRATOR_BASE_URL` | – | Base URL for retrieving uploaded artifacts in staged flows. |
| `RESULTS_UPLOAD_URL` | – | Override default HTTP upload endpoint. |
| `WORKER_COST_PER_HOUR` | `1.0` | Cost metadata reported during registration. |
| `WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC` | – | Optional bandwidth cap; affects reported hardware info and callback throttling. |
| `WORKER_EXECUTOR_MASK` | – | Limit worker to specific executors (comma separated). |
| `WORKER_LOG_TO_STDOUT` | `true` | Disable to log only to file handler. |
## Testing

轻量级单元测试覆盖任务池、manifest、聚合与指标逻辑：

```bash
uv run pytest tests/test_core_flow.py
```
运行结果会记录在 `.codex/testing.md` 与 `verification.md`，方便回溯。

## Validation Helpers

- `scripts/worker_validate.py` —— 通过 `echo` 模板快速验证本地 orchestrator/worker 流；支持 `--scenario echo-local` 与 `--scenario echo-http`。
- `scripts/validate_echo_local.sh` 与 `scripts/validate_echo_http.sh` —— 基于环境变量 `ORCHESTRATOR_URL`、`ORCHESTRATOR_TOKEN` 进行常用场景验证。
- `scripts/replay_task.py` —— 从状态快照中读取原始 YAML 并重新提交任务（便于失败重放）。
- `scripts/export_results.py` —— 收集 `RESULTS_DIR` 下的 `responses.json`，导出为 CSV（可结合 `--state-file` 丰富元数据）。
- `scripts/task_profile_report.py` —— 基于导出的 CSV 生成 Markdown 报告，统计成功率与平均耗时。

## Docker Compose Deployment

提供 `docker-compose.yml` 与 `.env.example`，可一键启动 Redis、Orchestrator 与 Worker：

```bash
cp .env.example .env
docker compose up --build
```

默认将结果与状态写入 `./data/` 目录，可按需调整 `.env` 中的路径或添加额外 extras（例如训练、RAG）。

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
