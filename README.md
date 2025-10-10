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
pip install mloc[inference]            # Orchestrator + CPU worker baseline
pip install mloc[inference,rag,agent]  # 增加 RAG 与 Agent 执行链路
pip install mloc[inference,training]   # 启用微调、LoRA、PPO/DPO 等训练能力
```
详见 `docs/executors.md` 获取每个 `taskType` 对应的执行器和可选依赖说明。
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

每次任务运行后会调用 `manifest_utils.sync_manifest` 同步 `manifest.json`，其中对
在模板中声明的 `spec.output.artifacts` 会标注 `status` 为 `present` 或 `missing`，
方便在验证阶段快速定位缺失工件。

## State & Metrics

- `StateManager` 会定期将 `TaskStore`、任务记录和父子分片信息写入
  `${ORCHESTRATOR_STATE_DIR:-./state}/task_state.json`，重启时自动恢复。
- 行为指标通过 `metrics/metrics.json` 持久化，同时在 `/metrics` HTTP 接口中提供
  实时快照。原始事件以 JSONL 形式写入 `metrics/events.log`。

### Key environment variables

| 变量 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `ORCHESTRATOR_STATE_DIR` | `./state` | 状态快照与指标输出目录 |
| `STATE_FLUSH_INTERVAL_SEC` | `5` | 状态写盘周期 |
| `RESULTS_DIR` | `./results_host` | Orchestrator 结果目录 |
| `ORCHESTRATOR_TOKEN` | 无 | 可选的 Bearer Token，用于保护 API |

## Testing

轻量级单元测试覆盖任务池、manifest、聚合与指标逻辑：

```bash
pytest tests/test_core_flow.py
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
