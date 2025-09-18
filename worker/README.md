# Worker Guide

Worker 负责订阅 `tasks` 频道、执行任务并把结果写入共享目录。它会自动：
- 连接 Redis，完成注册/心跳/状态上报
- 根据任务类型选择执行器（`inference` -> vLLM/Transformers，`ppo`/`dpo` -> TRL）
- 将执行结果和辅助产物落地到 `RESULTS_DIR/<task_id>/`

## 环境变量
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `REDIS_URL` | (必填) | Redis 地址，例如 `redis://orchestrator:6379/0` |
| `RESULTS_DIR` | `./results` | Worker 本地/挂载的输出根目录 |
| `TASK_TOPIC` | `tasks` | 订阅的 Redis 频道 |
| `HEARTBEAT_INTERVAL_SEC` | `30` | 心跳发送间隔 |
| `WORKER_ID` | 随机生成 | 自定义 Worker ID（可选）|
| `WORKER_TAGS` | 空 | 逗号分隔的标签，调度时可用于筛选 |
| `LOG_LEVEL` | `INFO` | Worker 日志级别 |

> YAML 可通过 `spec.output.destination.path` 覆盖单次任务输出目录，相对路径会基于 `RESULTS_DIR` 解析。

## 启动
```bash
export REDIS_URL="redis://localhost:6379/0"
export RESULTS_DIR=/mnt/mloc-results   # 或其他目录
python worker/main.py
```
Worker 启动后会：
1. 采集硬件信息（CPU/内存/GPU）并注册到 Redis；
2. 开启心跳线程，维持 `HEARTBEAT_INTERVAL_SEC` 间隔更新；
3. 按任务类型执行相应的 Executor；
4. 在任务完成/失败时通过 `tasks.events` 上报结果。

## 输出目录策略
- Runner 会优先读取任务 YAML 中的 `spec.output.destination.path`，若为相对路径则拼接在 `RESULTS_DIR` 下；
- 最终所有产出都会位于 `<目标路径>/<task_id>/responses.json`（执行器返回值由 Runner 统一写入）；
- PPO/DPO 等执行器仍可在 `out_dir` 下创建其他子目录（如 `checkpoints/`）。

## 与 NFS 或共享存储配合
1. 在 Orchestrator 主机导出 NFS（参考 `orchestrator/README.md`）。
2. Worker 主机挂载该共享目录，例如：
   ```bash
   sudo mkdir -p /mnt/mloc-results
   sudo mount orchestrator-ip:/srv/mloc/results /mnt/mloc-results
   ```
3. 启动 Worker 前设置 `RESULTS_DIR=/mnt/mloc-results`，即可将所有结果集中存放。

## Docker / Compose
- `worker/docker/` 提供 CPU/GPU Dockerfile。
- `worker/docker-compose.yml` 集成了 NFS 卷示例：
  1. 在同目录创建 `.env`，定义 `NFS_SERVER`、`NFS_EXPORT_PATH`（以及可选 `NFS_VERSION`）。
  2. 运行 `docker compose up -d`，Worker 容器会自动将 `/mnt/mloc-results` 挂载到共享存储，并把 `RESULTS_DIR` 指向该路径。
- 详细步骤请阅读 `worker/docker/README.md`。

## 调试 & 运维
- 日志写入 `worker.log`（滚动）并同步在控制台输出。
- 可通过 `WORKER_TAGS` 标记硬件特性，协助调度与监控。
- 当 Worker 宕机或网络异常导致心跳丢失时，Orchestrator 会在 TTL 超时后将其标记为 stale，可通过 `/admin/cleanup` 清理。

如需编写自定义执行器，请参考 `worker/executors/base_executor.py`，并在 `worker/main.py` 中注册。
