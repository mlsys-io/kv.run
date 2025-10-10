# MLOC Runbook（2025-10-09，编写者：Codex）

## 启动顺序
1. **Redis**：`redis-server`，确保 `REDIS_URL` 指向可访问的实例。
2. **Orchestrator**：
   ```bash
   export REDIS_URL="redis://localhost:6379/0"
   export ORCHESTRATOR_STATE_DIR=./state
   python orchestrator/main.py
   ```
3. **Worker**（按需多实例）：
   ```bash
   export RESULTS_DIR=./results_workers
   pip install mloc[inference]  # 或根据需求安装其它 extras
   python worker/main.py
   ```

> 使用 Docker Compose 时，可运行 `docker compose up`，编排脚本会自动安装依赖并挂载 `./data/` 目录以保存状态与结果。

## 状态与恢复
- 快照写入 `${ORCHESTRATOR_STATE_DIR}/task_state.json`，重启时会自动恢复任务池、父子分片与 TaskRecord 状态。
- 指标写入 `${ORCHESTRATOR_STATE_DIR}/metrics/metrics.json`，原始事件记录在 `events.log`。
- 若需强制清理，可安全删除 `state/` 目录后重启，系统会从空白状态启动。

## 常用诊断
| 目标 | 命令 |
| ---- | ---- |
| 浏览指标 | `curl http://127.0.0.1:8000/metrics` |
| 查看 manifest | `cat results_host/<task_id>/manifest.json` |
| 检查死信 | `curl http://127.0.0.1:8000/api/v1/dead_letters` |
| 快速重队任务 | `curl -X POST http://127.0.0.1:8000/admin/cleanup` |

## 验证脚本
- `scripts/validate_echo_local.sh`：验证本地结果落盘（EchoExecutor + Local output）。
- `scripts/validate_echo_http.sh`：验证 HTTP 回传与 orchestrator 聚合。
- 两者依赖 `ORCHESTRATOR_URL` 与 `ORCHESTRATOR_TOKEN` 环境变量，更多选项可参考 `scripts/worker_validate.py --help`。

## 故障排查提示
- **任务卡住**：检查 `/metrics` 中 `tasks_requeued` 计数与 `dead_letters` API，确认是否超出重试上限。
- **工件缺失**：核对 manifest 中 `status == "missing"` 的条目，并检查执行器是否安装了正确的 extras。
- **Worker 离线**：`metrics.json` 的 `active_workers` 为空时，查看 `metrics/events.log` 判断最后一次心跳时间，同时确保 Redis 权限和网络正常。

## 例行维护
- 建议定期备份 `state/` 与 `results_host/`，便于灾难恢复。
- 清理旧任务：删除相应目录后运行 `curl -X POST /admin/cleanup`，以移除无效 worker 关联。
- 升级依赖：根据需要运行 `pip install --upgrade mloc[<extras>]`，升级前请阅读 `docs/executors.md` 了解依赖分层。
- 使用 Docker Compose 升级：更新代码后执行 `docker compose build --no-cache` 以重新安装依赖。
