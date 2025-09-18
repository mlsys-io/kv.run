# Orchestrator Service Guide

Orchestrator 是整套系统的控制面，负责：
- 解析 YAML 任务并构建依赖图 (`spec.stages` / `spec.dependsOn`)
- 根据 Worker 资源、心跳与调度策略，派发任务或自动进行数据并行分片
- 监听 `tasks.events`，更新任务状态、处理重试、聚合分片结果
- 维护 Worker 注册信息，提供监控与清理接口

## 启动方式
```bash
export REDIS_URL="redis://localhost:6379/0"
export ORCHESTRATOR_TOKEN="dev-token"   # 可选：开启 Bearer 鉴权
export LOG_LEVEL=INFO                     # 可选：调节日志级别

python orchestrator/main.py              # 默认监听 0.0.0.0:8080
```
若需要自定义端口：`export PORT=8000`。

### 主要环境变量
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `REDIS_URL` | (必填) | Redis 连接串，例如 `redis://host:6379/0` |
| `ORCHESTRATOR_TOKEN` | 空 | 若设置，则所有 API 请求需携带 `Authorization: Bearer <token>` |
| `PORT` | `8080` | HTTP 服务端口 |
| `LOG_LEVEL` | `INFO` | 日志级别（DEBUG/INFO/WARN/ERROR） |
| `LOG_FILE` | `orchestrator.log` | 滚动日志文件路径 |
| `LOG_MAX_BYTES` | `5_242_880` | 单个日志文件大小上限 |
| `LOG_BACKUP_COUNT` | `5` | 日志轮转份数 |
| `HEARTBEAT_TTL_SEC` | `120` | Worker 心跳 TTL，超时将被判定为 stale |

## API 概览
| 方法 & 路径 | 说明 |
|-------------|------|
| `GET /healthz` | 健康检查 |
| `GET /workers` | 列出所有 Worker（需鉴权）|
| `GET /workers/{worker_id}` | 查看单个 Worker 详情（需鉴权）|
| `POST /admin/cleanup` | 清理过期 Worker（需鉴权）|
| `POST /api/v1/tasks` | 提交 YAML 任务（`Content-Type: text/yaml`）|
| `GET /api/v1/tasks` | 查看任务列表（需鉴权）|
| `GET /api/v1/tasks/{task_id}` | 查看单个任务详情（需鉴权）|

> 注意：若配置了 `ORCHESTRATOR_TOKEN`，上述带鉴权接口需要在请求头增加 `Authorization: Bearer <token>`。

## 调度与分片
- 调度器先筛选满足 `spec.resources` 的空闲 Worker，再根据可用度选择策略（best-fit / first-fit / data-parallel）。
- 推理任务开启 `spec.parallel.enabled: true` 后，可在资源充足时按 `spec.data.split` 均分出多份子任务；子任务完成后会自动聚合。
- 任务 `spec.output.destination.path` 会原样传递给 Worker，Orchestrator 不会改写路径，只负责下发到子任务中。

## NFS 共享存储（可选）
常见做法是在 Orchestrator 主机导出 NFS，方便集中存放 Worker 结果。

```bash
# 1. 创建目录
sudo mkdir -p /srv/mloc/results

# 2. 安装 NFS server (Debian/Ubuntu)
sudo apt install -y nfs-kernel-server

# 3. 配置导出（示例：允许 172.28.176.0/24 网段）
echo "/srv/mloc/results 172.28.176.0/24(rw,sync,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -rav

# 4. 启动服务
sudo systemctl enable --now nfs-server
```
在 Worker 侧挂载后，将 `RESULTS_DIR` 指向挂载点即可（详见 `worker/README.md`）。

## 运行监控
- 日志：`orchestrator.log`（支持滚动），同时输出到控制台。
- Redis 中的 `workers.events` 与 `tasks.events` 频道可用于自建监控。
- 需要更详细调试时，将 `LOG_LEVEL` 调为 `DEBUG`。

## 常见排查
1. **任务提交成功但无 Worker 执行**：检查 Worker 是否心跳正常、`TASK_TOPIC` 是否正确；调用 `/workers` 查看状态。
2. **Worker 被判定为 stale**：确认 Worker 机器时间同步、网络稳定，或适当调高 `HEARTBEAT_TTL_SEC`。
3. **分片任务未并行**：确认任务为 `taskType=inference` 且已设置 `parallel.enabled=true`，并检查是否有足够空闲 Worker。

更多执行层细节和部署示例，请参考顶层 `README.md` 及 `worker/` 目录中的文档。
