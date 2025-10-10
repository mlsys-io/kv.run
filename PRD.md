# MLOC 产品需求文档

- 修订日期：2025-10-09
- 撰写者：Codex

## 1. 项目愿景与目标

### 1.1 愿景概述
构建一个高质量、开源的模块化 LLM 任务调度与执行系统，为多模态推理、增量训练、强化学习等工作负载提供统一的控制面。MLOC 将现有的 host/worker 分离架构打磨为可复制、可观测、易扩展的基础设施，成为研究团队与平台工程师搭建 LLM 生产体系的标准样板。

### 1.2 目标用户与价值主张
- **模型研发/研究团队**：以 YAML 模板描述实验，快速切换模型、数据与执行方式，获得统一的日志与工件管理。
- **平台与基础设施工程师**：复用 FastAPI 控制平面与 Redis 调度逻辑（`orchestrator/`），在现有环境接入新型执行器，保障资源利用率。
- **业务解决方案团队 / 应用研发**：以模板库（`templates/`, `exp/`）快速落地推理、RAG、SFT、PPO 等流水线，复用 artifacts 上传与回收机制。

### 1.3 成功指标
- 任务提交流水端到端成功率 ≥ 99%，单任务失败可自动重试 ≤ 3 次。
- 单任务结果产出 95th latency ≤ 90 秒（推理） / ≤ 15 分钟（训练类任务）。
- 数据并行开启时支持 ≥ 8 个 shard 并行，聚合等待时间不超过最长子任务完成时间 + 15 秒。
- 控制平面在 4 节点 worker 池下 CPU 利用率 ≤ 40%，Redis 事件滞留 < 10 秒。
- 提供完整本地自动化验证脚本，覆盖核心流程（提交、调度、工件上传、聚合）。

## 2. 核心功能模块

### 2.1 Orchestrator 控制平面（FastAPI，`orchestrator/`）
- **职责**：解析 YAML 任务（`parser.py`）、维护内存态 TaskRecord（`task.py`）、调用调度器（`scheduler.py`、`dispatch.py`）、聚合分片结果（`aggregation.py`），并通过 HTTP API 暴露提交、查询、结果下载接口（`main.py`）。
- **关键能力**：
  - 支持 SLO 感知的任务池，按照 SLO 进度出队（`task_store.py`）。
  - 数据并行：当 `spec.parallel.enabled=true` 时自动切分 dataset/list 输入，记录父子关系并聚合结果。
  - 结果管理：`POST /api/v1/results` 写入 `RESULTS_DIR`，支持子任务合并与 HTTP 结果转发。
- **对外接口**：`/api/v1/tasks`, `/api/v1/results`, `/workers`, `/admin/cleanup` 等；所有敏感操作可选 Bearer Token。
- **依赖**：Redis（worker 注册、心跳、任务发布）、本地文件系统（结果）、可选 HTTP artifact 存储。

### 2.2 Worker 执行平面（`worker/`）
- **职责**：订阅 Redis topics，按任务类型选择执行器（`executors/`），写入结果目录并按需回传 HTTP。
- **执行器矩阵**：
  - 推理：`VLLMExecutor`（GPU）、`HFTransformersExecutor`（CPU/回退）。
  - 训练：`SFTExecutor`、`LoRASFTExecutor`、`PPOExecutor`、`DPOExecutor`。
  - 应用：`RAGExecutor`、`AgentExecutor` 等。
- **运行机制**：
  - 生命周期管理（`lifecycle.py`）负责心跳、状态迁移（IDLE → RUNNING → SUCCEEDED/FAILED）。
  - 输出目录通过 `spec.output.destination` 可定制，HTTP 目标由 `Runner._maybe_emit_http` 实现。
  - Worker 根据 `torch.cuda.is_available()` 动态加载 GPU 执行器，缺少依赖时自动降级。

### 2.3 任务模板与图谱（`templates/`, `exp/`, `worker/executors/graph_templates.py`）
- **职责**：以 YAML 定义模型、数据、资源、并行及输出配置；为典型任务提供可复用示例。
- **关键能力**：
  - 支持 dataset/list 输入、数据随机化与分片、Prompt 图模板（graph templates）。
  - 提供推理/训练/强化学习/代理/RAG 等 10+ 示例，为 PRD 用户故事与验收标准提供素材。
  - 通过 `spec.output.destination` 控制本地目录或 HTTP 上传，推动可观测与工件管理。

### 2.4 工件管理与可观测性管道
- **职责**：统一记录任务输出、上传额外文件、聚合分片；通过 Redis Pub/Sub（`workers.events`, `tasks.events`）传递生命周期事件。
- **改进方向**：
  - 增补结果目录结构约定（responses.json、logs、checkpoints、archives）。
  - 指定事件 schema，为后续监控/仪表盘接入铺路。
  - 制定本地验证脚本，验证结果落盘与 HTTP 上传的幂等性。

## 3. 系统架构与流程

### 3.1 总体拓扑
```
Client → FastAPI Orchestrator → Redis Pub/Sub → Worker Fleet → Executors → Results Store
```
- 控制平面 `orchestrator/main.py` 监听 0.0.0.0:8000，承担任务入口。
- Worker 通过 `worker/main.py` 注册硬件信息、心跳与标签，Idle 池由调度器筛选。
- 结果写入主机 `RESULTS_DIR` 或经 HTTP 上传至 Orchestrator，再被 Stage 2 任务引用。

### 3.2 任务生命周期
1. **提交**：客户端上传 YAML；`TaskStore.enqueue_for_dispatch` 记录任务并解析依赖。
2. **调度**：`DispatchManager.try_dispatch_one` 评估并行开关、筛选满足硬件约束的 Worker，发布 Redis 消息。
3. **执行**：Worker `Runner` 确认任务归属、分配执行器、写入 `responses.json`，必要时上传工件。
4. **聚合**：对于分片任务，`maybe_aggregate_parent` 汇总子任务结果并回写父任务状态。
5. **完成**：TaskRecord 状态迁移到 DONE，事件写入 Redis 通道，客户端可轮询或获取回调。

### 3.3 运维与部署路径
- **本地运行**：README 指引 Redis → Orchestrator → Worker 手动启动；脚本位于 `scripts/`。
- **容器化**：`worker/docker/` 提供 CPU/GPU Dockerfile，`docker-compose.yml` 演示 Redis + Worker + NFS 共享盘。
- **缺口**：
  - 缺少 orchestrator Dockerfile/Compose 与一键部署脚本。
  - 没有集中化配置（env 文件散落），缺少环境模板。
  - 监控与日志仅靠 stdout/文件，未提供指标暴露与仪表盘。
  - 状态持久化依赖内存字典，进程重启需明确恢复策略。

## 4. 非功能与质量要求

### 4.1 可靠性与恢复
- 任务状态应在内存与文件系统之外具备恢复手段（Redis 备份或快照），重启后可重新构建调度池。
- Requeue 线性退避上限 ≤ 2 分钟，异常次数 ≥ 3 时进入人工干预队列。
- 心跳 TTL 默认 120 秒，可配置；缺失心跳需触发 worker 清理与任务重派。

### 4.2 性能与扩展性
- 控制平面可在 4 CPU 实例上处理 ≥ 50 TPS 任务入队；Redis publish 95th latency ≤ 1 秒。
- 数据并行 shard 聚合需保证在单机文件系统上完成，聚合逻辑对 8-16 份子任务维持线性复杂度。
- 执行器加载模型需缓存/复用，避免重复 download；vLLM executor 支持多 GPU 节点（tensor parallel）。

### 4.3 可运维性与可观测性
- 补充运行 Runbook：启动顺序、环境变量说明、常见故障与缓解步骤。
- 事件总线需定义结构化 schema（event type、task_id、timestamp、payload），以便后续写入指标。
- 日志等级（INFO/DEBUG）与滚动策略（默认 5×5MB）需在文档中明确。

### 4.4 文档、测试与质量保障
- 交付 PRD 后需补齐开发指南与 API 参考，保证 README 统一链接。
- 提供本地验证脚本（python 或 shell）覆盖：推理任务、数据并行、HTTP artifact 上传、训练回退。
- 所有验证由本地自动化执行，禁止引入外部 CI/安全机制。

## 5. 分阶段路线图

> 节奏以 2 周冲刺为单位，可按资源调整。每个里程碑要求更新 `.codex/testing.md` 与自审报告。

### Milestone 0 — 基线梳理（1 Sprint）
- **目标**：整理现状、补齐文档结构。
- **交付物**：本 PRD、模块边界说明、依赖审计（标记必须/可选）、运行 Runbook 初稿。

### Milestone 1 — 控制平面强化（2 Sprints）
- **目标**：提升调度可靠性与状态恢复。
- **交付物**：
  - TaskStore 状态持久化设计（Redis snapshot / 文件重建机制）。
  - 数据并行聚合的错误处理与重试策略。
  - `/workers`、`/tasks` API 增补分页与过滤，提供 SLO 指标。

### Milestone 2 — 执行平面与模板整备（2 Sprints）
- **目标**：规范执行器契约与模板体验。
- **交付物**：
  - 执行器输入/输出 schema 文档与样例。
  - 模板库分层：核心模板 + 实验模板，附验收案例与预期输出。
  - Worker 验证脚本（含 GPU/CPU、HTTP 上传、本地模式）。

### Milestone 3 — 可观测性与测试体系（2 Sprints）
- **目标**：建立可视化指标与自动化测试。
- **交付物**：
  - Redis 事件 → 指标转换脚本，生成 Prometheus 兼容指标或本地报告。
  - 覆盖核心链路的自动化测试套件及日志采集。
  - 失败重放工具，支持按 task_id 重触发任务或回放输出。

### Milestone 4 — 部署与运维自动化（2 Sprints）
- **目标**：提供生产级部署路径。
- **交付物**：
  - orchestrator + worker + Redis 的 Docker Compose / Helm 栈。
  - 环境变量模板与配置打包（示例 `.env`、bootstrap 脚本）。
  - 运维 Runbook v1（扩缩容、日志轮转、灾难恢复）。

### Milestone 5 — 数据洞察与复盘（2 Sprints）
- **目标**：沉淀工件管理、指标分析与知识库。
- **交付物**：
  - 任务结果与工件整理脚本（JSON/Parquet 导出、版本化目录）。
  - 任务画像报告（执行器成功率、失败原因、资源使用）。
  - 指标订阅/告警方案（例如 Redis backlog、任务失败率）。

## 附录：约束与假设
- 系统坚持 host/worker 分离；禁止引入额外安全机制（RBAC、认证代理），部署方可根据需要在边界层补充。
- 所有功能优先复用现有生态（FastAPI、Redis、vLLM、TRL），禁止自研重复轮子。
- 依赖安装通过 `uv` 管理，保持 Python 3.12 基线。
- 任务与标签信息仅存于 Redis + 文件系统，后续持久化设计需遵守无外部 CI 的限制。
- 所有验证步骤由本地自动执行脚本完成，不接入云端 CI/CD 或人工流转。
