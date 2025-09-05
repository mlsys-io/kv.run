# MLOC (Modular LLM Operations Container)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/kubernetes-ready-blue.svg)](https://kubernetes.io)

MLOC (Modular LLM Operations Container) 是一个统一的容器化框架，旨在简化和标准化 LLM 的训练、微调、推理和应用流程。通过配置文件驱动，即可在异构硬件集群中一键部署，并执行指定的 LLM 任务。

## ✨ 核心特性

- **🎯 配置驱动**: 节点角色和任务完全由 YAML 配置文件定义
- **🧩 模块化架构**: 核心功能（SFT, PPO, RAG 等）作为可插拔模块
- **🔧 开源集成**: 深度集成 TRL, vLLM, LangChain, Hugging Face 等优秀开源库  
- **🖥️ 硬件感知**: 智能识别和调度不同型号的 GPU 资源
- **☁️ 云原生**: 为 Kubernetes 设计，支持广域网分布式部署
- **📊 可观测**: 内置监控、日志聚合和用量统计

## 🏗️ 系统架构

MLOC 采用 Orchestrator/Worker 架构：

- **Orchestrator (主控节点)**: 负责任务调度、状态监控和资源管理
- **Worker (工作节点)**: 负责执行具体的训练和推理任务

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Orchestrator  │────│     Redis       │────│     Worker      │
│                 │    │  (Message Queue)│    │                 │
│ • API Server    │    │  • Task Queue   │    │ • Task Listener │
│ • Scheduler     │    │  • Worker       │    │ • Module Loader │
│ • Monitor       │    │    Registry     │    │ • Executor      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 快速开始

### 使用 Docker Compose (推荐开发环境)

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd mloc
   ```

2. **构建镜像**
   ```bash
   ./scripts/build.sh
   ```

3. **启动服务**
   ```bash
   docker-compose up -d
   ```

4. **验证部署**
   ```bash
   curl http://localhost:8000/health
   ```

### 使用 Kubernetes (推荐生产环境)

1. **部署到 K8s 集群**
   ```bash
   ./scripts/deploy_k8s.sh
   ```

2. **访问 API**
   ```bash
   kubectl port-forward service/orchestrator 8000:8000 -n mloc
   ```

## 📋 支持的任务类型

### 🎓 监督微调 (SFT)
使用 TRL 库进行监督微调，支持 LoRA、QLoRA 等高效适配器方法。

```yaml
taskType: "sft"
model:
  source:
    type: "huggingface"
    identifier: "mistralai/Mistral-7B-Instruct-v0.1"
  adapter:
    type: "qlora"
    r: 16
    lora_alpha: 32
```

### 🏆 强化学习 (PPO)
使用 TRL 进行 PPO 训练，实现人类反馈的强化学习（RLHF）。

```yaml
taskType: "ppo"
hyperparameters:
  reward_model: "OpenAssistant/reward-model-deberta-v3-large-v2"
  ppo_epochs: 4
  target_kl: 0.1
```

### 📚 检索增强生成 (RAG)
使用 LangChain 构建 RAG 系统，支持向量数据库和文档索引。

```yaml
taskType: "rag_inference"
hyperparameters:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  vector_db: "chromadb"
  retrieval_k: 5
```

### 🤖 智能代理 (Agent)
使用 LangChain 构建智能代理，支持工具调用和多轮对话。

```yaml
taskType: "agent_inference"
hyperparameters:
  agent_type: "react"
  tools: ["python_repl", "web_search"]
```

## 📝 提交任务

1. **准备任务配置**
   ```bash
   cp configs/sft_mistral_7b.yaml my_task.yaml
   # 编辑配置文件...
   ```

2. **提交任务**
   ```bash
   curl -X POST http://localhost:8000/api/v1/tasks \
     -H "Content-Type: application/json" \
     -d @my_task.yaml
   ```

3. **查看任务状态**
   ```bash
   curl http://localhost:8000/api/v1/tasks/<task_id>
   ```

## 📊 监控和统计

### 查看任务列表
```bash
curl http://localhost:8000/api/v1/tasks?page=1&page_size=10
```

### 查看工作节点
```bash
curl http://localhost:8000/api/v1/workers
```

### 获取用量统计
```bash
curl "http://localhost:8000/api/v1/stats?user=john-doe&start_date=2024-01-01"
```

## ⚙️ 配置管理

### 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `MLOC_NODE_TYPE` | `WORKER` | 节点类型 (`ORCHESTRATOR` 或 `WORKER`) |
| `MLOC_REDIS_URL` | `redis://localhost:6379` | Redis 连接 URL |
| `MLOC_LOG_LEVEL` | `INFO` | 日志级别 |
| `MLOC_HOST` | `0.0.0.0` | 服务监听地址 |
| `MLOC_PORT` | `8000` | 服务端口 |

### 本地开发

1. **安装依赖**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

2. **启动 Redis**
   ```bash
   docker run -d -p 6379:6379 redis:7-alpine
   ```

3. **启动 Orchestrator**
   ```bash
   mloc start --node-type orchestrator
   ```

4. **启动 Worker**
   ```bash
   mloc start --node-type worker
   ```

## 🔧 开发指南

### 添加新的任务模块

1. **创建模块类**
   ```python
   # src/mloc/modules/my_module.py
   from .base_module import BaseModule
   
   class MyModule(BaseModule):
       async def execute(self, progress_callback=None):
           # 实现任务逻辑
           pass
   ```

2. **注册模块**
   ```python
   # src/mloc/modules/__init__.py
   from .my_module import MyModule
   
   MODULE_REGISTRY[TaskType.MY_TASK] = MyModule
   ```

### 项目结构

```
mloc/
├── src/mloc/
│   ├── common/           # 通用工具和定义
│   ├── orchestrator/     # 主控节点实现
│   ├── worker/          # 工作节点实现
│   ├── modules/         # 任务执行模块
│   └── integrations/    # 外部库集成
├── configs/             # 示例配置文件
├── docker/              # Docker 构建文件
├── scripts/            # 部署和构建脚本
└── tests/              # 测试用例
```

## 🛠️ 依赖项目

- **[TRL](https://github.com/huggingface/trl)**: Transformer Reinforcement Learning
- **[vLLM](https://github.com/vllm-project/vllm)**: 高性能 LLM 推理引擎
- **[LangChain](https://github.com/langchain-ai/langchain)**: LLM 应用框架
- **[Hugging Face](https://huggingface.co/)**: 模型和数据集生态
- **[Redis](https://redis.io/)**: 消息队列和状态存储
- **[FastAPI](https://fastapi.tiangolo.com/)**: 现代 Web API 框架

## 📄 许可证

本项目采用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解更多信息。

## 📞 支持

- 📧 Email: support@mloc.dev
- 💬 Discord: [MLOC Community](https://discord.gg/mloc)
- 📖 Documentation: [docs.mloc.dev](https://docs.mloc.dev)