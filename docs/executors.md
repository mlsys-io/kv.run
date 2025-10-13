# 执行器矩阵与依赖分层（2025-10-09，编写者：Codex）

本文档汇总 worker 侧可用的执行器、对应的 `taskType`、需要安装的可选依赖以及产出规范，便于在不同场景下选择合适的安装包和模板。

## 1. 快速对照表

| taskType        | 执行器类              | 依赖分层（extras）    | 说明 |
| --------------- | --------------------- | --------------------- | ---- |
| `echo`          | `EchoExecutor`           | 内置，无需额外依赖        | 简易验证执行器，用于冒烟测试与脚本校验，输出原样回显。 |
| `inference`     | `HFTransformersExecutor` | `mloc[inference]`      | 默认 CPU 推理路径，落盘 `responses.json`；若环境具备 GPU 可同时启用 `vllm`。 |
| `inference`     | `VLLMExecutor`        | `mloc[inference]` + GPU | 需要可用的 CUDA/GPU，自动在运行时检测。 |
| `rag`           | `RAGExecutor`         | `mloc[rag]`            | 依赖 Qdrant/fastembed，查询结果写入 `artifacts/`。 |
| `agent`         | `AgentExecutor`       | `mloc[agent]`          | 集成 youtu-agent，要求 OpenAI/Google 等 API 凭据。 |
| `sft`           | `SFTExecutor`         | `mloc[training]`       | 基于 TRL 的监督微调，会在 `artifacts/` 输出 checkpoint。 |
| `lora_sft`      | `LoRASFTExecutor`     | `mloc[training]`       | LoRA 增量训练，产出 LoRA 权重与日志。 |
| `ppo`           | `PPOExecutor`         | `mloc[training]`       | 需要 GPU；依赖 `ray`/`trl`/`deepspeed`。 |
| `dpo`           | `DPOExecutor`         | `mloc[training]`       | 同上，执行 DPO 流程。 |

> **提示**：CPU 基线部署仅需要安装 `pip install mloc[inference]`。其余 extras 可按需组合，例如 `pip install mloc[inference,rag]`。

## 2. 目录与 Manifest 规范

所有执行器按照以下结构写入结果目录：

```
<RESULTS_DIR>/<task_id>/
├── responses.json           # 主结果
├── manifest.json            # orchestrator.manifest_utils 生成的索引
├── logs/                    # 执行日志（可为空）
└── artifacts/               # 额外工件，如 checkpoint、检索数据等
```

`manifest.json` 会根据模板中声明的 `spec.output.artifacts` 自动更新状态：

- 若某个工件尚未生成，manifest 中 `status` 标记为 `missing`，便于调试。
- HTTP 上传接口会将附件写入 `artifacts/` 并同步 manifest。
- 聚合任务在父、子任务完成时均会刷新 manifest，保证幂等性。

## 3. 安装示例

```bash
# 仅运行 orchestrator + CPU worker
pip install mloc[inference]

# 启用 RAG 与 Agent 流水线
pip install mloc[inference,rag,agent]

# 完整训练栈（SFT / LoRA / PPO / DPO）
pip install mloc[inference,training]
```

## 4. 运行时降级策略

- `worker/executors/__init__.py` 采用按需导入策略，缺失依赖时会记录在 `IMPORT_ERRORS` 并由 `worker/main.py` 打印日志。
- GPU 相关执行器会在初始化时检测 `torch.cuda.is_available()`，若不可用则自动跳过。
- 若核心 `HFTransformersExecutor` 缺失，worker 启动会直接中止并提示安装 `mloc[inference]` extras。

## 5. 模板编写注意事项

1. 明确 `spec.taskType` 与表格中的执行器映射，避免拼写错误。
2. 在 `spec.output.artifacts` 中列出预期产物名称（例如 `artifacts/checkpoint.pt`、`logs`）。manifest 会依据该列表进行校验。
3. 对于数据并行任务，确保 `spec.parallel.enabled` 设置正确；聚合器会验证所有分片结果是否到齐。

---

如需扩展新的执行器，请复用本表的约定：为新执行器定义专属 extras、补充模板示例，并在代码中通过 `EXECUTOR_REGISTRY` 登记以获得统一的降级行为。
