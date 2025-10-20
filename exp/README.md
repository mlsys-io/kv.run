# Experiment Templates

Top-level templates under `exp/` use two-digit numeric prefixes that mirror the sections below and target OpenAI's `gsm8k` dataset. Inference configs consume 200 rows (`train[:200]`) and training-oriented configs use 100 rows (or 60/40 splits) to keep jobs lightweight. Multi-GPU support is enabled wherever the executor allows it.

## Inference
- `exp/01_inference_single.yaml` — Single-turn vLLM inference with `Meta-Llama-3.1-8B-Instruct`, tensor parallel size 2.
- `exp/02_inference_multi_stage.yaml` — Three-step draft/refine/format pipeline, each stage runs vLLM on two GPUs.
- `exp/03_inference_dag.yaml` — DAG with two parallel solution branches feeding a synthesis node.

## SFT Training
- `exp/04_lora_sft_single_turn.yaml` — Single-turn LoRA SFT on `Llama-3.2-3B-Instruct`, 100 samples, DeepSpeed-style multi-GPU toggle.
- `exp/05_lora_sft_multi_turn.yaml` — Two-stage LoRA SFT to mimic multi-turn tutoring (60 + 40 samples).
- `exp/06_sft_full_single_turn.yaml` — Full-parameter SFT on 100 GSM8K questions with gradient checkpointing enabled.

## RLHF Training
- `exp/07_dpo_training.yaml` — DPO fine-tuning on GSM8K-inspired preference pairs (extendable to 100 pairs).
- `exp/08_ppo_training.yaml` — PPO loop over 100 GSM8K prompts with a lightweight reward model.

## Miscellaneous
- `exp/09_agent_query_search.yaml` — Search agent that gathers external hints for GSM8K-style reasoning.
- `exp/10_rag_pipeline.yaml` — Qdrant-based RAG lookup using GSM8K questions as queries.

## Compound Example
- `exp/11_lora_sft_plus_inference.yaml` — Multi-document template: first train a LoRA adapter (100 samples) then run vLLM inference (200 samples) that merges the emitted adapter archive.

## Composite Llama-1B Pipelines
- `exp/composite_llama1b/01_multi_dataset_linear_inference.yaml` — Sequential GSM8K → TruthfulQA → MMLU inference to compare cross-domain behavior.
- `exp/composite_llama1b/02_reasoning_chain_self_check.yaml` — Draft, critique, and rewrite flow that probes GSM8K self-consistency.
- `exp/composite_llama1b/03_sft_then_inference.yaml` — Lightweight GSM8K SFT with automatic TruthfulQA evaluation using the exported checkpoint.
- `exp/composite_llama1b/04_lora_then_inference.yaml` — MathQA LoRA adapter training followed by merged-weight GSM8K inference.
- `exp/composite_llama1b/05_dpo_plus_inference_report.yaml` — DPO alignment, evaluation briefing, and TruthfulQA follow-up pass.
- `exp/composite_llama1b/06_ppo_plus_mtbench_inference.yaml` — PPO reward shaping with subsequent MT-Bench style inference guidance.

## Batch submission helpers

Use `exp/run/submit_tasks.sh` to post _N_ YAML templates to the host orchestrator.
Files are sampled uniformly at random without replacement on each invocation:

```bash
cd exp/run
./submit_tasks.sh 3  # submits 3 distinct tasks (per run) in random order
```

For rate-limited randomized workloads, `exp/run/submit_random_tasks.sh` submits at a
configurable rate (requests per second) while sampling with replacement:

```bash
cd exp/run
./submit_random_tasks.sh 10 0.5  # 10 requests at 0.5 req/s
```

Override `HOST_URL`, `TASK_ENDPOINT`, or `TOKEN` to target different hosts or credentials.
