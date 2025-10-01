# Experiment Templates

All templates under `exp/` use two-digit numeric prefixes that mirror the sections below and target OpenAI's `gsm8k` dataset. Inference configs consume 200 rows (`train[:200]`) and training-oriented configs use 100 rows (or 60/40 splits) to keep jobs lightweight. Multi-GPU support is enabled wherever the executor allows it.

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
