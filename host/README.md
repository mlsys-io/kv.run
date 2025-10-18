# Host Scheduler & Elastic Overview

The host (orchestrator) converts YAML submissions into runnable tasks, keeps
track of dependencies, and assigns work to workers through Redis. This document
summarises the scheduling heuristics and the elastic controls introduced in this
revision.

## Dispatch pipeline

1. **Ready queue** – tasks enter a FIFO queue once dependencies are satisfied.
   `TaskRuntime` exposes the queue length for the elastic manager.
2. **Candidate discovery** – for each task we build a pool of idle workers whose
   advertised hardware satisfies the task requirements (`idle_satisfying_pool`).
   Workers disabled by the elastic coordinator are filtered out here.
3. **Context reuse** – when `ENABLE_CONTEXT_REUSE=true`, workers that recently
   cached the required model/dataset are preferred. Cache hints carry a TTL
   (`WORKER_CACHE_TTL_SEC`, default 1h); stale hints automatically fall back to
   neutral priority so workers that refreshed their model are favoured.
4. **Worker scoring** – the adaptive dispatcher applies a convex combination
   score

   ```
   s_λ(v, w) = λ · μ̂(v, w) − (1 − λ) · ĉ(w)
   ```

   - `μ̂` (throughput proxy) is derived from GPU count/VRAM, system RAM, and CPU
     cores.
   - `ĉ` (cost proxy) uses the worker-reported `cost_per_hour` (falls back to 1).
   - `λ` defaults to 0.4 for inference, 0.8 for training, and 0.5 otherwise and
     can be tuned via `SCHEDULER_LAMBDA_*` environment variables.
   - A deterministic jitter (`SCHEDULER_SELECTION_JITTER`, default `1e-3`) based
     on `hash(task_id, worker_id)` breaks ties so the same candidate set rotates
     fairly across workers.
5. **Logging & observability** – every dispatch logs the candidate pool size,
   reuse hits, lambda, and the final score of the chosen worker. Metrics exports
   (`/metrics`) include elastic disabled workers so tuning is auditable.

The fallback strategies remain available:
- `ORCHESTRATOR_DISPATCH_MODE=fixed_pipeline` pins each task type to a single
  worker.
- `ORCHESTRATOR_DISPATCH_MODE=static_round_robin` performs FIFO assignment over
  idle workers.
- `ORCHESTRATOR_WORKER_SELECTION=first_fit` honours the Redis order without any
  re-sorting.

## Elastic coordinator

`ENABLE_ELASTIC_SCALING=true` activates background management that combines
manual toggles (via `POST /workers/{id}/elastic`) with automated heuristics:

| Env | Default | Meaning |
|-----|---------|---------|
| `ELASTIC_AUTO_DISABLE_IDLE_SEC` | `60` | Idle duration before disabling a worker when the queue length ≤ `ELASTIC_AUTO_DISABLE_QUEUE_MAX`. |
| `ELASTIC_AUTO_DISABLE_QUEUE_MAX` | `0` | Maximum ready-queue size that still allows auto-disable. |
| `ELASTIC_AUTO_ENABLE_QUEUE_THRESHOLD` | `0` | Queue length that triggers auto re-enable of disabled workers. |
| `ELASTIC_AUTO_MIN_ACTIVE_WORKERS` | `1` | Minimum enabled workers kept alive. |
| `ELASTIC_AUTO_POLL_INTERVAL_SEC` | `30` | Manager loop interval. |
| `ELASTIC_AUTO_TOGGLE_COOLDOWN_SEC` | `≥60` | Cooldown before the same worker can be toggled again. |

Disabled workers are skipped during candidate discovery and excluded from
aggregated cost/energy metrics. Once re-enabled, they immediately rejoin the
candidate pool.

## Environment knobs

A consolidated list of host-side environment variables is available in the
root `README.md`. Key additions related to the scheduler:

- `WORKER_CACHE_TTL_SEC` – reuse metadata expiry (seconds).
- `SCHEDULER_LAMBDA_INFERENCE`, `SCHEDULER_LAMBDA_TRAINING`, `SCHEDULER_LAMBDA_OTHER`
  – category-specific λ weights.
- `SCHEDULER_SELECTION_JITTER` – jitter magnitude used to break ties.

## Testing ideas

When extending the scheduler, consider the following targeted checks:

1. **Reuse priority** – craft two workers where only one advertises the cached
   model; ensure the dispatcher prefers it until the TTL lapses.
2. **Tie breaking** – present identical workers and observe the deterministic
   rotation induced by the jitter hash.
3. **Elastic recovery** – disable a worker, queue new work, and verify that
   crossing the enable threshold brings it back online.
4. **Strategy coverage** – set `ORCHESTRATOR_WORKER_SELECTION=first_fit` and
   confirm ordering matches the Redis snapshot.
5. **Mixed workloads** – submit both inference and PPO/SFT jobs and inspect the
   λ-driven score balancing between throughput and cost.

These scenarios can be automated with lightweight unit tests that mock the
Redis facade and worker registry, ensuring the heuristics remain stable over
future refactors.
