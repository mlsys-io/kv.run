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
   Requirements now include per-device VRAM via
   `spec.resources.hardware.gpu.memory`; workers report their VRAM at
   registration so the pool automatically excludes nodes with insufficient
   memory. Workers disabled by the elastic coordinator are filtered out here.
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

### Stage affinity for fine-tunes
When `ENABLE_STAGE_WEIGHT_STICKINESS=true` the adaptive dispatcher inspects each
task's payload for `${stage.result...}` references under checkpoint/adapter
fields. If a downstream stage needs the parent weights and the original worker
is still alive, the dispatcher pins the child task to that worker to avoid
re-downloading and unpacking large archives. If the worker is busy but healthy,
the task is requeued with a `sticky_worker_busy` reason so metrics reflect the
deferred dispatch. If the worker disappears, the dispatcher gracefully falls
back to normal scoring.

## Dispatcher modes

The host exposes several mutually exclusive dispatchers so we can compare
baselines against the adaptive scheduler:

- `adaptive` (default) – capability-aware scoring with context reuse, task
  merging, and elastic worker filters as detailed above.
- `static_worker` – Static worker assignment baseline. Every YAML submission
  is bound to a single worker for its entire lifetime. Once any task from that
  workflow is running, the worker is removed from the general pool until all of
  its sibling tasks finish or fail, mimicking traditional serverless isolation.
- `fixed_pipeline` – Fixed pipeline assignment. Each task type observed is
  permanently assigned to the first compatible worker that picked it up. Tasks
  wait in the queue whenever their dedicated worker is busy or unavailable.
- `static_round_robin` – Round-robin scheduler. The dispatcher cycles across
  idle workers in Redis snapshot order without checking task requirements,
  providing a simple load-balancing baseline.

`ORCHESTRATOR_DISPATCH_MODE` selects the mode. In addition, you can combine
the dispatchers with the worker selection strategies:

- `ORCHESTRATOR_WORKER_SELECTION=first_fit` disables reordering and always picks
  the first compatible worker from Redis.
- `ORCHESTRATOR_WORKER_SELECTION=min_satisfying` favours the smallest-capacity
  worker that meets the task requirements, helping preserve high-end nodes for
  heavier jobs.

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
- `HOST_METRICS_ENABLE_DENSITY_PLOT` – when set to `true`, the orchestrator records
  task submission density and active worker counts, exporting the raw series to
  `${HOST_METRICS_DIR}/task_worker_density.json` (bucket size controlled by
  `HOST_METRICS_DENSITY_BUCKET_SEC`, default 60s).
  Use `pt/density_plot.py` to render the JSON into a figure when needed.

## Metrics instrumentation

Host-side retries/failures now go through shared helpers:

- `_requeue_task` records `TASK_REQUEUED` with a descriptive `reason`. This
  keeps `/metrics` and `metrics.json` in sync with actual dispatch decisions,
  even when the dispatcher loops waiting for a sticky worker or a publish retry.
- `_fail_task` wraps `TaskRuntime.mark_failed`, emitting `TASK_FAILED` for the
  primary task, any impacted dependents, and merged children. The payload
  includes `dependency_failure`/`parent_task_id` hints so cascades are easy to
  trace.
- Worker-originated `TASK_FAILED` events now trigger automatic retries up to
  `max_attempts` (per task). Only when the attempt budget is exhausted does the
  orchestrator mark the task as failed and emit the corresponding metrics event.
  Temporary dependency waits (`stage_reference_pending`) do not count toward the
  attempt budget or requeue counters.

Because the recorder is injected into every dispatcher variant, the counters stay
accurate regardless of `ORCHESTRATOR_DISPATCH_MODE`.

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
