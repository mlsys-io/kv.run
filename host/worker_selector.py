from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional, Tuple

from .worker_registry import Worker

DEFAULT_WORKER_SELECTION = "best_fit"


def select_worker(
    pool: Iterable[Worker],
    strategy: str = DEFAULT_WORKER_SELECTION,
    *,
    logger=None,
    task_category: Optional[str] = None,
    lambda_overrides: Optional[Dict[str, float]] = None,
    task_id: Optional[str] = None,
    jitter_epsilon: float = 1e-3,
    task_age: Optional[float] = None,
) -> Tuple[Optional[Worker], Dict[str, object]]:
    """
    Select a worker from the candidate pool according to the scheduling strategy.

    Strategies:
    - best_fit (default): delegate to sort_workers for capacity-aware ordering.
    - first_fit: accept the first worker in the given iterable without reordering.
    - min_satisfying: pick the smallest-capacity worker that still satisfies the task.
    """

    candidates: List[Worker] = list(pool or [])
    if not candidates:
        return None, {"strategy": strategy, "reason": "empty_pool"}

    normalized = (strategy or DEFAULT_WORKER_SELECTION).strip().lower()

    if normalized == "first_fit":
        chosen = candidates[0]
        info = {
            "strategy": "first_fit",
            "candidate_count": len(candidates),
            "chosen": chosen.worker_id,
            "task_age": task_age,
        }
        return chosen, info

    if normalized == "min_satisfying":
        chosen, info = _select_min_capacity(
            candidates,
            task_id=task_id,
            jitter_epsilon=jitter_epsilon,
            task_age=task_age,
        )
        if chosen is None and logger:
            logger.debug("Min-capacity selection returned no worker; pool size=%d", len(candidates))
        return chosen, info

    if normalized != "best_fit":
        if logger:
            logger.warning(
                "Unknown worker selection strategy '%s'; falling back to best_fit",
                strategy,
            )
        normalized = DEFAULT_WORKER_SELECTION

    chosen, debug = _select_best_fit(
        candidates,
        task_category=task_category,
        lambda_overrides=lambda_overrides,
        task_id=task_id,
        jitter_epsilon=jitter_epsilon,
        task_age=task_age,
    )
    if chosen is None and logger:
        logger.debug("Best-fit selection returned no worker; pool size=%d", len(candidates))
    return chosen, debug


def _select_best_fit(
    candidates: List[Worker],
    *,
    task_category: Optional[str],
    lambda_overrides: Optional[Dict[str, float]],
    task_id: Optional[str],
    jitter_epsilon: float,
    task_age: Optional[float],
) -> Tuple[Optional[Worker], Dict[str, object]]:
    lambda_config = lambda_overrides or {}
    category_key = (task_category or "other").lower()
    lam = float(lambda_config.get(category_key, lambda_config.get("other", 0.5)))
    lam = max(0.0, min(1.0, lam))
    scores: List[Tuple[float, Worker, Dict[str, object]]] = []

    metric_payloads: List[Tuple[Worker, Dict[str, object]]] = []
    for worker in candidates:
        metric_payloads.append((worker, _collect_worker_metrics(worker)))

    if not metric_payloads:
        return None, {"strategy": "best_fit", "candidate_count": 0, "reason": "no_scores"}

    throughputs = [payload["throughput"] for _, payload in metric_payloads]
    costs = [payload["cost"] for _, payload in metric_payloads]
    throughput_min = min(throughputs)
    throughput_range = max(throughputs) - throughput_min
    cost_min = min(costs)
    cost_range = max(costs) - cost_min

    for worker, metrics in metric_payloads:
        throughput = metrics["throughput"]
        cost = metrics["cost"]
        norm_throughput = 0.0 if throughput_range <= 0 else (throughput - throughput_min) / throughput_range
        norm_cost = 0.0 if cost_range <= 0 else (cost - cost_min) / cost_range
        score = lam * norm_throughput - (1.0 - lam) * norm_cost
        if task_age is not None:
            score += min(task_age, 300.0) * 1e-4
        if task_id:
            score += _stable_jitter(task_id, worker.worker_id, jitter_epsilon)
        metrics["normalized_throughput"] = norm_throughput
        metrics["normalized_cost"] = norm_cost
        metrics["lambda"] = lam
        metrics["score"] = score
        scores.append((score, worker, metrics))

    scores.sort(key=lambda item: item[2]["worker_id"])  # stable by worker id
    scores.sort(key=lambda item: item[0], reverse=True)
    best_score, best_worker, best_metrics = scores[0]
    debug = {
        "strategy": "best_fit",
        "candidate_count": len(candidates),
        "chosen": best_worker.worker_id,
        "chosen_metrics": best_metrics,
        "task_age": task_age,
        "top_scores": [
            {
                "worker_id": metrics["worker_id"],
                "score": metrics["score"],
                "throughput": metrics["throughput"],
                "normalized_throughput": metrics.get("normalized_throughput"),
                "cost": metrics["cost"],
                "normalized_cost": metrics.get("normalized_cost"),
            }
            for _, _, metrics in scores[:5]
        ],
    }
    return best_worker, debug


def _select_min_capacity(
    candidates: List[Worker],
    *,
    task_id: Optional[str],
    jitter_epsilon: float,
    task_age: Optional[float],
) -> Tuple[Optional[Worker], Dict[str, object]]:
    scored: List[Tuple[float, float, str, Worker, Dict[str, object]]] = []
    for worker in candidates:
        metrics = _collect_worker_metrics(worker)
        adjusted_throughput = metrics["throughput"]
        if task_id:
            adjusted_throughput += _stable_jitter(task_id, worker.worker_id, jitter_epsilon)
        scored.append(
            (
                adjusted_throughput,
                metrics["cost"],
                worker.worker_id,
                worker,
                metrics,
            )
        )

    if not scored:
        return None, {"strategy": "min_satisfying", "candidate_count": 0, "reason": "no_scores"}

    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    adjusted, _, _, chosen_worker, chosen_metrics = scored[0]
    chosen_metrics = dict(chosen_metrics)
    chosen_metrics["adjusted_throughput"] = adjusted
    debug = {
        "strategy": "min_satisfying",
        "candidate_count": len(candidates),
        "chosen": chosen_worker.worker_id,
        "chosen_metrics": chosen_metrics,
        "task_age": task_age,
        "top_candidates": [
            {
                "worker_id": entry[4]["worker_id"],
                "throughput": entry[4]["throughput"],
                "adjusted_throughput": entry[0],
                "cost": entry[4]["cost"],
            }
            for entry in scored[:5]
        ],
    }
    return chosen_worker, debug


def _collect_worker_metrics(worker: Worker) -> Dict[str, float]:
    hardware = worker.hardware or {}
    gpus = hardware.get("gpu", {}).get("gpus") or hardware.get("gpus") or []
    if not isinstance(gpus, list):
        gpus = []
    gpu_count = len(gpus)
    total_vram = 0
    for gpu in gpus:
        mem = (
            gpu.get("memory", {}).get("total_bytes")
            if isinstance(gpu, dict)
            else None
        )
        if mem is None and isinstance(gpu, dict):
            mem = gpu.get("memory_bytes") or gpu.get("vram_bytes") or gpu.get("memory.total")
        try:
            total_vram += int(mem or 0)
        except Exception:
            pass
    sys_ram = hardware.get("memory", {}).get("total_bytes")
    try:
        sys_ram = int(sys_ram or 0)
    except Exception:
        sys_ram = 0
    cpu_cores = hardware.get("cpu", {}).get("logical_cores")
    try:
        cpu_cores = int(cpu_cores or 0)
    except Exception:
        cpu_cores = 0

    throughput = (
        gpu_count * 100.0
        + (total_vram or 0) / (1 << 30)
        + sys_ram / (1 << 31)
        + cpu_cores * 0.5
    )
    cost = worker.cost_per_hour if worker.cost_per_hour is not None else 1.0

    return {
        "worker_id": worker.worker_id,
        "throughput": throughput,
        "cost": cost,
        "gpu_count": float(gpu_count),
        "vram_gb": float((total_vram or 0) / (1 << 30)),
        "cpu_cores": float(cpu_cores),
        "sys_ram_gb": float(sys_ram / (1 << 30)),
    }


def _stable_jitter(task_id: str, worker_id: str, magnitude: float) -> float:
    if magnitude <= 0:
        return 0.0
    payload = f"{task_id}:{worker_id}".encode("utf-8", "ignore")
    digest = hashlib.md5(payload).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return (value - 0.5) * magnitude * 2
