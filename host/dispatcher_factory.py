from __future__ import annotations

from typing import Optional, Type

from .dispatcher import Dispatcher
from .dispatcher_fixed_pipeline import FixedPipelineDispatcher
from .dispatcher_static_worker import StaticWorkerDispatcher
from .dispatcher_static_round_robin import StaticRoundRobinDispatcher
from .worker_selector import DEFAULT_WORKER_SELECTION

DEFAULT_DISPATCH_MODE = "adaptive"

_DISPATCHER_REGISTRY = {
    "adaptive": Dispatcher,
    "fixed_pipeline": FixedPipelineDispatcher,
    "static_round_robin": StaticRoundRobinDispatcher,
    "static_worker": StaticWorkerDispatcher,
}


def create_dispatcher(
    mode: str,
    runtime,
    redis_client,
    results_dir,
    *,
    logger,
    worker_selection_strategy: str = DEFAULT_WORKER_SELECTION,
    enable_context_reuse: bool = True,
    enable_task_merge: bool = True,
    task_merge_max_batch_size: int = 4,
    elastic_coordinator=None,
    reuse_cache_ttl_sec: int = 3600,
    lambda_config: Optional[dict] = None,
    selection_jitter_epsilon: float = 1e-3,
):
    """
    Instantiate a dispatcher according to the selected baseline.

    Supported modes:
    - adaptive (default): original capability-aware dispatcher
    - static_worker: assigns an entire submission to a single worker
    - fixed_pipeline: locks onto the first observed task type
    - static_round_robin: simple round-robin scheduler ignoring capabilities
    """

    normalized = (mode or DEFAULT_DISPATCH_MODE).strip().lower()
    if normalized not in _DISPATCHER_REGISTRY:
        logger.warning(
            "Unknown dispatcher mode '%s'; falling back to adaptive dispatcher",
            mode or "<empty>",
        )
        normalized = DEFAULT_DISPATCH_MODE

    dispatcher_cls: Type[Dispatcher] = _DISPATCHER_REGISTRY[normalized]
    logger.info("Initializing dispatcher in %s mode", normalized)
    base_kwargs = {
        "runtime": runtime,
        "redis_client": redis_client,
        "results_dir": results_dir,
        "logger": logger,
        "worker_selection_strategy": worker_selection_strategy,
    }
    if dispatcher_cls is Dispatcher:
        base_kwargs.update(
            enable_context_reuse=enable_context_reuse,
            enable_task_merge=enable_task_merge,
            task_merge_max_batch_size=task_merge_max_batch_size,
            elastic_coordinator=elastic_coordinator,
            reuse_cache_ttl_sec=reuse_cache_ttl_sec,
            lambda_config=lambda_config,
            selection_jitter_epsilon=selection_jitter_epsilon,
        )
    return dispatcher_cls(**base_kwargs)
