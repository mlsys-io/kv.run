from __future__ import annotations

from typing import Type

from .dispatcher import Dispatcher
from .dispatcher_fixed_pipeline import FixedPipelineDispatcher
from .dispatcher_static_round_robin import StaticRoundRobinDispatcher
from .worker_selector import DEFAULT_WORKER_SELECTION

DEFAULT_DISPATCH_MODE = "adaptive"

_DISPATCHER_REGISTRY = {
    "adaptive": Dispatcher,
    "fixed_pipeline": FixedPipelineDispatcher,
    "static_round_robin": StaticRoundRobinDispatcher,
}


def create_dispatcher(
    mode: str,
    runtime,
    redis_client,
    results_dir,
    *,
    logger,
    worker_selection_strategy: str = DEFAULT_WORKER_SELECTION,
):
    """
    Instantiate a dispatcher according to the selected baseline.

    Supported modes:
    - adaptive (default): original capability-aware dispatcher
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
    return dispatcher_cls(
        runtime,
        redis_client,
        results_dir,
        logger=logger,
        worker_selection_strategy=worker_selection_strategy,
    )
