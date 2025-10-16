from __future__ import annotations

from typing import Iterable, List, Optional

from .worker_registry import Worker, sort_workers

DEFAULT_WORKER_SELECTION = "best_fit"


def select_worker(
    pool: Iterable[Worker],
    strategy: str = DEFAULT_WORKER_SELECTION,
    *,
    logger=None,
) -> Optional[Worker]:
    """
    Select a worker from the candidate pool according to the scheduling strategy.

    Strategies:
    - best_fit (default): delegate to sort_workers for capacity-aware ordering.
    - first_fit: accept the first worker in the given iterable without reordering.
    """

    candidates: List[Worker] = list(pool or [])
    if not candidates:
        return None

    normalized = (strategy or DEFAULT_WORKER_SELECTION).strip().lower()

    if normalized == "first_fit":
        return candidates[0]

    if normalized != "best_fit":
        if logger:
            logger.warning(
                "Unknown worker selection strategy '%s'; falling back to best_fit",
                strategy,
            )
        normalized = DEFAULT_WORKER_SELECTION

    ordered = sort_workers(candidates)
    if not ordered:
        return None
    return ordered[0]

