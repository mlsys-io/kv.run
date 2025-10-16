from __future__ import annotations

from typing import Any, Dict, List, Tuple


_MODEL_VALUE_KEYS = {
    "identifier",
    "model",
    "model_name",
    "base_model",
    "pretrained_model_name_or_path",
    "path",
}
_MODEL_PARENT_HINTS = {
    "model",
    "models",
    "source",
    "reward_model",
    "value_model",
    "ref_model",
    "embedding",
    "adapter",
    "adapters",
}

_DATA_VALUE_KEYS = {
    "dataset_name",
    "name",
    "url",
    "path",
}
_DATA_PRIMARY_PARENTS = {
    "data",
    "dataset",
    "datasets",
}
_DATA_PARENT_HINTS = _DATA_PRIMARY_PARENTS | {
    "training",
    "validation",
    "evaluation",
    "graph",
}


def _append_unique(bucket: List[str], seen: set[str], value: str) -> None:
    trimmed = value.strip()
    if not trimmed:
        return
    lowered = trimmed.lower()
    if lowered in seen:
        return
    seen.add(lowered)
    bucket.append(trimmed)


def _has_hint(parents: Tuple[str, ...], hints: set[str]) -> bool:
    return any(parent in hints for parent in parents)


def _should_collect_model(key: str, parents: Tuple[str, ...]) -> bool:
    if key not in _MODEL_VALUE_KEYS:
        return False
    return _has_hint(parents, _MODEL_PARENT_HINTS)


def _should_collect_dataset(key: str, parents: Tuple[str, ...]) -> bool:
    if key not in _DATA_VALUE_KEYS:
        return False
    if not _has_hint(parents, _DATA_PARENT_HINTS):
        return False
    if key == "name":
        if "template" in parents:
            return False
        if not _has_hint(parents, _DATA_PRIMARY_PARENTS):
            return False
    return True


def extract_model_dataset_names(task_payload: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Inspect parsed task payload and extract candidate model and dataset identifiers.

    Returns a tuple of (model_names, dataset_names) with duplicates removed.
    """
    if not isinstance(task_payload, dict):
        return [], []

    spec = task_payload.get("spec")
    if not isinstance(spec, dict):
        return [], []

    models: List[str] = []
    datasets: List[str] = []
    model_seen: set[str] = set()
    dataset_seen: set[str] = set()

    def walk(obj: Any, parents: Tuple[str, ...]) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    if _should_collect_model(key, parents):
                        _append_unique(models, model_seen, value)
                    if _should_collect_dataset(key, parents):
                        _append_unique(datasets, dataset_seen, value)
                elif isinstance(value, dict):
                    walk(value, parents + (key,))
                elif isinstance(value, list):
                    walk_list(value, parents + (key,))
        elif isinstance(obj, list):
            walk_list(obj, parents)

    def walk_list(items: List[Any], parents: Tuple[str, ...]) -> None:
        for item in items:
            if isinstance(item, dict):
                walk(item, parents)
            elif isinstance(item, list):
                walk_list(item, parents)

    walk(spec, tuple())
    return models, datasets
