from __future__ import annotations

"""Helpers for downloading stage artifacts required by training executors."""

import itertools
import os
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import requests

from .base_executor import ExecutionError


def resolve_jsonl_path(
    path_value: str,
    *,
    out_dir: Path,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
    logger=None,
) -> Path:
    """Ensure a JSONL reference is available locally and return the path.

    Supports local filesystem paths and HTTP(S) URLs. When a URL is provided the
    content is downloaded into ``out_dir / "inputs"`` before returning the local
    filename.
    """

    if not path_value:
        raise ExecutionError("JSONL path is empty")

    value = str(path_value).strip()
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        target_dir = (out_dir / "inputs").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(parsed.path).name or "dataset.jsonl"
        target_path = (target_dir / filename).resolve()
        request_headers = {str(k): str(v) for k, v in (headers or {}).items()}
        lower_keys = {k.lower() for k in request_headers}
        token = os.getenv("ORCHESTRATOR_TOKEN")
        if token and "authorization" not in lower_keys:
            request_headers["Authorization"] = f"Bearer {token}"
        try:
            with requests.get(value, headers=request_headers, timeout=timeout, stream=True) as response:
                response.raise_for_status()
                with target_path.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
        except requests.RequestException as exc:
            raise ExecutionError(f"Failed to download JSONL dataset from {value}: {exc}") from exc

        if logger:
            try:
                size = target_path.stat().st_size
            except OSError:
                size = None
            logger.info(
                "Downloaded JSONL dataset from %s to %s%s",
                value,
                target_path,
                f" ({size} bytes)" if size is not None else "",
            )
        return target_path

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        rel_candidate = (out_dir / candidate).resolve()
        if rel_candidate.exists():
            candidate = rel_candidate
        else:
            candidate = candidate.resolve(strict=False)

    if candidate.exists():
        return candidate

    search_paths = [candidate]
    base_dir = Path(out_dir).resolve()
    for parent in itertools.islice(base_dir.parents, 0, 2):
        guess = (parent / candidate.name).resolve()
        search_paths.append(guess)
        if guess.exists():
            return guess

    raise ExecutionError(f"JSONL dataset not found: {candidate}")
