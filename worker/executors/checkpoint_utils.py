from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from .base_executor import ExecutionError


def determine_resume_path(
    spec: Dict[str, Any],
    training_cfg: Dict[str, Any],
    out_dir: Path,
    *,
    logger=None,
) -> Optional[Path]:
    checkpoint_cfg = (spec or {}).get("checkpoint") or {}
    load_cfg = checkpoint_cfg.get("load") or {}

    if load_cfg:
        return resolve_checkpoint_load(load_cfg, out_dir, logger=logger)

    resume_from = training_cfg.get("resume_from_path")
    if resume_from:
        path = Path(str(resume_from)).expanduser()
        if not path.exists():
            raise ExecutionError(f"Checkpoint path {path} not found for resume")
        return path
    return None


def resolve_checkpoint_load(load_cfg: Dict[str, Any], out_dir: Path, *, logger=None) -> Path:
    source_type = str(load_cfg.get("type", "local")).lower()
    # Local path hint (works for any source type)
    hinted_path = load_cfg.get("path")
    if hinted_path:
        resolved = Path(str(hinted_path)).expanduser()
        if resolved.exists():
            log = logger or load_cfg.get("_logger")
            if log:
                log.info("Resolved checkpoint from local path: %s", resolved)
            return resolved

    task_hint = load_cfg.get("taskId") or load_cfg.get("task_id")
    if task_hint:
        task_str = str(task_hint).strip()
        if task_str:
            base_dir = Path(os.getenv("RESULTS_DIR", "./results_workers")).expanduser().resolve()
            guesses = [
                base_dir / task_str / "final_model",
                base_dir / task_str / "final_model.bin",
                base_dir / task_str / "final_model.safetensors",
                base_dir / task_str / "final_model.pt",
                base_dir / task_str,
            ]
            for candidate in guesses:
                if candidate.exists():
                    log = logger or load_cfg.get("_logger")
                    if log:
                        log.info("Resolved checkpoint from local task cache: %s", candidate)
                    return candidate

    if source_type == "local":
        local_path = load_cfg.get("path") or load_cfg.get("uri")
        if not local_path:
            raise ExecutionError("checkpoint.load.path is required for local checkpoints")
        resolved = Path(str(local_path)).expanduser()
        if not resolved.exists():
            raise ExecutionError(f"Checkpoint path {resolved} not found")
        log = logger or load_cfg.get("_logger")
        if log:
            log.info("Using local checkpoint path: %s", resolved)
        return resolved

    if source_type == "http":
        url = load_cfg.get("url")
        if not url:
            task_id = load_cfg.get("task_id")
            filename = load_cfg.get("filename")
            base_url = load_cfg.get("base_url") or os.getenv("ORCHESTRATOR_BASE_URL")

            if base_url and task_id and filename:
                from urllib.parse import urljoin

                base = base_url.rstrip("/") + "/"
                artifact_path = f"api/v1/results/{task_id}/files/{filename}"
                load_cfg = dict(load_cfg)
                load_cfg["url"] = urljoin(base, artifact_path)
            else:
                raise ExecutionError(
                    "checkpoint.load.url is required for HTTP checkpoints (set base_url or ORCHESTRATOR_BASE_URL)"
                )
        log = logger or load_cfg.get("_logger")
        if log:
            log.info("Downloading checkpoint from %s", load_cfg.get("url"))
        return download_and_unpack(load_cfg, out_dir)

    raise ExecutionError(f"Unsupported checkpoint load type '{source_type}'")


def download_and_unpack(load_cfg: Dict[str, Any], out_dir: Path) -> Path:
    url = load_cfg.get("url")
    if not url:
        raise ExecutionError("checkpoint.load.url is required for HTTP checkpoints")

    headers = {str(k): str(v) for k, v in (load_cfg.get("headers") or {}).items()}
    timeout = float(load_cfg.get("timeoutSec", 60))
    dest_root = out_dir / "imported_checkpoint"
    if dest_root.exists():
        shutil.rmtree(dest_root, ignore_errors=True)
    dest_root.mkdir(parents=True, exist_ok=True)

    provided_name = load_cfg.get("filename")
    if provided_name:
        temp_name = str(provided_name)
    else:
        from urllib.parse import urlparse

        parsed_name = Path(urlparse(url).path).name
        temp_name = parsed_name or "checkpoint"

    temp_path = (dest_root / temp_name).resolve()
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temp_path.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    except urllib.error.URLError as exc:  # pragma: no cover - network failures
        raise ExecutionError(f"Failed to download checkpoint from {url}: {exc}") from exc

    archive_format = str(load_cfg.get("archive_format", "auto")).lower()
    if archive_format == "auto":
        if tarfile.is_tarfile(temp_path):
            fmt = "tar"
        else:
            fmt = "none"
    else:
        fmt = detect_archive_format(archive_format, temp_path.name)
    if fmt == "none":
        return temp_path

    extracted_dir = dest_root / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    try:
        if fmt == "zip":
            with zipfile.ZipFile(temp_path, "r") as archive:
                archive.extractall(extracted_dir)
        else:
            mode = "r:*" if fmt == "auto" else "r:*"
            with tarfile.open(temp_path, mode) as archive:
                archive.extractall(extracted_dir)
    except (tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ExecutionError(f"Failed to unpack checkpoint archive: {exc}") from exc

    subdir = load_cfg.get("subdir")
    candidate = select_extracted_subdir(extracted_dir, subdir)
    if not candidate.exists():
        raise ExecutionError(f"Extracted checkpoint path {candidate} does not exist")
    return candidate


def detect_archive_format(requested: str, filename: str) -> str:
    if requested in {"zip", "none"}:
        return requested
    if requested not in {"auto", "tar", "tar.gz", "tgz", "tar.bz2", "tar.xz"}:
        requested = "auto"

    lower_name = filename.lower()
    if requested == "zip" or lower_name.endswith(".zip"):
        return "zip"
    if lower_name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        return "tar"
    if requested == "none":
        return "none"
    return "tar"


def select_extracted_subdir(extracted_dir: Path, subdir: Optional[str]) -> Path:
    if subdir:
        return extracted_dir / subdir

    entries = [p for p in extracted_dir.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted_dir


def archive_model_dir(model_dir: Path, *, compression_level: Optional[int] = None) -> Path:
    archive_path = model_dir.parent / f"{model_dir.name}.tar.gz"
    if archive_path.exists():
        archive_path.unlink()

    level = _resolve_compression_level(compression_level)
    pigz_enabled = _should_use_pigz()
    pigz_binary = shutil.which(os.getenv("MODEL_ARCHIVE_PIGZ_BIN", "pigz")) if pigz_enabled else None
    tar_binary = shutil.which(os.getenv("MODEL_ARCHIVE_TAR_BIN", "tar")) if pigz_binary else None

    if pigz_binary and tar_binary:
        temp_tar = archive_path.with_suffix(".tar")
        if temp_tar.exists():
            temp_tar.unlink()
        try:
            subprocess.run(
                [
                    tar_binary,
                    "-C",
                    str(model_dir.parent),
                    "-cf",
                    str(temp_tar),
                    model_dir.name,
                ],
                check=True,
            )

            pigz_cmd = [pigz_binary, "-f"]
            pigz_threads = os.getenv("MODEL_ARCHIVE_PIGZ_THREADS")
            if pigz_threads:
                pigz_cmd.extend(["-p", pigz_threads])
            if level is not None:
                pigz_cmd.append(f"-{level}")
            pigz_cmd.append(str(temp_tar))
            subprocess.run(pigz_cmd, check=True)

            compressed_path = temp_tar.with_suffix(".tar.gz")
            compressed_path.rename(archive_path)
            return archive_path
        except Exception:
            # Fall back to python tarfile implementation below
            if temp_tar.exists():
                temp_tar.unlink(missing_ok=True)

    tarfile_kwargs: Dict[str, Any] = {"mode": "w:gz"}
    if level is not None:
        tarfile_kwargs["compresslevel"] = level
    with tarfile.open(archive_path, **tarfile_kwargs) as tf:
        tf.add(model_dir, arcname=model_dir.name)
    return archive_path


def _resolve_compression_level(level: Optional[int]) -> Optional[int]:
    if level is not None:
        return _clamp_compression(level)
    env_value = os.getenv("MODEL_ARCHIVE_COMPRESSION_LEVEL")
    if env_value is None:
        return None
    try:
        return _clamp_compression(int(env_value))
    except ValueError:
        return None


def _clamp_compression(level: int) -> int:
    return max(0, min(level, 9))


def _should_use_pigz() -> bool:
    setting = os.getenv("MODEL_ARCHIVE_USE_PIGZ")
    if setting is None:
        return True
    normalized = setting.strip().lower()
    return normalized not in {"0", "false", "no", "off"}


def get_http_destination(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    dest = (((task or {}).get("spec") or {}).get("output") or {}).get("destination") or {}
    if str(dest.get("type", "local")).lower() != "http":
        return None
    url = dest.get("url")
    if not url:
        return None
    return {
        "url": url,
        "headers": {str(k): str(v) for k, v in (dest.get("headers") or {}).items()},
        "timeout": float(dest.get("timeoutSec", 30)),
    }


def cleanup_artifact_path(path: Optional[Path], *, logger=None) -> None:
    """Remove a local file or directory if it exists, ignoring missing paths."""
    if not path:
        return
    try:
        target = Path(path)
    except TypeError:
        return
    try:
        if not target.exists():
            return
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    except FileNotFoundError:
        return
    except Exception as exc:  # pragma: no cover - best effort cleanup
        if logger:
            logger.warning("Failed to remove artifact at %s: %s", target, exc)


def is_cleanup_enabled() -> bool:
    setting = os.getenv("MODEL_CLEANUP_AFTER_UPLOAD")
    if setting is None:
        return False
    normalized = setting.strip().lower()
    return normalized not in {"0", "false", "no", "off"}
