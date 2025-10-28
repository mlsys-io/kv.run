#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Executor using TRL's simple approach

Simple implementation using TRL's DPOTrainer with built-in train() method.
"""

import gc
import json
import os
import subprocess
import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

from .base_executor import Executor, ExecutionError
from .checkpoint_utils import (
    archive_model_dir,
    cleanup_artifact_path,
    get_http_destination,
    is_cleanup_enabled,
)

logger = logging.getLogger("worker.dpo")


class DPOExecutor(Executor):
    """DPO training executor using TRL library."""

    def __init__(self):
        super().__init__()
        self._model_name = None
        self._current_model = None
        self._current_ref_model = None
        self._current_tokenizer = None
        self._current_trainer = None
        self._task_out_dir: Optional[Path] = None

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        logger.info("Starting DPO training task")
        spec = (task or {}).get("spec") or {}
        training_config = spec.get("training", {}) or {}
        launcher_flag = "KV_DPO_DISTRIBUTED"
        already_spawned = os.environ.get(launcher_flag) == "1"
        gpu_count = self._detect_gpu_count(training_config)
        allow_multi_cfg = training_config.get("allow_multi_gpu")
        allow_multi = bool(allow_multi_cfg) if allow_multi_cfg is not None else gpu_count > 1

        self._task_out_dir = out_dir
        try:
            if allow_multi and not already_spawned and gpu_count > 1:
                self._spawn_distributed(task, out_dir, gpu_count, launcher_flag, training_config)
                resp_path = out_dir / "responses.json"
                if resp_path.exists():
                    return self.load_json(resp_path)
                return {
                    "training_successful": True,
                    "spawned_torchrun": True,
                    "model_name": (spec.get("model", {}).get("source", {}) or {}).get("identifier"),
                    "output_dir": str(out_dir),
                }

            result = self._execute_training(task, out_dir)
            logger.info(
                "DPO training task completed in %.2f seconds",
                result.get("training_time_seconds", 0.0),
            )
            return result
        finally:
            self._task_out_dir = None

    def _load_dataset(self, spec: Dict[str, Any]) -> Dataset:
        """Load training dataset in DPO format"""
        data_config = spec.get("data", {})

        def _ensure_jsonl_local(jsonl_cfg: Dict[str, Any], default_name: str) -> Path:
            path_value = str(jsonl_cfg.get("path") or "").strip()
            url_value = str(jsonl_cfg.get("url") or jsonl_cfg.get("download_url") or "").strip()

            candidate: Optional[Path] = None
            if path_value:
                candidate_path = Path(path_value).expanduser()
                if not candidate_path.is_absolute() and self._task_out_dir:
                    candidate_path = (self._task_out_dir / candidate_path).resolve()
                else:
                    candidate_path = candidate_path.resolve()
                if candidate_path.exists():
                    jsonl_cfg["path"] = str(candidate_path)
                    return candidate_path
                candidate = candidate_path

            if url_value:
                if self._task_out_dir:
                    cache_dir = (self._task_out_dir / "_jsonl_cache").resolve()
                else:
                    cache_dir = Path(tempfile.mkdtemp(prefix="mloc_jsonl_")).resolve()
                cache_dir.mkdir(parents=True, exist_ok=True)
                parsed = urlparse(url_value)
                name = Path(parsed.path).name or (candidate.name if candidate else default_name)
                if not name:
                    name = default_name
                target = cache_dir / name
                try:
                    import requests

                    with requests.get(url_value, stream=True, timeout=60) as resp:
                        resp.raise_for_status()
                        with target.open("wb") as fh:
                            for chunk in resp.iter_content(chunk_size=65536):
                                if chunk:
                                    fh.write(chunk)
                except Exception as exc:  # pragma: no cover
                    raise ExecutionError(f"Failed to download JSONL dataset from {url_value}: {exc}") from exc
                jsonl_cfg["path"] = str(target)
                return target.resolve()

            if candidate is not None:
                raise ExecutionError(f"JSONL dataset not found: {candidate}")
            raise ExecutionError("data.jsonl.path is required when using JSONL input")

        jsonl_cfg = data_config.get("jsonl")
        jsonl_path = data_config.get("jsonl_path")
        if jsonl_cfg or jsonl_path:
            if jsonl_cfg is None:
                jsonl_cfg = {}
            else:
                jsonl_cfg = dict(jsonl_cfg)
            if jsonl_path:
                jsonl_cfg.setdefault("path", jsonl_path)

            default_name = Path(str(jsonl_cfg.get("path") or "dataset.jsonl")).name or "dataset.jsonl"
            jsonl_file = _ensure_jsonl_local(jsonl_cfg, default_name=default_name)

            prompt_field = (
                jsonl_cfg.get("prompt_field")
                or data_config.get("prompt_field")
                or "prompt"
            )
            chosen_field = (
                jsonl_cfg.get("chosen_field")
                or data_config.get("chosen_field")
                or "chosen"
            )
            rejected_field = (
                jsonl_cfg.get("rejected_field")
                or data_config.get("rejected_field")
                or "rejected"
            )

            prompts: List[str] = []
            chosen: List[str] = []
            rejected: List[str] = []

            with jsonl_file.open("r", encoding="utf-8") as fh:
                for line_number, raw in enumerate(fh, start=1):
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ExecutionError(
                            f"Invalid JSON on line {line_number} of {jsonl_file}: {exc}"
                        ) from exc

                    if prompt_field not in record or chosen_field not in record or rejected_field not in record:
                        raise ExecutionError(
                            "JSONL record missing required fields. "
                            f"Expected keys: '{prompt_field}', '{chosen_field}', '{rejected_field}'"
                        )

                    prompts.append(str(record[prompt_field]))
                    chosen.append(str(record[chosen_field]))
                    rejected.append(str(record[rejected_field]))

            if not prompts:
                raise ExecutionError(f"JSONL dataset at {jsonl_file} is empty")

            dataset = Dataset.from_dict({
                "prompt": prompts,
                "chosen": chosen,
                "rejected": rejected,
            })

        elif "dataset_name" in data_config:
            # Load from Hugging Face datasets (like ultrafeedback_binarized)
            from datasets import load_dataset
            dataset_name = data_config["dataset_name"]
            split = data_config.get("split", "train")

            config_name = data_config.get("config_name")
            revision = data_config.get("revision")
            load_kwargs: Dict[str, Any] = {"split": split}
            if config_name:
                load_kwargs["name"] = config_name
            if revision:
                load_kwargs["revision"] = revision

            try:
                dataset = load_dataset(dataset_name, **load_kwargs)
            except ValueError as exc:
                should_retry = False
                if config_name and "config" in str(exc):
                    should_retry = True
                if config_name and "BuilderConfig" in str(exc):
                    should_retry = True
                if should_retry:
                    logger.warning(
                        "Failed to load %s with config %s (split=%s). Retrying with default config.",
                        dataset_name,
                        config_name,
                        split,
                    )
                    fallback_kwargs = {k: v for k, v in load_kwargs.items() if k != "name"}
                    fallback_split = split
                    prefix = split.split("[", 1)[0]
                    if prefix != config_name and config_name:
                        fallback_split = split.replace(prefix, config_name, 1)
                    fallback_kwargs["split"] = fallback_split
                    try:
                        dataset = load_dataset(dataset_name, **fallback_kwargs)
                    except ValueError:
                        fallback_kwargs["split"] = split
                        dataset = load_dataset(dataset_name, **fallback_kwargs)
                else:
                    raise

            if data_config.get("max_samples"):
                dataset = dataset.select(range(min(len(dataset), data_config["max_samples"])))

            prompt_col = data_config.get("prompt_column", "prompt")
            chosen_col = data_config.get("chosen_column", "chosen")
            rejected_col = data_config.get("rejected_column", "rejected")

            required_source_columns = {prompt_col, chosen_col, rejected_col}
            missing_columns = required_source_columns - set(dataset.column_names)
            if missing_columns:
                raise ExecutionError(
                    f"Dataset {dataset_name} missing required columns: {sorted(missing_columns)}"
                )

            rename_map = {}
            if prompt_col != "prompt":
                rename_map[prompt_col] = "prompt"
            if chosen_col != "chosen":
                rename_map[chosen_col] = "chosen"
            if rejected_col != "rejected":
                rename_map[rejected_col] = "rejected"

            for old_name, new_name in rename_map.items():
                dataset = dataset.rename_column(old_name, new_name)

            extra_columns = set(dataset.column_names) - {"prompt", "chosen", "rejected"}
            if extra_columns:
                logger.info(
                    "Removing unused columns from dataset %s: %s",
                    dataset_name,
                    ", ".join(sorted(extra_columns)),
                )
                dataset = dataset.remove_columns(sorted(extra_columns))
                
        elif "preferences" in data_config:
            # Use provided preference data directly
            preferences = data_config["preferences"]
            dataset = Dataset.from_dict({
                "prompt": [item["prompt"] for item in preferences],
                "chosen": [item["chosen"] for item in preferences],
                "rejected": [item["rejected"] for item in preferences],
            })
        else:
            # Default demo dataset with preference pairs
            demo_data = [
                {
                    "prompt": "Write a helpful response about Python programming:",
                    "chosen": "Python is a versatile programming language known for its readability and extensive libraries. It's great for beginners and widely used in data science, web development, and automation.",
                    "rejected": "Python is just another programming language. Nothing special about it."
                },
                {
                    "prompt": "Explain machine learning in simple terms:",
                    "chosen": "Machine learning is like teaching computers to recognize patterns and make predictions from data, similar to how humans learn from experience.",
                    "rejected": "Machine learning is complicated math stuff that computers do."
                },
                {
                    "prompt": "Give advice on learning to code:",
                    "chosen": "Start with a beginner-friendly language like Python, practice regularly with small projects, and don't be afraid to make mistakes - they're part of learning!",
                    "rejected": "Just memorize syntax and you'll be fine."
                },
                {
                    "prompt": "Describe the importance of documentation:",
                    "chosen": "Good documentation is essential for code maintainability, team collaboration, and helping future developers (including yourself) understand what the code does and why.",
                    "rejected": "Documentation is just extra work that slows down development."
                }
            ]
            
            dataset = Dataset.from_dict({
                "prompt": [item["prompt"] for item in demo_data],
                "chosen": [item["chosen"] for item in demo_data],
                "rejected": [item["rejected"] for item in demo_data],
            })

        return dataset

    def _execute_training(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        spec = (task or {}).get("spec") or {}
        training_config = spec.get("training", {}) or {}
        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        dataset: Optional[Dataset] = None

        final_model_path: Optional[Path] = None
        final_archive_path: Optional[Path] = None

        try:
            model_config = spec.get("model", {})
            model_source = model_config.get("source", {})
            self._model_name = model_source.get("identifier", "microsoft/DialoGPT-small")

            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._current_tokenizer = tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(self._model_name)
            ref_model = AutoModelForCausalLM.from_pretrained(self._model_name)
            self._current_model = model
            self._current_ref_model = ref_model

            logger.info("Models loaded: %s", self._model_name)

            dataset = self._load_dataset(spec)
            logger.info("Dataset loaded with %d samples", len(dataset))

            dpo_config = DPOConfig(
                learning_rate=float(training_config.get("learning_rate", 5e-7)),
                per_device_train_batch_size=int(training_config.get("batch_size", 4)),
                gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 1)),
                num_train_epochs=int(training_config.get("num_train_epochs", 1)),
                output_dir=str(checkpoint_dir),
                save_steps=int(training_config.get("save_freq", 500)),
                logging_steps=10,
            )

            try:
                dpo_trainer = DPOTrainer(
                    model=model,
                    ref_model=ref_model,
                    args=dpo_config,
                    train_dataset=dataset,
                    tokenizer=tokenizer,
                )
                self._current_trainer = dpo_trainer
            except TypeError as exc:
                if "unexpected keyword argument 'tokenizer'" in str(exc):
                    try:
                        dpo_trainer = DPOTrainer(
                            model=model,
                            ref_model=ref_model,
                            args=dpo_config,
                            train_dataset=dataset,
                            processing_class=tokenizer,
                        )
                        self._current_trainer = dpo_trainer
                    except TypeError:
                        dpo_trainer = DPOTrainer(
                            model=model,
                            ref_model=ref_model,
                            args=dpo_config,
                            train_dataset=dataset,
                        )
                        self._current_trainer = dpo_trainer
                else:
                    raise

            logger.info("Starting DPO training...")
            dpo_trainer.train()
            logger.info("DPO training completed")

            if training_config.get("save_model", True):
                try:
                    model_save_path = checkpoint_dir / "final_model"
                    dpo_trainer.save_model(model_save_path)
                    logger.info("Model saved to: %s", model_save_path)
                    final_model_path = model_save_path
                    destination = get_http_destination(task)
                    if destination:
                        try:
                            final_archive_path = archive_model_dir(model_save_path)
                            logger.info("Archived DPO model to %s for HTTP delivery", final_archive_path)
                        except Exception as arch_exc:
                            logger.warning("Failed to archive DPO model for upload: %s", arch_exc)
                except Exception as exc:  # pragma: no cover - best effort save
                    logger.warning("Failed to save model: %s", exc)

            training_time = time.time() - start_time
            results = {
                "training_successful": True,
                "training_time_seconds": training_time,
                "error_message": None,
                "model_name": self._model_name,
                "dataset_size": len(dataset),
                "output_dir": str(out_dir),
            }
            if final_model_path is not None:
                results["final_model_path"] = str(final_model_path)
            if final_archive_path is not None:
                results["final_model_archive"] = final_archive_path.name
                results["final_model_archive_path"] = str(final_archive_path)
                self._upload_model_archive(task, final_archive_path, results)
            self._cleanup_local_artifacts(
                task,
                checkpoint_dir,
                final_model_path,
                final_archive_path,
                results,
            )
            self.save_json(out_dir / "responses.json", results)
            return results
        except Exception as exc:
            training_time = time.time() - start_time
            results = {
                "training_successful": False,
                "training_time_seconds": training_time,
                "error_message": str(exc),
                "model_name": self._model_name,
                "dataset_size": len(dataset) if dataset is not None else 0,
                "output_dir": str(out_dir),
            }
            self.save_json(out_dir / "responses.json", results)
            logger.exception("DPO training failed: %s", exc)
            raise ExecutionError(results["error_message"] or "DPO training failed") from exc

    def _upload_model_archive(self, task: Dict[str, Any], archive_path: Path, payload: Dict[str, Any]) -> None:
        destination = get_http_destination(task)
        task_id = (task or {}).get("task_id")
        if not destination or not task_id:
            return
        upload_url = destination["url"].rstrip("/") + f"/{task_id}/files"
        base_url = destination["url"].rstrip("/")
        try:
            import requests

            file_size = archive_path.stat().st_size if archive_path.exists() else 0
            with archive_path.open("rb") as fh:
                files = {"file": (archive_path.name, fh, "application/zip")}
                headers = {
                    k: v
                    for k, v in (destination.get("headers") or {}).items()
                    if str(k).lower() != "content-type"
                }
                response = requests.post(upload_url, files=files, headers=headers, timeout=destination["timeout"])
                response.raise_for_status()
            download_url = f"{base_url}/{task_id}/files/{archive_path.name}"
            payload["final_model_archive_url"] = download_url
            payload["final_model_archive_bytes"] = file_size
            payload["final_model_archive_uploaded"] = True
            logger.info("Uploaded DPO model archive to %s (%d bytes)", upload_url, file_size)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("DPO model archive upload failed for %s: %s", task_id, exc)

    def _cleanup_local_artifacts(
        self,
        task: Dict[str, Any],
        checkpoint_dir: Path,
        final_model_path: Optional[Path],
        final_archive_path: Optional[Path],
        payload: Dict[str, Any],
    ) -> None:
        if not is_cleanup_enabled():
            logger.info(
                "Skipping DPO checkpoint cleanup for task %s because MODEL_CLEANUP_AFTER_UPLOAD is disabled",
                (task or {}).get("task_id"),
            )
            return
        destination = get_http_destination(task)
        if not destination:
            return
        uploaded = bool(payload.get("final_model_archive_uploaded"))
        remove_checkpoints = uploaded or final_model_path is None
        if not remove_checkpoints:
            logger.info(
                "Skipping DPO checkpoint cleanup for task %s because archive upload did not complete",
                (task or {}).get("task_id"),
            )
            return
        if uploaded:
            cleanup_artifact_path(final_model_path, logger=logger)
            cleanup_artifact_path(final_archive_path, logger=logger)
        cleanup_artifact_path(checkpoint_dir, logger=logger)
        if not checkpoint_dir.exists():
            logger.info(
                "Deleted local DPO checkpoints for task %s at %s",
                (task or {}).get("task_id"),
                checkpoint_dir,
            )

    def _spawn_distributed(
        self,
        task: Dict[str, Any],
        out_dir: Path,
        n_gpus: int,
        launcher_flag: str,
        training_config: Dict[str, Any],
    ) -> None:
        launcher_dir = out_dir / "_launcher"
        launcher_dir.mkdir(parents=True, exist_ok=True)
        task_file = launcher_dir / "task_spec.json"
        with task_file.open("w", encoding="utf-8") as fh:
            json.dump(task, fh)

        nproc = int(training_config.get("nproc_per_node", n_gpus))
        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(nproc),
            "-m",
            "worker.executors.dpo_dist_entry",
            str(task_file),
            str(out_dir),
        ]
        env = os.environ.copy()
        env[launcher_flag] = "1"
        try:
            repo_root = Path(__file__).resolve().parents[2]
            env["PYTHONPATH"] = f"{str(repo_root)}:" + env.get("PYTHONPATH", "")
        except Exception:
            pass
        logger.info(
            "Spawning torchrun for DPO: %s (CUDA_VISIBLE_DEVICES=%s)",
            " ".join(cmd),
            env.get("CUDA_VISIBLE_DEVICES"),
        )
        subprocess.check_call(cmd, env=env)

    @staticmethod
    def _detect_gpu_count(training_config: Dict[str, Any]) -> int:
        vis = training_config.get("visible_devices") or os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis:
            tokens = [dev.strip() for dev in str(vis).split(",") if dev.strip()]
            if tokens:
                return len(tokens)
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            pass
        return 0

    def cleanup_after_run(self) -> None:
        dropped_objects = []
        for attr in (
            "_current_trainer",
            "_current_model",
            "_current_ref_model",
            "_current_tokenizer",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                dropped_objects.append(obj)
            setattr(self, attr, None)

        for obj in dropped_objects:
            try:
                del obj
            except Exception:
                pass

        try:
            import torch

            dist = getattr(torch, "distributed", None)
            if dist is not None:
                try:
                    if dist.is_available() and dist.is_initialized():
                        dist.destroy_process_group()
                        logger.debug("Destroyed torch distributed process group during cleanup")
                except Exception:
                    logger.debug("Failed to destroy torch distributed process group", exc_info=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    logger.debug("torch.cuda.ipc_collect failed", exc_info=True)
                if hasattr(torch.cuda, "reset_peak_memory_stats"):
                    try:
                        for idx in range(torch.cuda.device_count()):
                            torch.cuda.reset_peak_memory_stats(idx)
                    except Exception:
                        logger.debug("Failed to reset CUDA peak memory stats", exc_info=True)
        except Exception:
            pass

        gc.collect()
