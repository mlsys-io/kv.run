"""
Task Executor

This module implements the core task execution logic for workers.
It manages module loading, resource downloading, and result uploading.
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

import structlog

from mloc.common.constants import TaskType
from mloc.common.schemas import TaskConfig
from mloc.integrations.hf_downloader import HuggingFaceDownloader
from mloc.modules import get_module_class


logger = structlog.get_logger(__name__)


class TaskExecutor:
    """Task executor for running assigned tasks"""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.work_dir = Path("/tmp/mloc_work") / worker_id
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize downloaders
        self.hf_downloader = HuggingFaceDownloader()
    
    async def execute_task(
        self,
        task_id: str,
        config: TaskConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> None:
        """Execute a task from start to finish"""
        
        task_work_dir = self.work_dir / task_id
        task_work_dir.mkdir(exist_ok=True)
        
        logger.info("Starting task execution", 
                   task_id=task_id, 
                   task_type=config.spec.task_type,
                   work_dir=str(task_work_dir))
        
        try:
            # Step 1: Download resources (10% progress)
            if progress_callback:
                await progress_callback(task_id, 0.0)
            
            await self._download_resources(task_id, config, task_work_dir)
            
            if progress_callback:
                await progress_callback(task_id, 0.1)
            
            # Step 2: Load and execute module (10% - 90% progress)
            await self._execute_module(task_id, config, task_work_dir, progress_callback)
            
            # Step 3: Upload results (90% - 100% progress)
            if progress_callback:
                await progress_callback(task_id, 0.9)
            
            await self._upload_results(task_id, config, task_work_dir)
            
            if progress_callback:
                await progress_callback(task_id, 1.0)
            
            logger.info("Task execution completed successfully", task_id=task_id)
            
        except Exception as e:
            logger.error("Task execution failed", task_id=task_id, error=str(e))
            raise
        finally:
            # Clean up work directory
            await self._cleanup_work_dir(task_work_dir)
    
    async def _download_resources(
        self,
        task_id: str,
        config: TaskConfig,
        work_dir: Path
    ) -> None:
        """Download required resources (model, dataset, etc.)"""
        logger.info("Downloading resources", task_id=task_id)
        
        # Create directories
        model_dir = work_dir / "model"
        dataset_dir = work_dir / "dataset"
        artifacts_dir = work_dir / "artifacts"
        
        model_dir.mkdir(exist_ok=True)
        dataset_dir.mkdir(exist_ok=True)
        artifacts_dir.mkdir(exist_ok=True)
        
        # Download model
        model_config = config.spec.model
        if model_config.source.type == "huggingface":
            await self.hf_downloader.download_model(
                repo_id=model_config.source.identifier,
                local_dir=str(model_dir),
                revision=model_config.source.revision,
                token=model_config.source.access_token
            )
            logger.info("Model downloaded", repo_id=model_config.source.identifier)
        
        # Download dataset if specified
        dataset_config = config.spec.dataset
        if dataset_config and dataset_config.source.type == "huggingface":
            await self.hf_downloader.download_dataset(
                repo_id=dataset_config.source.identifier,
                local_dir=str(dataset_dir),
                split=dataset_config.split,
                token=dataset_config.source.access_token
            )
            logger.info("Dataset downloaded", repo_id=dataset_config.source.identifier)
    
    async def _execute_module(
        self,
        task_id: str,
        config: TaskConfig,
        work_dir: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> None:
        """Execute the task module"""
        logger.info("Executing task module", 
                   task_id=task_id,
                   task_type=config.spec.task_type)
        
        # Get module class
        module_class = get_module_class(config.spec.task_type)
        if not module_class:
            raise ValueError(f"Unsupported task type: {config.spec.task_type}")
        
        # Create module instance
        module = module_class(
            task_id=task_id,
            config=config,
            work_dir=work_dir
        )
        
        # Define progress wrapper
        async def module_progress_callback(progress: float):
            if progress_callback:
                # Map module progress (0-1) to overall progress (0.1-0.9)
                overall_progress = 0.1 + (progress * 0.8)
                await progress_callback(task_id, overall_progress)
        
        # Execute module
        await module.execute(progress_callback=module_progress_callback)
        
        logger.info("Task module execution completed", task_id=task_id)
    
    async def _upload_results(
        self,
        task_id: str,
        config: TaskConfig,
        work_dir: Path
    ) -> None:
        """Upload task results to specified destination"""
        logger.info("Uploading results", task_id=task_id)
        
        artifacts_dir = work_dir / "artifacts"
        output_config = config.spec.output
        
        if not artifacts_dir.exists():
            logger.warning("No artifacts directory found", task_id=task_id)
            return
        
        if output_config.destination.type == "s3":
            await self._upload_to_s3(
                artifacts_dir=artifacts_dir,
                bucket=output_config.destination.bucket,
                path=output_config.destination.path,
                artifacts=output_config.artifacts
            )
        elif output_config.destination.type == "huggingface":
            await self._upload_to_huggingface(
                artifacts_dir=artifacts_dir,
                repo_id=output_config.destination.path,
                artifacts=output_config.artifacts,
                token=output_config.destination.access_token
            )
        else:
            logger.warning("Unsupported output destination type",
                         type=output_config.destination.type)
    
    async def _upload_to_s3(
        self,
        artifacts_dir: Path,
        bucket: str,
        path: str,
        artifacts: list
    ) -> None:
        """Upload artifacts to S3"""
        # TODO: Implement S3 upload
        # This would use boto3 to upload files
        logger.warning("S3 upload not implemented yet")
    
    async def _upload_to_huggingface(
        self,
        artifacts_dir: Path,
        repo_id: str,
        artifacts: list,
        token: Optional[str] = None
    ) -> None:
        """Upload artifacts to Hugging Face Hub"""
        # TODO: Implement HF Hub upload
        # This would use huggingface_hub to push files
        logger.warning("Hugging Face upload not implemented yet")
    
    async def _cleanup_work_dir(self, work_dir: Path) -> None:
        """Clean up work directory after task completion"""
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir)
                logger.debug("Work directory cleaned up", work_dir=str(work_dir))
        except Exception as e:
            logger.warning("Failed to clean up work directory", 
                         work_dir=str(work_dir), error=str(e))