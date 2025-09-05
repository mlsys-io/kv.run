"""
Base Module for Task Execution

This module defines the abstract base class for all task execution modules.
All specific task modules (SFT, PPO, RAG, etc.) should inherit from this class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import structlog

from mloc.common.schemas import TaskConfig


logger = structlog.get_logger(__name__)


class BaseModule(ABC):
    """Abstract base class for all task execution modules"""
    
    def __init__(self, task_id: str, config: TaskConfig, work_dir: Path):
        self.task_id = task_id
        self.config = config
        self.work_dir = work_dir
        
        # Commonly used directories
        self.model_dir = work_dir / "model"
        self.dataset_dir = work_dir / "dataset"
        self.artifacts_dir = work_dir / "artifacts"
        
        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(exist_ok=True)
        
        self.logger = logger.bind(
            task_id=task_id,
            task_type=config.spec.task_type,
            module=self.__class__.__name__
        )
    
    @abstractmethod
    async def execute(
        self,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> None:
        """
        Execute the task.
        
        Args:
            progress_callback: Optional callback to report progress (0.0 to 1.0)
        """
        pass
    
    async def _report_progress(
        self,
        progress: float,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> None:
        """Report progress to callback if provided"""
        if progress_callback:
            await progress_callback(max(0.0, min(1.0, progress)))
        
        self.logger.debug("Progress update", progress=progress)
    
    def _validate_config(self) -> None:
        """Validate module-specific configuration"""
        # Base validation - can be overridden by subclasses
        if not self.model_dir.exists():
            raise ValueError(f"Model directory not found: {self.model_dir}")
        
        self.logger.info("Configuration validated")
    
    def _prepare_environment(self) -> None:
        """Prepare the execution environment"""
        # Set up environment variables, CUDA settings, etc.
        # Can be overridden by subclasses for specific needs
        
        import os
        import torch
        
        # Set CUDA memory management
        if torch.cuda.is_available():
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            self.logger.info(
                "GPU environment prepared",
                gpu_count=torch.cuda.device_count(),
                current_device=torch.cuda.current_device() if torch.cuda.is_available() else None
            )
        else:
            self.logger.info("CPU-only environment prepared")
    
    def _save_metadata(self, metadata: dict) -> None:
        """Save task execution metadata"""
        import json
        
        metadata_file = self.artifacts_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info("Metadata saved", metadata_file=str(metadata_file))
    
    def _log_system_info(self) -> None:
        """Log system information for debugging"""
        import torch
        import psutil
        
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(),
            })
        
        self.logger.info("System information", **system_info)