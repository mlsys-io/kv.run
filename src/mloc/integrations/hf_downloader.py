"""
Hugging Face Downloader

This module provides utilities for downloading models and datasets
from Hugging Face Hub.
"""

import asyncio
from pathlib import Path
from typing import Optional

import structlog
from huggingface_hub import snapshot_download
from datasets import load_dataset


logger = structlog.get_logger(__name__)


class HuggingFaceDownloader:
    """Downloader for Hugging Face models and datasets"""
    
    def __init__(self):
        self.logger = logger.bind(component="hf_downloader")
    
    async def download_model(
        self,
        repo_id: str,
        local_dir: str,
        revision: str = "main",
        token: Optional[str] = None
    ) -> str:
        """Download model from Hugging Face Hub"""
        
        self.logger.info(
            "Downloading model from Hugging Face",
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir
        )
        
        try:
            # Run the blocking download operation in a thread pool
            local_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    revision=revision,
                    token=token,
                    local_dir_use_symlinks=False,  # Use actual files, not symlinks
                    resume_download=True,
                )
            )
            
            self.logger.info(
                "Model downloaded successfully",
                repo_id=repo_id,
                local_path=local_path
            )
            
            return local_path
            
        except Exception as e:
            self.logger.error(
                "Failed to download model",
                repo_id=repo_id,
                error=str(e)
            )
            raise
    
    async def download_dataset(
        self,
        repo_id: str,
        local_dir: str,
        split: str = "train",
        token: Optional[str] = None
    ) -> str:
        """Download dataset from Hugging Face Hub"""
        
        self.logger.info(
            "Downloading dataset from Hugging Face",
            repo_id=repo_id,
            split=split,
            local_dir=local_dir
        )
        
        try:
            # Run the blocking download operation in a thread pool
            dataset_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._download_dataset_sync(repo_id, local_dir, split, token)
            )
            
            self.logger.info(
                "Dataset downloaded successfully",
                repo_id=repo_id,
                local_path=dataset_path
            )
            
            return dataset_path
            
        except Exception as e:
            self.logger.error(
                "Failed to download dataset",
                repo_id=repo_id,
                error=str(e)
            )
            raise
    
    def _download_dataset_sync(
        self,
        repo_id: str,
        local_dir: str,
        split: str,
        token: Optional[str]
    ) -> str:
        """Synchronous dataset download helper"""
        
        # Load dataset
        dataset = load_dataset(
            repo_id,
            split=split,
            token=token,
            trust_remote_code=True
        )
        
        # Save to disk
        dataset.save_to_disk(local_dir)
        
        return local_dir
    
    async def check_model_exists(self, repo_id: str, token: Optional[str] = None) -> bool:
        """Check if model exists on Hugging Face Hub"""
        try:
            from huggingface_hub import model_info
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model_info(repo_id, token=token)
            )
            return True
        except Exception:
            return False
    
    async def check_dataset_exists(self, repo_id: str, token: Optional[str] = None) -> bool:
        """Check if dataset exists on Hugging Face Hub"""
        try:
            from huggingface_hub import dataset_info
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: dataset_info(repo_id, token=token)
            )
            return True
        except Exception:
            return False