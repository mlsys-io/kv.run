"""
Task Scheduler

This module implements the core scheduling logic for the orchestrator.
It manages task queuing, worker assignment, and resource allocation.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

import aioredis
import structlog

from mloc.common.constants import (
    REDIS_KEYS, REDIS_TOPICS, TaskStatus, WorkerStatus, GPUType
)
from mloc.common.schemas import TaskConfig, TaskInfo, WorkerInfo, HardwareInfo
from mloc.common.utils import generate_task_id


logger = structlog.get_logger(__name__)


class TaskScheduler:
    """Task scheduler for managing task assignment to workers"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.scheduler_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self) -> None:
        """Start the scheduler"""
        self.redis = aioredis.from_url(self.redis_url)
        self.running = True
        
        # Start the scheduler loop
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("Task scheduler started")
    
    async def stop(self) -> None:
        """Stop the scheduler"""
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        if self.redis:
            await self.redis.close()
        
        logger.info("Task scheduler stopped")
    
    async def submit_task(self, config: TaskConfig) -> str:
        """Submit a new task"""
        task_id = generate_task_id()
        
        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            config=config,
            status=TaskStatus.PENDING,
            created_at=datetime.utcnow()
        )
        
        # Store task info
        await self.redis.hset(
            REDIS_KEYS["TASK_STATUS"],
            task_id,
            task_info.model_dump_json()
        )
        
        # Add to task queue
        await self.redis.lpush(
            REDIS_KEYS["TASK_QUEUE"],
            task_id
        )
        
        logger.info("Task submitted to queue", task_id=task_id)
        return task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        # Get current task info
        task_data = await self.redis.hget(REDIS_KEYS["TASK_STATUS"], task_id)
        if not task_data:
            return False
        
        task_info = TaskInfo.model_validate_json(task_data)
        
        # Only allow cancellation of pending or scheduled tasks
        if task_info.status not in [TaskStatus.PENDING, TaskStatus.SCHEDULED]:
            return False
        
        # Update task status
        task_info.status = TaskStatus.CANCELLED
        await self.redis.hset(
            REDIS_KEYS["TASK_STATUS"],
            task_id,
            task_info.model_dump_json()
        )
        
        # Remove from queue if still there
        await self.redis.lrem(REDIS_KEYS["TASK_QUEUE"], 0, task_id)
        
        logger.info("Task cancelled", task_id=task_id)
        return True
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        logger.info("Scheduler loop started")
        
        while self.running:
            try:
                await self._process_pending_tasks()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error("Error in scheduler loop", error=str(e))
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _process_pending_tasks(self) -> None:
        """Process pending tasks in the queue"""
        # Get next task from queue (blocking with timeout)
        result = await self.redis.brpop(REDIS_KEYS["TASK_QUEUE"], timeout=1)
        
        if not result:
            return  # No tasks available
        
        _, task_id = result
        task_id = task_id.decode('utf-8')
        
        logger.info("Processing task", task_id=task_id)
        
        # Get task info
        task_data = await self.redis.hget(REDIS_KEYS["TASK_STATUS"], task_id)
        if not task_data:
            logger.warning("Task not found in status store", task_id=task_id)
            return
        
        task_info = TaskInfo.model_validate_json(task_data)
        
        # Skip if task was cancelled
        if task_info.status == TaskStatus.CANCELLED:
            logger.info("Skipping cancelled task", task_id=task_id)
            return
        
        # Find suitable worker
        worker_id = await self._find_suitable_worker(task_info.config)
        
        if worker_id:
            # Assign task to worker
            await self._assign_task_to_worker(task_info, worker_id)
        else:
            # No suitable worker found, put task back in queue
            logger.info("No suitable worker found, requeueing task", task_id=task_id)
            await self.redis.lpush(REDIS_KEYS["TASK_QUEUE"], task_id)
    
    async def _find_suitable_worker(self, config: TaskConfig) -> Optional[str]:
        """Find a suitable worker for the task"""
        # Get all registered workers
        worker_ids = await self.redis.hkeys(REDIS_KEYS["WORKER_REGISTRY"])
        
        if not worker_ids:
            logger.debug("No workers registered")
            return None
        
        suitable_workers = []
        
        for worker_id in worker_ids:
            worker_id = worker_id.decode('utf-8')
            
            # Get worker info
            worker_data = await self.redis.hget(REDIS_KEYS["WORKER_REGISTRY"], worker_id)
            if not worker_data:
                continue
            
            worker_info = WorkerInfo.model_validate_json(worker_data)
            
            # Check if worker is idle
            if worker_info.status != WorkerStatus.IDLE:
                continue
            
            # Check resource requirements
            if self._check_resource_compatibility(config, worker_info.hardware):
                suitable_workers.append((worker_id, worker_info))
        
        if not suitable_workers:
            return None
        
        # Select best worker (simple strategy: first suitable)
        # Could be enhanced with more sophisticated scheduling algorithms
        selected_worker_id, _ = suitable_workers[0]
        
        logger.info("Selected worker for task", 
                   worker_id=selected_worker_id)
        
        return selected_worker_id
    
    def _check_resource_compatibility(self, config: TaskConfig, hardware: HardwareInfo) -> bool:
        """Check if worker hardware meets task requirements"""
        resources = config.spec.resources
        
        # Check CPU requirements
        try:
            required_cpu = int(resources.cpu)
            if hardware.cpu_count < required_cpu:
                return False
        except ValueError:
            # Handle cases like "4" or "2000m" CPU specifications
            pass
        
        # Check memory requirements
        required_memory_gb = self._parse_memory(resources.memory)
        if hardware.memory_gb < required_memory_gb:
            return False
        
        # Check GPU requirements
        if resources.gpu:
            if hardware.gpu_count < resources.gpu.count:
                return False
            
            # Check GPU type compatibility
            if resources.gpu.type != GPUType.ANY:
                if resources.gpu.type.value not in hardware.available_gpu_types:
                    return False
        
        return True
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory specification to GB"""
        memory_str = memory_str.upper()
        
        if memory_str.endswith('GI') or memory_str.endswith('GB'):
            return float(memory_str[:-2])
        elif memory_str.endswith('MI') or memory_str.endswith('MB'):
            return float(memory_str[:-2]) / 1024
        elif memory_str.endswith('G'):
            return float(memory_str[:-1])
        elif memory_str.endswith('M'):
            return float(memory_str[:-1]) / 1024
        else:
            # Assume it's in bytes
            return float(memory_str) / (1024 ** 3)
    
    async def _assign_task_to_worker(self, task_info: TaskInfo, worker_id: str) -> None:
        """Assign task to a specific worker"""
        # Update task status
        task_info.status = TaskStatus.SCHEDULED
        task_info.worker_id = worker_id
        
        await self.redis.hset(
            REDIS_KEYS["TASK_STATUS"],
            task_info.task_id,
            task_info.model_dump_json()
        )
        
        # Update worker status
        worker_data = await self.redis.hget(REDIS_KEYS["WORKER_REGISTRY"], worker_id)
        if worker_data:
            worker_info = WorkerInfo.model_validate_json(worker_data)
            worker_info.status = WorkerStatus.RUNNING
            worker_info.current_task_id = task_info.task_id
            
            await self.redis.hset(
                REDIS_KEYS["WORKER_REGISTRY"],
                worker_id,
                worker_info.model_dump_json()
            )
        
        # Send task assignment message
        task_assignment = {
            "task_id": task_info.task_id,
            "worker_id": worker_id,
            "config": task_info.config.model_dump(),
            "assigned_at": datetime.utcnow().isoformat()
        }
        
        await self.redis.publish(
            REDIS_TOPICS["TASK_ASSIGNMENT"],
            json.dumps(task_assignment)
        )
        
        logger.info("Task assigned to worker", 
                   task_id=task_info.task_id, 
                   worker_id=worker_id)