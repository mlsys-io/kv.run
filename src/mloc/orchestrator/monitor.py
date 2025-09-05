"""
State and Usage Monitor

This module implements monitoring functionality for the orchestrator.
It tracks worker status, task progress, and collects usage statistics.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import aioredis
import structlog

from mloc.common.constants import REDIS_KEYS, REDIS_TOPICS, TaskStatus, WorkerStatus
from mloc.common.schemas import (
    TaskInfo, WorkerInfo, UsageStats, QueryParameters
)
from mloc.common.utils import calculate_gpu_hours


logger = structlog.get_logger(__name__)


class StateMonitor:
    """Monitor for tracking system state and usage statistics"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.monitor_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self) -> None:
        """Start the monitor"""
        self.redis = aioredis.from_url(self.redis_url)
        self.running = True
        
        # Start monitoring tasks
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        # Start message subscriptions
        asyncio.create_task(self._subscribe_to_events())
        
        logger.info("State monitor started")
    
    async def stop(self) -> None:
        """Stop the monitor"""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.redis:
            await self.redis.close()
        
        logger.info("State monitor stopped")
    
    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task by ID"""
        task_data = await self.redis.hget(REDIS_KEYS["TASK_STATUS"], task_id)
        if not task_data:
            return None
        
        return TaskInfo.model_validate_json(task_data)
    
    async def get_tasks(
        self,
        page: int = 1,
        page_size: int = 10,
        status: Optional[str] = None,
        owner: Optional[str] = None
    ) -> Tuple[List[TaskInfo], int]:
        """Get tasks with pagination and filtering"""
        # Get all task IDs
        all_task_data = await self.redis.hgetall(REDIS_KEYS["TASK_STATUS"])
        
        tasks = []
        for task_id, task_data in all_task_data.items():
            try:
                task_info = TaskInfo.model_validate_json(task_data)
                
                # Apply filters
                if status and task_info.status.value != status:
                    continue
                
                if owner and task_info.config.metadata.owner != owner:
                    continue
                
                tasks.append(task_info)
            except Exception as e:
                logger.warning("Failed to parse task data", 
                             task_id=task_id.decode(), error=str(e))
        
        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        # Apply pagination
        total = len(tasks)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        return tasks[start_idx:end_idx], total
    
    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker by ID"""
        worker_data = await self.redis.hget(REDIS_KEYS["WORKER_REGISTRY"], worker_id)
        if not worker_data:
            return None
        
        return WorkerInfo.model_validate_json(worker_data)
    
    async def get_workers(self) -> List[WorkerInfo]:
        """Get all workers"""
        worker_data = await self.redis.hgetall(REDIS_KEYS["WORKER_REGISTRY"])
        
        workers = []
        for worker_id, data in worker_data.items():
            try:
                worker_info = WorkerInfo.model_validate_json(data)
                workers.append(worker_info)
            except Exception as e:
                logger.warning("Failed to parse worker data",
                             worker_id=worker_id.decode(), error=str(e))
        
        return workers
    
    async def get_usage_stats(self, query: QueryParameters) -> UsageStats:
        """Get usage statistics based on query parameters"""
        # Get all task data
        all_task_data = await self.redis.hgetall(REDIS_KEYS["TASK_STATUS"])
        
        stats = UsageStats()
        
        for task_data in all_task_data.values():
            try:
                task_info = TaskInfo.model_validate_json(task_data)
                
                # Skip non-completed tasks
                if task_info.status != TaskStatus.COMPLETED:
                    continue
                
                # Apply filters
                if query.user and task_info.config.metadata.owner != query.user:
                    continue
                
                if query.project and task_info.config.metadata.project != query.project:
                    continue
                
                # Apply date range filter
                if query.start_date or query.end_date:
                    if not task_info.completed_at:
                        continue
                    
                    if query.start_date:
                        start_date = datetime.fromisoformat(query.start_date)
                        if task_info.completed_at < start_date:
                            continue
                    
                    if query.end_date:
                        end_date = datetime.fromisoformat(query.end_date)
                        if task_info.completed_at > end_date:
                            continue
                
                # Calculate GPU hours
                if task_info.started_at and task_info.completed_at:
                    gpu_count = 0
                    gpu_type = "unknown"
                    
                    if task_info.config.spec.resources.gpu:
                        gpu_count = task_info.config.spec.resources.gpu.count
                        gpu_type = task_info.config.spec.resources.gpu.type.value
                    
                    if gpu_count > 0:
                        gpu_hours = calculate_gpu_hours(
                            task_info.started_at,
                            task_info.completed_at,
                            gpu_count
                        )
                        
                        # Update statistics
                        stats.total_gpu_hours += gpu_hours
                        stats.total_tasks_completed += 1
                        
                        # Breakdown by GPU type
                        if gpu_type not in stats.breakdown_by_gpu:
                            stats.breakdown_by_gpu[gpu_type] = 0.0
                        stats.breakdown_by_gpu[gpu_type] += gpu_hours
                        
                        # Breakdown by user
                        user = task_info.config.metadata.owner
                        if user not in stats.breakdown_by_user:
                            stats.breakdown_by_user[user] = 0.0
                        stats.breakdown_by_user[user] += gpu_hours
                        
                        # Breakdown by project
                        project = task_info.config.metadata.project
                        if project not in stats.breakdown_by_project:
                            stats.breakdown_by_project[project] = 0.0
                        stats.breakdown_by_project[project] += gpu_hours
                
            except Exception as e:
                logger.warning("Failed to process task for statistics", error=str(e))
        
        return stats
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        logger.info("Monitor loop started")
        
        while self.running:
            try:
                await self._check_worker_health()
                await self._cleanup_stale_tasks()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error("Error in monitor loop", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_worker_health(self) -> None:
        """Check worker health and mark offline workers"""
        current_time = datetime.utcnow()
        threshold = current_time - timedelta(minutes=2)  # 2 minutes timeout
        
        workers = await self.get_workers()
        
        for worker in workers:
            if worker.last_heartbeat < threshold:
                if worker.status != WorkerStatus.OFFLINE:
                    # Mark worker as offline
                    worker.status = WorkerStatus.OFFLINE
                    
                    await self.redis.hset(
                        REDIS_KEYS["WORKER_REGISTRY"],
                        worker.worker_id,
                        worker.model_dump_json()
                    )
                    
                    logger.warning("Worker marked as offline", worker_id=worker.worker_id)
                    
                    # If worker had a running task, mark it as failed
                    if worker.current_task_id:
                        await self._handle_worker_failure(worker.worker_id, worker.current_task_id)
    
    async def _handle_worker_failure(self, worker_id: str, task_id: str) -> None:
        """Handle worker failure by marking task as failed"""
        task_data = await self.redis.hget(REDIS_KEYS["TASK_STATUS"], task_id)
        if task_data:
            task_info = TaskInfo.model_validate_json(task_data)
            
            if task_info.status == TaskStatus.RUNNING:
                task_info.status = TaskStatus.FAILED
                task_info.error_message = f"Worker {worker_id} went offline"
                task_info.completed_at = datetime.utcnow()
                
                await self.redis.hset(
                    REDIS_KEYS["TASK_STATUS"],
                    task_id,
                    task_info.model_dump_json()
                )
                
                logger.error("Task marked as failed due to worker failure",
                           task_id=task_id, worker_id=worker_id)
    
    async def _cleanup_stale_tasks(self) -> None:
        """Clean up stale tasks that have been running too long"""
        current_time = datetime.utcnow()
        timeout_threshold = current_time - timedelta(hours=12)  # 12 hour timeout
        
        all_task_data = await self.redis.hgetall(REDIS_KEYS["TASK_STATUS"])
        
        for task_id, task_data in all_task_data.items():
            try:
                task_info = TaskInfo.model_validate_json(task_data)
                
                # Check for stale running tasks
                if (task_info.status == TaskStatus.RUNNING and 
                    task_info.started_at and 
                    task_info.started_at < timeout_threshold):
                    
                    task_info.status = TaskStatus.FAILED
                    task_info.error_message = "Task timed out"
                    task_info.completed_at = current_time
                    
                    await self.redis.hset(
                        REDIS_KEYS["TASK_STATUS"],
                        task_id.decode(),
                        task_info.model_dump_json()
                    )
                    
                    logger.warning("Task marked as failed due to timeout",
                                 task_id=task_id.decode())
                    
            except Exception as e:
                logger.warning("Failed to process task in cleanup",
                             task_id=task_id.decode(), error=str(e))
    
    async def _subscribe_to_events(self) -> None:
        """Subscribe to Redis events"""
        pubsub = self.redis.pubsub()
        
        await pubsub.subscribe(
            REDIS_TOPICS["WORKER_HEARTBEAT"],
            REDIS_TOPICS["TASK_PROGRESS"],
            REDIS_TOPICS["TASK_COMPLETION"]
        )
        
        logger.info("Subscribed to Redis events")
        
        try:
            while self.running:
                message = await pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    await self._handle_event(message)
        except Exception as e:
            logger.error("Error in event subscription", error=str(e))
        finally:
            await pubsub.unsubscribe()
    
    async def _handle_event(self, message: Dict) -> None:
        """Handle incoming events"""
        try:
            topic = message["channel"].decode()
            data = json.loads(message["data"])
            
            if topic == REDIS_TOPICS["WORKER_HEARTBEAT"]:
                await self._handle_worker_heartbeat(data)
            elif topic == REDIS_TOPICS["TASK_PROGRESS"]:
                await self._handle_task_progress(data)
            elif topic == REDIS_TOPICS["TASK_COMPLETION"]:
                await self._handle_task_completion(data)
                
        except Exception as e:
            logger.error("Failed to handle event", error=str(e))
    
    async def _handle_worker_heartbeat(self, data: Dict) -> None:
        """Handle worker heartbeat"""
        worker_id = data.get("worker_id")
        if not worker_id:
            return
        
        # Update worker's last heartbeat
        worker_data = await self.redis.hget(REDIS_KEYS["WORKER_REGISTRY"], worker_id)
        if worker_data:
            worker_info = WorkerInfo.model_validate_json(worker_data)
            worker_info.last_heartbeat = datetime.utcnow()
            
            # Update status if it was offline
            if worker_info.status == WorkerStatus.OFFLINE:
                worker_info.status = WorkerStatus.IDLE
            
            await self.redis.hset(
                REDIS_KEYS["WORKER_REGISTRY"],
                worker_id,
                worker_info.model_dump_json()
            )
    
    async def _handle_task_progress(self, data: Dict) -> None:
        """Handle task progress update"""
        task_id = data.get("task_id")
        progress = data.get("progress", 0.0)
        
        if not task_id:
            return
        
        # Update task progress
        task_data = await self.redis.hget(REDIS_KEYS["TASK_STATUS"], task_id)
        if task_data:
            task_info = TaskInfo.model_validate_json(task_data)
            task_info.progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
            
            # Set started_at if this is the first progress update
            if task_info.status == TaskStatus.SCHEDULED:
                task_info.status = TaskStatus.RUNNING
                task_info.started_at = datetime.utcnow()
            
            await self.redis.hset(
                REDIS_KEYS["TASK_STATUS"],
                task_id,
                task_info.model_dump_json()
            )
    
    async def _handle_task_completion(self, data: Dict) -> None:
        """Handle task completion"""
        task_id = data.get("task_id")
        worker_id = data.get("worker_id")
        success = data.get("success", False)
        error_message = data.get("error_message")
        
        if not task_id or not worker_id:
            return
        
        # Update task status
        task_data = await self.redis.hget(REDIS_KEYS["TASK_STATUS"], task_id)
        if task_data:
            task_info = TaskInfo.model_validate_json(task_data)
            task_info.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            task_info.completed_at = datetime.utcnow()
            task_info.progress = 1.0 if success else task_info.progress
            
            if error_message:
                task_info.error_message = error_message
            
            await self.redis.hset(
                REDIS_KEYS["TASK_STATUS"],
                task_id,
                task_info.model_dump_json()
            )
        
        # Update worker status back to idle
        worker_data = await self.redis.hget(REDIS_KEYS["WORKER_REGISTRY"], worker_id)
        if worker_data:
            worker_info = WorkerInfo.model_validate_json(worker_data)
            worker_info.status = WorkerStatus.IDLE
            worker_info.current_task_id = None
            
            await self.redis.hset(
                REDIS_KEYS["WORKER_REGISTRY"],
                worker_id,
                worker_info.model_dump_json()
            )
        
        logger.info("Task completed",
                   task_id=task_id,
                   worker_id=worker_id,
                   success=success)