"""
Worker Task Listener

This module implements the worker's task listening and lifecycle management.
It handles worker registration, task assignment, and coordination with the executor.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Optional

import aioredis
import structlog

from mloc.common.constants import REDIS_KEYS, REDIS_TOPICS, WorkerStatus
from mloc.common.schemas import WorkerInfo, WorkerRegistration, TaskConfig
from mloc.common.utils import generate_worker_id, get_hardware_info
from .executor import TaskExecutor


logger = structlog.get_logger(__name__)


class WorkerListener:
    """Worker listener for handling task assignments"""
    
    def __init__(
        self,
        worker_id: Optional[str] = None,
        redis_url: str = "redis://localhost:6379",
        log_level: str = "INFO"
    ):
        self.worker_id = worker_id or generate_worker_id()
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.executor = TaskExecutor(self.worker_id)
        self.running = False
        
        # Task management
        self.current_task_id: Optional[str] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.listener_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the worker listener"""
        logger.info("Starting worker listener", worker_id=self.worker_id)
        
        # Connect to Redis
        self.redis = aioredis.from_url(self.redis_url)
        
        # Register worker
        await self._register_worker()
        
        # Start background tasks
        self.running = True
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.listener_task = asyncio.create_task(self._listen_for_tasks())
        
        try:
            # Wait for tasks to complete
            await asyncio.gather(
                self.heartbeat_task,
                self.listener_task,
                return_exceptions=True
            )
        except KeyboardInterrupt:
            logger.info("Worker interrupted, shutting down...")
        finally:
            await self._shutdown()
    
    async def _register_worker(self) -> None:
        """Register worker with orchestrator"""
        # Get hardware information
        hardware_info = get_hardware_info()
        
        # Create worker registration
        registration = WorkerRegistration(
            worker_id=self.worker_id,
            hardware=hardware_info
        )
        
        # Create worker info
        worker_info = WorkerInfo(
            worker_id=self.worker_id,
            status=WorkerStatus.IDLE,
            hardware=hardware_info,
            registered_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow()
        )
        
        # Register in Redis
        await self.redis.hset(
            REDIS_KEYS["WORKER_REGISTRY"],
            self.worker_id,
            worker_info.model_dump_json()
        )
        
        logger.info(
            "Worker registered successfully",
            worker_id=self.worker_id,
            cpu_count=hardware_info["cpu_count"],
            memory_gb=hardware_info["memory_gb"],
            gpu_count=hardware_info["gpu_count"],
            available_gpu_types=hardware_info["available_gpu_types"]
        )
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to orchestrator"""
        while self.running:
            try:
                # Send heartbeat
                heartbeat_data = {
                    "worker_id": self.worker_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "current_task_id": self.current_task_id,
                    "status": WorkerStatus.RUNNING.value if self.current_task_id else WorkerStatus.IDLE.value
                }
                
                await self.redis.publish(
                    REDIS_TOPICS["WORKER_HEARTBEAT"],
                    json.dumps(heartbeat_data)
                )
                
                logger.debug("Heartbeat sent", worker_id=self.worker_id)
                
                # Wait for next heartbeat
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
            except Exception as e:
                logger.error("Failed to send heartbeat", error=str(e))
                await asyncio.sleep(10)  # Retry sooner on error
    
    async def _listen_for_tasks(self) -> None:
        """Listen for task assignments"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(REDIS_TOPICS["TASK_ASSIGNMENT"])
        
        logger.info("Listening for task assignments", worker_id=self.worker_id)
        
        try:
            while self.running:
                message = await pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    await self._handle_task_assignment(message)
        except Exception as e:
            logger.error("Error in task listener", error=str(e))
        finally:
            await pubsub.unsubscribe()
    
    async def _handle_task_assignment(self, message) -> None:
        """Handle task assignment message"""
        try:
            data = json.loads(message["data"])
            
            # Check if this assignment is for us
            if data.get("worker_id") != self.worker_id:
                return
            
            task_id = data.get("task_id")
            config_data = data.get("config")
            
            if not task_id or not config_data:
                logger.warning("Invalid task assignment message", data=data)
                return
            
            # Check if we're already busy
            if self.current_task_id:
                logger.warning(
                    "Received task assignment while busy",
                    current_task=self.current_task_id,
                    new_task=task_id
                )
                return
            
            # Parse task configuration
            task_config = TaskConfig.model_validate(config_data)
            
            logger.info(
                "Received task assignment",
                task_id=task_id,
                task_type=task_config.spec.task_type
            )
            
            # Start task execution
            self.current_task_id = task_id
            await self._execute_task(task_id, task_config)
            
        except Exception as e:
            logger.error("Failed to handle task assignment", error=str(e))
    
    async def _execute_task(self, task_id: str, config: TaskConfig) -> None:
        """Execute the assigned task"""
        success = False
        error_message = None
        
        try:
            logger.info("Starting task execution", task_id=task_id)
            
            # Execute task using the executor
            await self.executor.execute_task(
                task_id=task_id,
                config=config,
                progress_callback=self._send_progress_update
            )
            
            success = True
            logger.info("Task completed successfully", task_id=task_id)
            
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error("Task execution failed", task_id=task_id, error=error_message)
            
        finally:
            # Send completion notification
            await self._send_completion_notification(task_id, success, error_message)
            
            # Reset current task
            self.current_task_id = None
    
    async def _send_progress_update(self, task_id: str, progress: float) -> None:
        """Send task progress update"""
        try:
            progress_data = {
                "task_id": task_id,
                "worker_id": self.worker_id,
                "progress": progress,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis.publish(
                REDIS_TOPICS["TASK_PROGRESS"],
                json.dumps(progress_data)
            )
            
            logger.debug("Progress update sent", task_id=task_id, progress=progress)
            
        except Exception as e:
            logger.error("Failed to send progress update", error=str(e))
    
    async def _send_completion_notification(
        self,
        task_id: str,
        success: bool,
        error_message: Optional[str] = None
    ) -> None:
        """Send task completion notification"""
        try:
            completion_data = {
                "task_id": task_id,
                "worker_id": self.worker_id,
                "success": success,
                "error_message": error_message,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            await self.redis.publish(
                REDIS_TOPICS["TASK_COMPLETION"],
                json.dumps(completion_data)
            )
            
            logger.info(
                "Completion notification sent",
                task_id=task_id,
                success=success
            )
            
        except Exception as e:
            logger.error("Failed to send completion notification", error=str(e))
    
    async def _shutdown(self) -> None:
        """Shutdown worker gracefully"""
        logger.info("Shutting down worker", worker_id=self.worker_id)
        
        self.running = False
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            
        if self.listener_task:
            self.listener_task.cancel()
        
        # Unregister worker
        try:
            await self.redis.hdel(REDIS_KEYS["WORKER_REGISTRY"], self.worker_id)
            logger.info("Worker unregistered", worker_id=self.worker_id)
        except Exception as e:
            logger.error("Failed to unregister worker", error=str(e))
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        logger.info("Worker shutdown complete", worker_id=self.worker_id)