"""
Orchestrator API Server

This module implements the FastAPI-based REST API server for the orchestrator.
It handles task submissions, status queries, and usage statistics.
"""

from datetime import datetime
from typing import List, Optional

import structlog
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware

from mloc import __version__
from mloc.common.constants import API_ROUTES, NodeType
from mloc.common.schemas import (
    TaskRequest, TaskResponse, TaskInfo, TaskListResponse,
    WorkerListResponse, StatsResponse, HealthResponse, QueryParameters
)
from .scheduler import TaskScheduler
from .monitor import StateMonitor


logger = structlog.get_logger(__name__)


class OrchestratorAPI:
    """Orchestrator API server"""
    
    def __init__(self):
        self.scheduler = TaskScheduler()
        self.monitor = StateMonitor()
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="MLOC Orchestrator",
            description="Modular LLM Operations Container - Orchestrator API",
            version=__version__,
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)
        
        # Add startup/shutdown events
        app.add_event_handler("startup", self._startup)
        app.add_event_handler("shutdown", self._shutdown)
        
        return app
    
    def _add_routes(self, app: FastAPI) -> None:
        """Add API routes"""
        
        @app.get(API_ROUTES["HEALTH"], response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                timestamp=datetime.utcnow(),
                version=__version__,
                node_type=NodeType.ORCHESTRATOR.value
            )
        
        @app.post(API_ROUTES["TASKS"], response_model=TaskResponse)
        async def create_task(request: TaskRequest):
            """Create a new task"""
            try:
                task_id = await self.scheduler.submit_task(request.config)
                logger.info("Task submitted", task_id=task_id, 
                          task_type=request.config.spec.task_type)
                
                return TaskResponse(
                    task_id=task_id,
                    status="pending",
                    message=f"Task {task_id} submitted successfully"
                )
            except Exception as e:
                logger.error("Failed to submit task", error=str(e))
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.get(API_ROUTES["TASKS"], response_model=TaskListResponse)
        async def list_tasks(
            page: int = Query(1, ge=1),
            page_size: int = Query(10, ge=1, le=100),
            status: Optional[str] = Query(None),
            owner: Optional[str] = Query(None)
        ):
            """List tasks with pagination and filtering"""
            try:
                tasks, total = await self.monitor.get_tasks(
                    page=page,
                    page_size=page_size,
                    status=status,
                    owner=owner
                )
                
                return TaskListResponse(
                    tasks=tasks,
                    total=total,
                    page=page,
                    page_size=page_size
                )
            except Exception as e:
                logger.error("Failed to list tasks", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get(f"{API_ROUTES['TASKS']}/{{task_id}}", response_model=TaskInfo)
        async def get_task(task_id: str):
            """Get task by ID"""
            try:
                task = await self.monitor.get_task(task_id)
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")
                return task
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to get task", task_id=task_id, error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.delete(f"{API_ROUTES['TASKS']}/{{task_id}}", response_model=TaskResponse)
        async def cancel_task(task_id: str):
            """Cancel a task"""
            try:
                success = await self.scheduler.cancel_task(task_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
                
                return TaskResponse(
                    task_id=task_id,
                    status="cancelled",
                    message=f"Task {task_id} cancelled successfully"
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to cancel task", task_id=task_id, error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get(API_ROUTES["WORKERS"], response_model=WorkerListResponse)
        async def list_workers():
            """List all registered workers"""
            try:
                workers = await self.monitor.get_workers()
                return WorkerListResponse(
                    workers=workers,
                    total=len(workers)
                )
            except Exception as e:
                logger.error("Failed to list workers", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get(f"{API_ROUTES['WORKERS']}/{{worker_id}}")
        async def get_worker(worker_id: str):
            """Get worker by ID"""
            try:
                worker = await self.monitor.get_worker(worker_id)
                if not worker:
                    raise HTTPException(status_code=404, detail="Worker not found")
                return worker
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Failed to get worker", worker_id=worker_id, error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get(API_ROUTES["STATS"], response_model=StatsResponse)
        async def get_usage_stats(
            user: Optional[str] = Query(None),
            project: Optional[str] = Query(None),
            start_date: Optional[str] = Query(None),
            end_date: Optional[str] = Query(None)
        ):
            """Get usage statistics"""
            try:
                query_params = QueryParameters(
                    user=user,
                    project=project,
                    start_date=start_date,
                    end_date=end_date
                )
                
                usage_stats = await self.monitor.get_usage_stats(query_params)
                
                return StatsResponse(
                    query_parameters=query_params,
                    usage_stats=usage_stats
                )
            except Exception as e:
                logger.error("Failed to get usage stats", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _startup(self) -> None:
        """Startup event handler"""
        logger.info("Starting MLOC Orchestrator API")
        await self.scheduler.start()
        await self.monitor.start()
    
    async def _shutdown(self) -> None:
        """Shutdown event handler"""
        logger.info("Shutting down MLOC Orchestrator API")
        await self.scheduler.stop()
        await self.monitor.stop()


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    orchestrator = OrchestratorAPI()
    return orchestrator.app