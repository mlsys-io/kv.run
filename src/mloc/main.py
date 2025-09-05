"""
MLOC Main Entry Point

This module serves as the main entry point for the MLOC container.
It can start as either an Orchestrator or Worker based on configuration.
"""

import asyncio
import os
import sys
from typing import Optional

import typer
import uvicorn
from pydantic_settings import BaseSettings

from mloc.common.constants import NodeType
from mloc.common.utils import setup_logging
from mloc.orchestrator.api import create_app as create_orchestrator_app
from mloc.worker.listener import WorkerListener


class Settings(BaseSettings):
    """Application settings"""
    node_type: NodeType = NodeType.WORKER
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    
    # Worker specific
    worker_id: Optional[str] = None
    
    class Config:
        env_prefix = "MLOC_"


app = typer.Typer(help="MLOC - Modular LLM Operations Container")
settings = Settings()


@app.command()
def start(
    node_type: Optional[str] = typer.Option(
        None,
        "--node-type",
        "-t",
        help="Node type: orchestrator or worker"
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p", 
        help="Port to run on"
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Log level: DEBUG, INFO, WARNING, ERROR"
    ),
) -> None:
    """Start MLOC node as orchestrator or worker"""
    
    # Override settings with CLI args
    if node_type:
        try:
            settings.node_type = NodeType(node_type.upper())
        except ValueError:
            typer.echo(f"Invalid node type: {node_type}")
            raise typer.Exit(1)
    
    if port:
        settings.port = port
        
    if log_level:
        settings.log_level = log_level.upper()
    
    setup_logging(settings.log_level)
    
    if settings.node_type == NodeType.ORCHESTRATOR:
        start_orchestrator()
    elif settings.node_type == NodeType.WORKER:
        start_worker()
    else:
        typer.echo(f"Unknown node type: {settings.node_type}")
        raise typer.Exit(1)


def start_orchestrator() -> None:
    """Start the orchestrator node"""
    typer.echo(f"🚀 Starting MLOC Orchestrator on {settings.host}:{settings.port}")
    
    orchestrator_app = create_orchestrator_app()
    
    uvicorn.run(
        orchestrator_app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


def start_worker() -> None:
    """Start the worker node"""
    worker_id = settings.worker_id or f"worker-{os.getpid()}"
    
    typer.echo(f"🔨 Starting MLOC Worker: {worker_id}")
    
    worker = WorkerListener(
        worker_id=worker_id,
        redis_url=settings.redis_url,
        log_level=settings.log_level
    )
    
    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        typer.echo("👋 Worker shutting down...")
    except Exception as e:
        typer.echo(f"❌ Worker failed: {e}")
        sys.exit(1)


@app.command()
def version() -> None:
    """Show MLOC version"""
    from mloc import __version__
    typer.echo(f"MLOC v{__version__}")


if __name__ == "__main__":
    app()