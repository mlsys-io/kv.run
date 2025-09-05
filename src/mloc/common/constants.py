"""
Constants used throughout the MLOC system.
"""

from enum import Enum
from typing import Dict, Any


class NodeType(str, Enum):
    """Node type enumeration"""
    ORCHESTRATOR = "ORCHESTRATOR"
    WORKER = "WORKER"


class TaskType(str, Enum):
    """Task type enumeration"""
    SFT = "sft"                    # Supervised Fine-Tuning
    REWARD_MODEL = "reward_model"  # Reward Model Training  
    PPO = "ppo"                   # PPO Training
    RAG_INFERENCE = "rag_inference"    # RAG Inference Service
    RAG_INDEXING = "rag_indexing"      # RAG Index Building
    AGENT_INFERENCE = "agent_inference" # Agent Inference Service


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    SCHEDULED = "scheduled" 
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerStatus(str, Enum):
    """Worker status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
    OFFLINE = "offline"


class ResourceType(str, Enum):
    """Resource type enumeration"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"


class StorageType(str, Enum):
    """Storage type enumeration"""
    HUGGINGFACE = "huggingface"
    S3 = "s3"
    LOCAL = "local"
    GIT = "git"


class GPUType(str, Enum):
    """GPU type enumeration"""
    NVIDIA_A100_80GB = "nvidia-a100-80gb"
    NVIDIA_A100_40GB = "nvidia-a100-40gb"
    NVIDIA_H100 = "nvidia-h100"
    NVIDIA_V100 = "nvidia-v100"
    NVIDIA_RTX_4090 = "nvidia-rtx-4090"
    NVIDIA_RTX_3090 = "nvidia-rtx-3090"
    NVIDIA_L40 = "nvidia-l40"
    ANY = "any"


class AdapterType(str, Enum):
    """Adapter type enumeration for fine-tuning"""
    LORA = "lora"
    QLORA = "qlora"
    ADALORA = "adalora"
    FULL = "full"


# Redis keys and topics
REDIS_KEYS = {
    "TASK_QUEUE": "mloc:tasks:queue",
    "WORKER_REGISTRY": "mloc:workers:registry",
    "TASK_STATUS": "mloc:tasks:status",
    "WORKER_STATUS": "mloc:workers:status",
    "USAGE_STATS": "mloc:stats:usage",
}

REDIS_TOPICS = {
    "TASK_ASSIGNMENT": "mloc:tasks:assignment",
    "WORKER_HEARTBEAT": "mloc:workers:heartbeat",
    "TASK_PROGRESS": "mloc:tasks:progress",
    "TASK_COMPLETION": "mloc:tasks:completion",
}

# API endpoints
API_ROUTES = {
    "TASKS": "/api/v1/tasks",
    "STATS": "/api/v1/stats", 
    "WORKERS": "/api/v1/workers",
    "HEALTH": "/health",
}

# Default timeouts and intervals (in seconds)
TIMEOUTS = {
    "TASK_EXECUTION": 3600 * 8,  # 8 hours
    "WORKER_HEARTBEAT": 30,      # 30 seconds
    "RESOURCE_DOWNLOAD": 1800,   # 30 minutes
    "MODEL_LOADING": 600,        # 10 minutes
}

# Default resource requirements
DEFAULT_RESOURCES: Dict[str, Any] = {
    "cpu": "2",
    "memory": "8Gi", 
    "gpu": {
        "type": GPUType.ANY,
        "count": 1
    }
}

# Supported model sources
SUPPORTED_MODEL_SOURCES = [
    StorageType.HUGGINGFACE,
    StorageType.S3,
    StorageType.LOCAL
]

# Supported dataset sources  
SUPPORTED_DATASET_SOURCES = [
    StorageType.HUGGINGFACE,
    StorageType.S3,
    StorageType.LOCAL
]