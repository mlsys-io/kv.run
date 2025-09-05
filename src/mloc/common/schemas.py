"""
Pydantic schemas for MLOC system.

These schemas define the data models used for configuration, API requests/responses,
and internal communication between components.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

from .constants import (
    TaskType, TaskStatus, WorkerStatus, ResourceType, 
    StorageType, GPUType, AdapterType
)


class BaseSchema(BaseModel):
    """Base schema with common configuration"""
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid"
    )


# Resource schemas
class GPUResource(BaseSchema):
    """GPU resource specification"""
    type: GPUType = GPUType.ANY
    count: int = Field(default=1, ge=1, le=8)


class ResourceSpec(BaseSchema):
    """Resource specification"""
    cpu: str = "2"
    memory: str = "8Gi" 
    gpu: Optional[GPUResource] = None


# Storage schemas
class StorageSource(BaseSchema):
    """Storage source specification"""
    type: StorageType
    identifier: str  # repo_id, s3://bucket/path, /local/path
    revision: Optional[str] = "main"
    access_token: Optional[str] = None


# Model and adapter schemas
class AdapterConfig(BaseSchema):
    """Adapter configuration for fine-tuning"""
    type: AdapterType = AdapterType.LORA
    r: int = Field(default=16, ge=1, le=512)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class ModelConfig(BaseSchema):
    """Model configuration"""
    source: StorageSource
    adapter: Optional[AdapterConfig] = None


# Dataset schemas  
class DataPreprocessing(BaseSchema):
    """Data preprocessing configuration"""
    prompt_template: Optional[str] = None
    max_seq_length: int = Field(default=2048, ge=1, le=32768)
    padding: str = "longest"
    truncation: bool = True


class DatasetConfig(BaseSchema):
    """Dataset configuration"""
    source: StorageSource
    split: str = "train"
    preprocessing: Optional[DataPreprocessing] = None


# Training schemas
class TrainingHyperparameters(BaseSchema):
    """Training hyperparameters"""
    output_dir: str = "/artifacts"
    num_train_epochs: int = Field(default=3, ge=1)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_steps: int = Field(default=100, ge=0)
    logging_steps: int = Field(default=10, ge=1)
    eval_steps: Optional[int] = None
    save_steps: int = Field(default=500, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False


# Output schemas
class OutputDestination(BaseSchema):
    """Output destination specification"""
    type: StorageType
    bucket: Optional[str] = None  # For S3
    path: str = "/"
    access_token: Optional[str] = None


class OutputConfig(BaseSchema):
    """Output configuration"""
    destination: OutputDestination
    artifacts: List[str] = Field(default_factory=lambda: ["adapter_weights", "training_logs"])


# Task metadata
class TaskMetadata(BaseSchema):
    """Task metadata"""
    name: str
    owner: str = "unknown"
    project: str = "default"
    description: Optional[str] = None
    annotations: Dict[str, str] = Field(default_factory=dict)


# Complete task specification
class TaskSpec(BaseSchema):
    """Complete task specification"""
    task_type: TaskType
    resources: ResourceSpec
    model: ModelConfig
    dataset: Optional[DatasetConfig] = None
    hyperparameters: Optional[TrainingHyperparameters] = None
    output: OutputConfig


class TaskConfig(BaseSchema):
    """Complete task configuration (matches YAML structure)"""
    api_version: str = Field(alias="apiVersion", default="mloc/v1")
    kind: str = "TrainingTask"
    metadata: TaskMetadata
    spec: TaskSpec


# Task execution schemas
class TaskRequest(BaseSchema):
    """Task creation request"""
    config: TaskConfig


class TaskInfo(BaseSchema):
    """Task information"""
    task_id: str
    config: TaskConfig
    status: TaskStatus
    worker_id: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)


class TaskResponse(BaseSchema):
    """Task response"""
    task_id: str
    status: TaskStatus
    message: str


# Worker schemas
class HardwareInfo(BaseSchema):
    """Hardware information"""
    cpu_count: int
    memory_gb: float
    gpu_info: List[Dict[str, Any]] = Field(default_factory=list)
    gpu_count: int = 0
    available_gpu_types: List[GPUType] = Field(default_factory=list)


class WorkerInfo(BaseSchema):
    """Worker information"""
    worker_id: str
    status: WorkerStatus
    hardware: HardwareInfo
    current_task_id: Optional[str] = None
    registered_at: datetime
    last_heartbeat: datetime


class WorkerRegistration(BaseSchema):
    """Worker registration request"""
    worker_id: str
    hardware: HardwareInfo


# API response schemas
class TaskListResponse(BaseSchema):
    """Task list response"""
    tasks: List[TaskInfo]
    total: int
    page: int = 1
    page_size: int = 10


class WorkerListResponse(BaseSchema):
    """Worker list response"""
    workers: List[WorkerInfo]
    total: int


# Statistics schemas
class UsageStats(BaseSchema):
    """Usage statistics"""
    total_gpu_hours: float = 0.0
    total_tasks_completed: int = 0
    breakdown_by_gpu: Dict[str, float] = Field(default_factory=dict)
    breakdown_by_user: Dict[str, float] = Field(default_factory=dict)
    breakdown_by_project: Dict[str, float] = Field(default_factory=dict)


class QueryParameters(BaseSchema):
    """Query parameters for statistics"""
    user: Optional[str] = None
    project: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class StatsResponse(BaseSchema):
    """Statistics response"""
    query_parameters: QueryParameters
    usage_stats: UsageStats


# Health check schemas
class HealthResponse(BaseSchema):
    """Health check response"""
    status: str = "healthy"
    timestamp: datetime
    version: str
    node_type: str