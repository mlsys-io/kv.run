from __future__ import annotations

"""Pydantic 模型定义，用于规范化并校验任务 YAML 结构。"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


def _to_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


class MetadataModel(BaseModel):
    """任务元数据，允许额外字段（annotations、labels 等）。"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="任务名称，作为唯一标识前缀。")
    owner: Optional[str] = Field(default=None, description="任务拥有者，可选。")
    annotations: Dict[str, Any] = Field(default_factory=dict, description="任意键值注释。")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        value = _to_str(value)
        if not value:
            raise ValueError("metadata.name must not be empty")
        return value

    @field_validator("owner")
    @classmethod
    def _trim_owner(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        trimmed = _to_str(value)
        return trimmed or None


class GPUConfig(BaseModel):
    """GPU 资源配置。当定义该块时必须提供 count。"""

    model_config = ConfigDict(extra="allow")

    count: int = Field(..., ge=0, description="GPU 数量，必须为非负整数。")
    type: Optional[str] = Field(default=None, description="GPU 型号，可选。")

    @field_validator("type")
    @classmethod
    def _trim_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        trimmed = _to_str(value)
        return trimmed or None


class HardwareConfig(BaseModel):
    """硬件资源需求。"""

    model_config = ConfigDict(extra="allow")

    cpu: str = Field(..., description="CPU 配额，例如 '8' 或 '8c'。")
    memory: str = Field(..., description="内存配额，例如 '32Gi'。")
    gpu: Optional[GPUConfig] = Field(default=None, description="可选 GPU 配置。")

    @field_validator("cpu", "memory")
    @classmethod
    def _ensure_non_empty(cls, value: str, field) -> str:  # type: ignore[override]
        trimmed = _to_str(value)
        if not trimmed:
            raise ValueError(f"spec.resources.hardware.{field.field_name} must not be empty")
        return trimmed


class ResourceConfig(BaseModel):
    """任务资源配置。"""

    model_config = ConfigDict(extra="allow")

    replicas: int = Field(default=1, ge=1, description="调度副本数量，默认 1。")
    hardware: HardwareConfig = Field(..., description="硬件配置。")


class ParallelConfig(BaseModel):
    """并行策略配置。"""

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(default=False, description="是否启用数据并行。")
    max_shards: Optional[int] = Field(default=None, ge=1, description="分片数量上限。")
    strategy: Optional[str] = Field(default=None, description="自定义策略名称。")


class OutputDestination(BaseModel):
    """输出目标配置，支持 local 与 http。"""

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="local", description="输出类型：local 或 http。")
    path: Optional[str] = Field(default=None, description="local 模式下的目录或相对路径。")
    url: Optional[str] = Field(default=None, description="http 模式上传目标 URL。")
    method: Optional[str] = Field(default="POST", description="HTTP 方法，默认 POST。")
    headers: Dict[str, Any] = Field(default_factory=dict, description="HTTP 头部配置。")
    timeoutSec: Optional[float] = Field(default=None, ge=0, description="HTTP 请求超时秒数。")

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, value: str) -> str:
        value = _to_str(value)
        return value.lower() or "local"

    @field_validator("method")
    @classmethod
    def _normalize_method(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        trimmed = _to_str(value)
        return trimmed.upper() or None

    @model_validator(mode="after")
    def _http_constraints(self) -> "OutputDestination":
        if self.type == "http":
            if not self.url:
                raise ValueError("spec.output.destination.url is required when destination.type == 'http'")
        return self


class OutputConfig(BaseModel):
    """输出配置，含目标与 artifact 白名单。"""

    model_config = ConfigDict(extra="allow")

    destination: OutputDestination = Field(default_factory=OutputDestination)
    artifacts: List[str] = Field(default_factory=list, description="期望产出的工件名称列表。")

    @field_validator("artifacts", mode="before")
    @classmethod
    def _coerce_artifacts(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (tuple, set)):
            iterable = list(value)
        else:
            iterable = value
        if not isinstance(iterable, list):
            raise ValueError("spec.output.artifacts must be a list of strings")
        result: List[str] = []
        for item in iterable:
            item_str = _to_str(item)
            if item_str:
                result.append(item_str)
        return result


class TaskSpecModel(BaseModel):
    """核心任务规范。"""

    model_config = ConfigDict(extra="allow")

    taskType: str = Field(..., description="任务类型，例如 inference、rag、sft。")
    resources: ResourceConfig = Field(..., description="资源与硬件配置。")
    output: OutputConfig = Field(default_factory=OutputConfig, description="输出配置。")
    parallel: ParallelConfig = Field(default_factory=ParallelConfig, description="并行配置。")
    dependsOn: List[str] = Field(default_factory=list, description="上游任务 ID 列表。")
    sloSeconds: Optional[float] = Field(default=None, gt=0, description="目标时延（秒），可选。")
    stages: Optional[List[Dict[str, Any]]] = Field(default=None, description="线性阶段定义。")
    graph: Optional[Dict[str, Any]] = Field(default=None, description="DAG 图定义。")

    @field_validator("taskType")
    @classmethod
    def _normalize_task_type(cls, value: str) -> str:
        value = _to_str(value)
        if not value:
            raise ValueError("spec.taskType must be a non-empty string")
        return value

    @field_validator("dependsOn", mode="before")
    @classmethod
    def _coerce_depends(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (tuple, set)):
            value = list(value)
        if not isinstance(value, list):
            raise ValueError("spec.dependsOn must be a list")
        result: List[str] = []
        for item in value:
            candidate = _to_str(item)
            if candidate:
                result.append(candidate)
        return result


class TaskDocumentModel(BaseModel):
    """完整的任务 YAML 文档模型。"""

    model_config = ConfigDict(extra="allow")

    apiVersion: str = Field(..., description="API 版本，例 mloc/v1。")
    kind: str = Field(..., description="任务种类。")
    metadata: MetadataModel = Field(..., description="任务元数据。")
    spec: TaskSpecModel = Field(..., description="任务规范。")

    @field_validator("apiVersion", "kind")
    @classmethod
    def _trim_non_empty(cls, value: str, field) -> str:  # type: ignore[override]
        trimmed = _to_str(value)
        if not trimmed:
            raise ValueError(f"{field.field_name} must not be empty")
        return trimmed


def format_validation_error(exc: ValidationError) -> str:
    """将 Pydantic ValidationError 转换为可读的错误消息。"""

    def _format_loc(parts: List[Any]) -> str:
        formatted: List[str] = []
        for part in parts:
            if isinstance(part, int):
                formatted.append(f"[{part}]")
            else:
                formatted.append(str(part))
        loc = ".".join(formatted)
        return loc.replace(".[", "[")

    messages: List[str] = []
    for err in exc.errors():
        loc = _format_loc(list(err.get("loc", ())))
        msg = err.get("msg", "invalid value")
        if loc:
            messages.append(f"{loc}: {msg}")
        else:
            messages.append(msg)
    return "; ".join(messages)

