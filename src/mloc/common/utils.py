"""
Utility functions used throughout the MLOC system.
"""

import json
import logging
import os
import subprocess
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
import yaml


def setup_logging(level: str = "INFO") -> None:
    """Setup structured logging for the application"""
    
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def generate_task_id() -> str:
    """Generate a unique task ID"""
    return f"task-{uuid.uuid4().hex[:8]}"


def generate_worker_id() -> str:
    """Generate a unique worker ID"""
    hostname = os.getenv("HOSTNAME", "unknown")
    return f"worker-{hostname}-{uuid.uuid4().hex[:8]}"


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {e}")


def save_yaml_config(config: Dict[str, Any], file_path: str) -> None:
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_json_config(file_path: str) -> Dict[str, Any]:
    """Load JSON configuration file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON configuration: {e}")


def save_json_config(config: Dict[str, Any], file_path: str) -> None:
    """Save configuration to JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, default=str)


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get GPU information using nvidia-smi"""
    try:
        result = subprocess.run([
            "nvidia-smi", 
            "--query-gpu=name,memory.total,memory.free,utilization.gpu", 
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, check=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [part.strip() for part in line.split(',')]
                if len(parts) == 4:
                    gpu_info.append({
                        "name": parts[0],
                        "memory_total_mb": int(parts[1]),
                        "memory_free_mb": int(parts[2]), 
                        "utilization_percent": int(parts[3])
                    })
        
        return gpu_info
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # nvidia-smi not available or failed
        return []


def get_hardware_info() -> Dict[str, Any]:
    """Get comprehensive hardware information"""
    import psutil
    
    # CPU information
    cpu_count = psutil.cpu_count(logical=True)
    
    # Memory information  
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024 ** 3)
    
    # GPU information
    gpu_info = get_gpu_info()
    gpu_count = len(gpu_info)
    
    # Determine available GPU types
    available_gpu_types = []
    for gpu in gpu_info:
        gpu_name = gpu["name"].lower()
        if "a100" in gpu_name:
            if "80gb" in gpu_name or gpu["memory_total_mb"] > 70000:
                available_gpu_types.append("nvidia-a100-80gb")
            else:
                available_gpu_types.append("nvidia-a100-40gb")
        elif "h100" in gpu_name:
            available_gpu_types.append("nvidia-h100")
        elif "v100" in gpu_name:
            available_gpu_types.append("nvidia-v100")
        elif "rtx 4090" in gpu_name:
            available_gpu_types.append("nvidia-rtx-4090")
        elif "rtx 3090" in gpu_name:
            available_gpu_types.append("nvidia-rtx-3090")
        elif "l40" in gpu_name:
            available_gpu_types.append("nvidia-l40")
    
    return {
        "cpu_count": cpu_count,
        "memory_gb": memory_gb,
        "gpu_info": gpu_info,
        "gpu_count": gpu_count,
        "available_gpu_types": list(set(available_gpu_types))  # Remove duplicates
    }


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    else:
        return f"{seconds / 3600:.1f} hours"


def calculate_gpu_hours(start_time: datetime, end_time: datetime, gpu_count: int) -> float:
    """Calculate GPU hours used"""
    duration_hours = (end_time - start_time).total_seconds() / 3600
    return duration_hours * gpu_count


def validate_resource_requirements(resources: Dict[str, Any]) -> bool:
    """Validate resource requirements"""
    # Basic validation - can be extended
    if "hardware" not in resources:
        return False
    
    hardware = resources["hardware"]
    
    # Check GPU requirements
    if "gpu" in hardware:
        gpu = hardware["gpu"]
        if "count" in gpu and gpu["count"] > 8:
            return False  # Maximum 8 GPUs per task
    
    return True


def sanitize_string(value: str) -> str:
    """Sanitize string for safe usage in file paths and identifiers"""
    import re
    
    # Replace spaces and special characters with hyphens
    sanitized = re.sub(r'[^\w\-_.]', '-', value)
    
    # Remove consecutive hyphens
    sanitized = re.sub(r'-+', '-', sanitized)
    
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip('-')
    
    return sanitized.lower()


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or "Operation"
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.utcnow()
        duration = self.elapsed_seconds
        print(f"{self.name} took {format_duration(duration)}")
    
    @property
    def elapsed_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0