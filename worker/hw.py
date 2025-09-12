# worker/hw.py
"""Hardware introspection helpers.

Collects lightweight CPU/memory/GPU/network information for registration.
"""
from __future__ import annotations

import os
import re
import platform
import socket
import subprocess
import sys
from typing import Any, Dict, List


def _run(cmd: List[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def collect_hw() -> Dict[str, Any]:
    # CPU
    cpu = {"logical_cores": os.cpu_count() or 0, "model": platform.processor() or platform.machine()}
    # Memory
    mem = {"total_bytes": None}
    if sys.platform.startswith("linux") and os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem["total_bytes"] = int(line.split()[1]) * 1024
                    break
    # GPU (NVIDIA)
    full = _run(["nvidia-smi"])  # presence indicates NVIDIA stack
    cuda = None
    drv = None
    m = re.search(r"CUDA Version:\s*([\w\.\-]+)", full)
    cuda = m.group(1) if m else None
    m = re.search(r"Driver Version:\s*([\w\.\-]+)", full)
    drv = m.group(1) if m else None
    lst = _run(["nvidia-smi", "-L"]) if full else ""
    gpus: List[Dict[str, Any]] = []
    for line in lst.splitlines():
        m = re.match(r"GPU\s+(\d+):\s+(.+?)\s+\(UUID:\s*([^\)]+)\)", line.strip())
        if m:
            gpus.append({"index": int(m.group(1)), "name": m.group(2), "uuid": m.group(3)})
    gpu = {"driver_version": drv, "cuda_version": cuda, "gpus": gpus}

    # Network
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = None

    return {"cpu": cpu, "memory": mem, "gpu": gpu, "network": {"ip": ip}}