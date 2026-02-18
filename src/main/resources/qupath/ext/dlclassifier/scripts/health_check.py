"""
Health check and GPU info query.

Outputs:
    healthy: bool
    gpu_available: bool
    gpu_name: str
    gpu_memory_mb: int
    device: str
"""
import logging

logger = logging.getLogger("dlclassifier.appose.health")

task.outputs["healthy"] = inference_service is not None

if gpu_manager is not None:
    info = gpu_manager.get_info()
    task.outputs["gpu_available"] = info.get("available", False)
    task.outputs["gpu_name"] = info.get("name", "")
    task.outputs["gpu_memory_mb"] = info.get("total_memory_mb", 0)
    task.outputs["device"] = info.get("device_string", "unknown")
else:
    task.outputs["gpu_available"] = False
    task.outputs["gpu_name"] = ""
    task.outputs["gpu_memory_mb"] = 0
    task.outputs["device"] = "unknown"
