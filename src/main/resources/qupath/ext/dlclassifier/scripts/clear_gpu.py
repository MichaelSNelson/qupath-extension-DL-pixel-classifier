"""
Force-clear GPU memory.

Outputs:
    success: bool
    message: str
"""
import logging

logger = logging.getLogger("dlclassifier.appose.clear_gpu")

try:
    if inference_service is not None:
        inference_service.clear_model_cache()

    if gpu_manager is not None:
        before = gpu_manager.get_memory_info()
        gpu_manager.clear_cache()
        after = gpu_manager.get_memory_info()

        freed_mb = 0.0
        if before and after:
            freed_mb = before.get("allocated_mb", 0) - after.get("allocated_mb", 0)

        task.outputs["success"] = True
        task.outputs["message"] = "GPU memory cleared. Freed %.1f MB." % freed_mb
    else:
        task.outputs["success"] = True
        task.outputs["message"] = "No GPU manager available (CPU mode)."

except Exception as e:
    logger.error("Failed to clear GPU memory: %s", e)
    task.outputs["success"] = False
    task.outputs["message"] = "Error: %s" % str(e)
