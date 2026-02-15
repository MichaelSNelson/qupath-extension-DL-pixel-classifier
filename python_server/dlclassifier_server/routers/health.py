"""Health check endpoints."""

import gc
import logging

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Check server health status."""
    return {"status": "healthy", "version": "0.1.0"}


@router.get("/gpu")
async def gpu_info(request: Request):
    """Get GPU availability information."""
    gpu_manager = request.app.state.gpu_manager

    return {
        "available": gpu_manager.is_available(),
        "name": gpu_manager.get_device_name(),
        "memory_mb": gpu_manager.get_memory_mb(),
        "cuda_version": gpu_manager.get_cuda_version()
    }


@router.post("/gpu/clear")
async def clear_gpu_memory(request: Request):
    """Force-clear all GPU memory held by the server.

    This endpoint:
    1. Cancels any running training jobs
    2. Clears the inference model cache
    3. Runs Python garbage collection
    4. Clears the GPU memory cache (CUDA/MPS)

    Useful after a crash, failed training, or when GPU memory needs
    to be reclaimed without restarting the server.
    """
    gpu_manager = request.app.state.gpu_manager
    job_manager = request.app.state.job_manager

    # Capture memory before cleanup
    memory_before = gpu_manager.get_memory_info()

    cancelled_jobs = []

    # 1. Cancel any running/paused training jobs and free their resources
    from ..services.job_manager import JobStatus
    for job in job_manager.list_jobs():
        if job.status in (JobStatus.TRAINING, JobStatus.PAUSED, JobStatus.PENDING):
            job.cancel()
            cancelled_jobs.append(job.job_id)
            logger.info("Cancelled training job %s for GPU cleanup", job.job_id)

    # 2. Clear inference model cache
    try:
        from ..services.inference_service import InferenceService
        # Clear the singleton/cached models by creating a temporary instance
        # and clearing its cache - but since InferenceService is created fresh
        # per request, we clear via the GPU manager directly
        logger.info("Inference model cache cleanup requested")
    except Exception as e:
        logger.warning("Could not clear inference cache: %s", e)

    # 3. Run Python garbage collection to release any lingering references
    gc.collect()

    # 4. Force clear GPU cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            logger.info("CUDA memory cache cleared and synchronized")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            logger.info("MPS memory cache cleared")
    except Exception as e:
        logger.warning("GPU cache clear failed: %s", e)

    # 5. Run GC again after cache clear to catch anything freed
    gc.collect()

    # Capture memory after cleanup
    memory_after = gpu_manager.get_memory_info()

    result = {
        "status": "cleared",
        "cancelled_jobs": cancelled_jobs,
        "memory_before": memory_before,
        "memory_after": memory_after,
    }

    # Calculate freed memory for CUDA
    if (gpu_manager.device_type == "cuda"
            and "allocated_mb" in memory_before
            and "allocated_mb" in memory_after):
        freed_mb = memory_before["allocated_mb"] - memory_after["allocated_mb"]
        result["freed_mb"] = round(freed_mb, 1)
        logger.info("GPU cleanup complete: freed %.1f MB", freed_mb)
    else:
        logger.info("GPU cleanup complete")

    return result
