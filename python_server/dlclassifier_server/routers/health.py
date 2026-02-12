"""Health check endpoints."""

from fastapi import APIRouter, Request

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
