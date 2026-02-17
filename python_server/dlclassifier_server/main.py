"""Main entry point for the DL Classifier server."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .routers import health, models, training, inference, pretrained

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting DL Classifier Server...")
    # Initialize model registry, GPU detection, etc.
    from .services.gpu_manager import GPUManager
    gpu_manager = GPUManager()
    app.state.gpu_manager = gpu_manager

    from .services.model_registry import ModelRegistry
    model_registry = ModelRegistry()
    app.state.model_registry = model_registry

    from .services.job_manager import JobManager
    job_manager = JobManager(model_registry=model_registry)
    app.state.job_manager = job_manager

    # Persistent inference service -- avoids model reload per request
    from .services.inference_service import InferenceService
    inference_service = InferenceService(device="auto", gpu_manager=gpu_manager)
    app.state.inference_service = inference_service
    logger.info("Persistent InferenceService created on device: %s",
                inference_service._device_str)

    yield

    logger.info("Shutting down DL Classifier Server...")
    # Cleanup
    job_manager.shutdown()


app = FastAPI(
    title="DL Pixel Classifier Server",
    description="Deep learning pixel classification service for QuPath",
    version="0.2.0",
    lifespan=lifespan
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(models.router, prefix="/api/v1", tags=["models"])
app.include_router(training.router, prefix="/api/v1", tags=["training"])
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])
app.include_router(pretrained.router, prefix="/api/v1", tags=["pretrained"])


def run():
    """Run the server.

    Passes the app object directly to uvicorn instead of an import string.
    Using a string causes uvicorn to spawn a subprocess on Windows, which
    breaks Ctrl+C signal handling.
    """
    import argparse
    parser = argparse.ArgumentParser(description="DL Pixel Classifier Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    run()
