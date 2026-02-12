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

    yield

    logger.info("Shutting down DL Classifier Server...")
    # Cleanup
    job_manager.shutdown()


app = FastAPI(
    title="DL Pixel Classifier Server",
    description="Deep learning pixel classification service for QuPath",
    version="0.1.0",
    lifespan=lifespan
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(models.router, prefix="/api/v1", tags=["models"])
app.include_router(training.router, prefix="/api/v1", tags=["training"])
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])
app.include_router(pretrained.router, prefix="/api/v1", tags=["pretrained"])


def run():
    """Run the server."""
    uvicorn.run(
        "dlclassifier_server.main:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    run()
