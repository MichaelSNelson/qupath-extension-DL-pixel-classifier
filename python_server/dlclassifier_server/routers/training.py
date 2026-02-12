"""Training endpoints."""

import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()


class NormalizationConfig(BaseModel):
    """Normalization configuration."""
    strategy: str = "percentile_99"
    per_channel: bool = True
    clip_percentile: float = 99.0


class InputConfig(BaseModel):
    """Input configuration."""
    num_channels: int
    channel_names: List[str]
    bit_depth: int = 8
    normalization: NormalizationConfig


class ArchitectureConfig(BaseModel):
    """Model architecture configuration."""
    backbone: str = "resnet34"
    input_size: List[int] = [512, 512]
    use_pretrained: bool = True
    frozen_layers: Optional[List[str]] = None  # List of layer names to freeze


class TrainingParams(BaseModel):
    """Training hyperparameters."""
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    validation_split: float = 0.2
    augmentation: bool = True


class TrainRequest(BaseModel):
    """Training request."""
    model_type: str
    architecture: ArchitectureConfig
    input_config: InputConfig
    training: TrainingParams
    classes: List[str]
    data_path: str


class TrainResponse(BaseModel):
    """Training job response."""
    job_id: str
    status: str


class TrainStatusResponse(BaseModel):
    """Training status response."""
    status: str
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    train_loss: Optional[float] = None
    accuracy: Optional[float] = None
    model_path: Optional[str] = None
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    error: Optional[str] = None


@router.post("/train", response_model=TrainResponse)
async def start_training(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """Start a training job."""
    job_manager = request.app.state.job_manager
    gpu_manager = request.app.state.gpu_manager

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Create training job
    job = job_manager.create_training_job(
        job_id=job_id,
        model_type=train_request.model_type,
        architecture=train_request.architecture.model_dump(),
        input_config=train_request.input_config.model_dump(),
        training_params=train_request.training.model_dump(),
        classes=train_request.classes,
        data_path=train_request.data_path,
        device="cuda" if gpu_manager.is_available() else "cpu"
    )

    # Start training in background
    background_tasks.add_task(job.run)

    return TrainResponse(job_id=job_id, status="started")


@router.get("/train/{job_id}/status", response_model=TrainStatusResponse)
async def get_training_status(job_id: str, request: Request):
    """Get training job status."""
    job_manager = request.app.state.job_manager

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    status = job.get_status()
    return TrainStatusResponse(**status)


@router.post("/train/{job_id}/cancel")
async def cancel_training(job_id: str, request: Request):
    """Cancel a training job."""
    job_manager = request.app.state.job_manager

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job.cancel()
    return {"status": "cancelled", "job_id": job_id}
