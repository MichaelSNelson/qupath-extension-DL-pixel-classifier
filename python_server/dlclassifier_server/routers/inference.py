"""Inference endpoints."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()


class NormalizationConfig(BaseModel):
    """Normalization configuration."""
    strategy: str = "percentile_99"
    per_channel: bool = True
    clip_percentile: float = 99.0


class InferenceInputConfig(BaseModel):
    """Input configuration for inference."""
    num_channels: int
    selected_channels: List[int]
    normalization: NormalizationConfig


class TileData(BaseModel):
    """Tile data for inference."""
    id: str
    data: str  # Base64 encoded image or file path
    x: int
    y: int


class InferenceOptions(BaseModel):
    """Inference options."""
    use_gpu: bool = True
    blend_mode: str = "linear"


class InferenceRequest(BaseModel):
    """Inference request."""
    model_path: str
    input_config: InferenceInputConfig
    tiles: List[TileData]
    options: InferenceOptions


class InferenceResponse(BaseModel):
    """Inference response."""
    predictions: Dict[str, List[float]]


@router.post("/inference", response_model=InferenceResponse)
async def run_inference(
    inference_request: InferenceRequest,
    request: Request
):
    """Run inference on tiles."""
    gpu_manager = request.app.state.gpu_manager
    model_registry = request.app.state.model_registry

    # Determine device
    use_gpu = inference_request.options.use_gpu and gpu_manager.is_available()
    device = "cuda" if use_gpu else "cpu"

    try:
        # Load model
        from ..services.inference_service import InferenceService
        service = InferenceService(device=device)

        predictions = service.run_batch(
            model_path=inference_request.model_path,
            tiles=[t.model_dump() for t in inference_request.tiles],
            input_config=inference_request.input_config.model_dump()
        )

        return InferenceResponse(predictions=predictions)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {inference_request.model_path}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PixelInferenceRequest(BaseModel):
    """Request for pixel-level inference (returns spatial probability maps)."""
    model_path: str
    input_config: InferenceInputConfig
    tiles: List[TileData]
    output_dir: str
    options: InferenceOptions


class PixelInferenceResponse(BaseModel):
    """Response from pixel-level inference."""
    output_paths: Dict[str, str]
    num_classes: int


@router.post("/inference/pixel", response_model=PixelInferenceResponse)
async def run_pixel_inference(
    pixel_request: PixelInferenceRequest,
    request: Request
):
    """Run pixel-level inference, saving full probability maps to files.

    Unlike /inference which returns class-level averages, this endpoint
    saves per-pixel probability maps as numpy files (.npy) for use with
    tile blending on the Java side.
    """
    gpu_manager = request.app.state.gpu_manager

    use_gpu = pixel_request.options.use_gpu and gpu_manager.is_available()
    device = "cuda" if use_gpu else "cpu"

    try:
        from ..services.inference_service import InferenceService
        service = InferenceService(device=device)

        output_paths = service.run_pixel_inference(
            model_path=pixel_request.model_path,
            tiles=[t.model_dump() for t in pixel_request.tiles],
            input_config=pixel_request.input_config.model_dump(),
            output_dir=pixel_request.output_dir
        )

        # Determine num_classes from model metadata
        import json
        from pathlib import Path
        metadata_path = Path(pixel_request.model_path) / "metadata.json"
        num_classes = 2
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            num_classes = len(metadata.get("classes", [{"index": 0}, {"index": 1}]))

        return PixelInferenceResponse(
            output_paths=output_paths,
            num_classes=num_classes
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {pixel_request.model_path}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/batch")
async def run_batch_inference(
    model_path: str,
    tile_paths: List[str],
    request: Request
):
    """Run inference on a batch of tile files."""
    gpu_manager = request.app.state.gpu_manager

    use_gpu = gpu_manager.is_available()
    device = "cuda" if use_gpu else "cpu"

    try:
        from ..services.inference_service import InferenceService
        service = InferenceService(device=device)

        results = service.run_batch_files(
            model_path=model_path,
            tile_paths=tile_paths
        )

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
