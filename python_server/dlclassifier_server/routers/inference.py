"""Inference endpoints."""

import json as _json
from pathlib import Path as _Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Request, HTTPException, Form, File, UploadFile
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
    reflection_padding: int = 0
    gpu_batch_size: int = 16
    use_amp: bool = True
    compile_model: bool = True


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
    service = request.app.state.inference_service

    try:
        predictions = service.run_batch(
            model_path=inference_request.model_path,
            tiles=[t.model_dump() for t in inference_request.tiles],
            input_config=inference_request.input_config.model_dump(),
            gpu_batch_size=inference_request.options.gpu_batch_size,
            use_amp=inference_request.options.use_amp,
            compile_model=inference_request.options.compile_model,
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
    service = request.app.state.inference_service

    try:
        output_paths = service.run_pixel_inference(
            model_path=pixel_request.model_path,
            tiles=[t.model_dump() for t in pixel_request.tiles],
            input_config=pixel_request.input_config.model_dump(),
            output_dir=pixel_request.output_dir,
            reflection_padding=pixel_request.options.reflection_padding,
            gpu_batch_size=pixel_request.options.gpu_batch_size,
            use_amp=pixel_request.options.use_amp,
            compile_model=pixel_request.options.compile_model,
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
    service = request.app.state.inference_service

    try:
        results = service.run_batch_files(
            model_path=model_path,
            tile_paths=tile_paths
        )

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Binary Transfer Endpoints ====================


@router.post("/inference/binary", response_model=InferenceResponse)
async def run_inference_binary(
    metadata: str = Form(...),
    tiles: UploadFile = File(...),
    request: Request = None,
):
    """Run inference on tiles sent as raw binary data via multipart/form-data.

    This endpoint avoids the PNG+Base64 encoding overhead. Tiles are sent
    as a single concatenated binary blob of raw pixels (HWC order).

    The 'metadata' form field is a JSON string with:
    - model_path: str
    - input_config: InferenceInputConfig dict
    - options: InferenceOptions dict (optional)
    - tile_ids: list of str
    - tile_height: int
    - tile_width: int
    - num_channels: int
    - dtype: str ("uint8" or "float32", default "uint8")
    """
    service = request.app.state.inference_service

    try:
        meta = _json.loads(metadata)
        raw_bytes = await tiles.read()

        tile_ids = meta["tile_ids"]
        tile_height = meta["tile_height"]
        tile_width = meta["tile_width"]
        num_channels = meta["num_channels"]
        options = meta.get("options", {})

        # Pass dtype through to inference service via input_config
        input_config = {**meta["input_config"],
                        "dtype": meta.get("dtype", "uint8")}

        predictions = service.run_batch_from_buffer(
            model_path=meta["model_path"],
            raw_bytes=raw_bytes,
            tile_ids=tile_ids,
            tile_height=tile_height,
            tile_width=tile_width,
            num_channels=num_channels,
            input_config=input_config,
            gpu_batch_size=options.get("gpu_batch_size", 16),
            use_amp=options.get("use_amp", True),
            compile_model=options.get("compile_model", True),
        )

        return InferenceResponse(predictions=predictions)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Model not found: %s" % meta.get("model_path", "unknown")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/pixel/binary", response_model=PixelInferenceResponse)
async def run_pixel_inference_binary(
    metadata: str = Form(...),
    tiles: UploadFile = File(...),
    request: Request = None,
):
    """Run pixel-level inference on tiles sent as raw binary data.

    The 'metadata' form field is a JSON string with:
    - model_path: str
    - input_config: InferenceInputConfig dict
    - output_dir: str
    - options: InferenceOptions dict (optional)
    - tile_ids: list of str
    - tile_height: int
    - tile_width: int
    - num_channels: int
    - dtype: str ("uint8" or "float32", default "uint8")
    """
    service = request.app.state.inference_service

    try:
        meta = _json.loads(metadata)
        raw_bytes = await tiles.read()

        tile_ids = meta["tile_ids"]
        tile_height = meta["tile_height"]
        tile_width = meta["tile_width"]
        num_channels = meta["num_channels"]
        options = meta.get("options", {})

        # Pass dtype through to inference service via input_config
        input_config = {**meta["input_config"],
                        "dtype": meta.get("dtype", "uint8")}

        output_paths = service.run_pixel_inference_from_buffer(
            model_path=meta["model_path"],
            raw_bytes=raw_bytes,
            tile_ids=tile_ids,
            tile_height=tile_height,
            tile_width=tile_width,
            num_channels=num_channels,
            input_config=input_config,
            output_dir=meta["output_dir"],
            reflection_padding=options.get("reflection_padding", 0),
            gpu_batch_size=options.get("gpu_batch_size", 16),
            use_amp=options.get("use_amp", True),
            compile_model=options.get("compile_model", True),
        )

        # Determine num_classes from model metadata
        metadata_path = _Path(meta["model_path"]) / "metadata.json"
        num_classes = 2
        if metadata_path.exists():
            with open(metadata_path) as f:
                model_meta = _json.load(f)
            num_classes = len(model_meta.get("classes",
                                              [{"index": 0}, {"index": 1}]))

        return PixelInferenceResponse(
            output_paths=output_paths,
            num_classes=num_classes
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Model not found: %s" % meta.get("model_path", "unknown")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
