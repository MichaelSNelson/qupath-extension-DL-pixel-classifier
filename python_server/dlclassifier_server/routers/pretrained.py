"""Pretrained models API endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel

from ..services.pretrained_models import get_pretrained_service

router = APIRouter()


class EncoderResponse(BaseModel):
    """Response for encoder information."""
    name: str
    display_name: str
    family: str
    params_millions: float
    pretrained_weights: List[str]
    license: str
    recommended_for: List[str]


class ArchitectureResponse(BaseModel):
    """Response for architecture information."""
    name: str
    display_name: str
    description: str
    decoder_channels: List[int]
    supports_aux_output: bool


class LayerResponse(BaseModel):
    """Response for layer information."""
    name: str
    display_name: str
    param_count: int
    is_encoder: bool
    depth: int
    recommended_freeze: bool
    description: str


class FreezeRecommendationResponse(BaseModel):
    """Response for freeze recommendations."""
    dataset_size: str
    recommendations: dict


class ModelLayersRequest(BaseModel):
    """Request for getting model layers."""
    architecture: str
    encoder: str
    num_channels: int = 3
    num_classes: int = 2


@router.get("/pretrained/encoders", response_model=List[EncoderResponse])
async def list_encoders():
    """
    List available pretrained encoders.

    Returns encoders from segmentation-models-pytorch that can be used
    as backbones for semantic segmentation models. All encoders have
    ImageNet pretrained weights available.
    """
    service = get_pretrained_service()
    encoders = service.list_encoders()
    return [EncoderResponse(**e) for e in encoders]


@router.get("/pretrained/architectures", response_model=List[ArchitectureResponse])
async def list_architectures():
    """
    List available segmentation architectures.

    Returns decoder architectures (UNet, DeepLabV3+, FPN, etc.) that can
    be combined with any encoder for semantic segmentation.
    """
    service = get_pretrained_service()
    architectures = service.list_architectures()
    return [ArchitectureResponse(**a) for a in architectures]


@router.post("/pretrained/layers", response_model=List[LayerResponse])
async def get_model_layers(request: ModelLayersRequest):
    """
    Get the layer structure of a model for freeze/unfreeze configuration.

    This endpoint creates a temporary model to inspect its structure and
    returns information about each layer group that can be frozen for
    transfer learning.

    The layers are ordered from earliest (most general features) to latest
    (most task-specific features). Earlier layers typically capture basic
    visual features and can be frozen when fine-tuning on histopathology data.
    """
    service = get_pretrained_service()

    try:
        layers = service.get_model_layers(
            architecture=request.architecture,
            encoder=request.encoder,
            num_channels=request.num_channels,
            num_classes=request.num_classes
        )

        if not layers:
            raise HTTPException(
                status_code=500,
                detail="Could not inspect model layers. Ensure segmentation_models_pytorch is installed."
            )

        return [LayerResponse(**layer) for layer in layers]

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inspecting model: {str(e)}")


@router.get("/pretrained/freeze-recommendations/{dataset_size}",
            response_model=FreezeRecommendationResponse)
async def get_freeze_recommendations(
    dataset_size: str = Path(..., pattern="^(small|medium|large)$")
):
    """
    Get recommended layer freeze settings for transfer learning.

    The primary factor is what each layer learns and how well it transfers
    from ImageNet to histopathology (significant domain shift):

    - Depth 0-1: Universal features (edges, textures) - always freeze
    - Depth 2: Mid-level patterns - partial transfer, depends on data
    - Depth 3-4: Semantic features - ImageNet concepts don't apply, train
    - Depth 5+: Decoder - task-specific, always train

    Args:
        dataset_size: One of "small" (<500 tiles), "medium" (500-5000), "large" (>5000)
                      This affects overfitting risk when training more layers.

    Returns:
        Recommendations mapping layer depth to freeze (True) or train (False).
    """
    service = get_pretrained_service()
    recommendations = service.get_freeze_recommendations(dataset_size)

    return FreezeRecommendationResponse(
        dataset_size=dataset_size,
        recommendations=recommendations
    )


@router.get("/pretrained/encoder/{encoder_name}")
async def get_encoder_info(encoder_name: str):
    """Get detailed information about a specific encoder."""
    service = get_pretrained_service()
    encoders = {e["name"]: e for e in service.list_encoders()}

    if encoder_name not in encoders:
        raise HTTPException(
            status_code=404,
            detail=f"Encoder '{encoder_name}' not found. Use /pretrained/encoders to list available encoders."
        )

    return encoders[encoder_name]


@router.get("/pretrained/architecture/{architecture_name}")
async def get_architecture_info(architecture_name: str):
    """Get detailed information about a specific architecture."""
    service = get_pretrained_service()
    architectures = {a["name"]: a for a in service.list_architectures()}

    if architecture_name not in architectures:
        raise HTTPException(
            status_code=404,
            detail=f"Architecture '{architecture_name}' not found. Use /pretrained/architectures to list available."
        )

    return architectures[architecture_name]
