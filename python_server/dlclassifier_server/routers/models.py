"""Model management endpoints."""

from typing import List
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter()


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    name: str
    type: str
    path: str


class ModelsResponse(BaseModel):
    """Response for listing models."""
    models: List[ModelInfo]


@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request):
    """List available models."""
    model_registry = request.app.state.model_registry

    models = model_registry.list_models()
    return ModelsResponse(models=[
        ModelInfo(id=m.id, name=m.name, type=m.type, path=m.path)
        for m in models
    ])


@router.get("/models/{model_id}")
async def get_model(model_id: str, request: Request):
    """Get details for a specific model."""
    model_registry = request.app.state.model_registry

    model = model_registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return model.to_dict()


@router.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    request: Request = None
):
    """Upload a custom ONNX model."""
    model_registry = request.app.state.model_registry

    if not file.filename.endswith(".onnx"):
        raise HTTPException(status_code=400, detail="Only ONNX files are supported")

    try:
        model_info = await model_registry.upload_model(file)
        return {"status": "success", "model_id": model_info.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model(model_id: str, request: Request):
    """Delete a model."""
    model_registry = request.app.state.model_registry

    success = model_registry.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return {"status": "deleted", "model_id": model_id}
