"""Model registry service."""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a registered model."""
    id: str
    name: str
    type: str
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "metadata": self.metadata
        }


class ModelRegistry:
    """Registry for managing classifier models."""

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.expanduser("~/.dlclassifier/models")

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelInfo] = {}
        self._scan_models()

    def _scan_models(self):
        """Scan models directory for existing models."""
        logger.info(f"Scanning models directory: {self.models_dir}")

        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            metadata = json.load(f)

                        model_info = ModelInfo(
                            id=metadata.get("id", model_dir.name),
                            name=metadata.get("name", model_dir.name),
                            type=metadata.get("architecture", {}).get("type", "unknown"),
                            path=str(model_dir),
                            metadata=metadata
                        )
                        self._models[model_info.id] = model_info
                        logger.info(f"Loaded model: {model_info.name}")

                    except Exception as e:
                        logger.warning(f"Failed to load model from {model_dir}: {e}")

        logger.info(f"Found {len(self._models)} models")

    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        return list(self._models.values())

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a specific model by ID."""
        return self._models.get(model_id)

    def register_model(
        self,
        model_id: str,
        name: str,
        model_type: str,
        model_path: str,
        metadata: Dict[str, Any]
    ) -> ModelInfo:
        """Register a new model."""
        # Create model directory
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)

        # Copy model file if needed (skip if model_path is already the model directory)
        src_path = Path(model_path)
        if src_path.is_file() and src_path.parent != model_dir:
            dest_path = model_dir / src_path.name
            shutil.copy(src_path, dest_path)
            model_path = str(dest_path)
        elif src_path.is_dir():
            model_path = str(src_path)

        # Save metadata
        metadata["id"] = model_id
        metadata["name"] = name
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create model info
        model_info = ModelInfo(
            id=model_id,
            name=name,
            type=model_type,
            path=str(model_dir),
            metadata=metadata
        )
        self._models[model_id] = model_info

        logger.info(f"Registered model: {name} ({model_id})")
        return model_info

    async def upload_model(self, file) -> ModelInfo:
        """Upload and register an ONNX model."""
        import uuid

        model_id = str(uuid.uuid4())[:8]
        model_name = Path(file.filename).stem

        # Create model directory
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)

        # Save model file
        model_path = model_dir / file.filename
        content = await file.read()
        with open(model_path, "wb") as f:
            f.write(content)

        # Create basic metadata
        metadata = {
            "id": model_id,
            "name": model_name,
            "architecture": {"type": "custom_onnx"},
            "source": "uploaded"
        }

        return self.register_model(
            model_id=model_id,
            name=model_name,
            model_type="custom_onnx",
            model_path=str(model_path),
            metadata=metadata
        )

    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        model = self._models.get(model_id)
        if model is None:
            return False

        try:
            # Remove directory
            shutil.rmtree(model.path)
            del self._models[model_id]
            logger.info(f"Deleted model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
