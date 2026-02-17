"""
List available models from the model registry.

Outputs:
    models: list of dicts with id, name, type, path
"""
import logging

logger = logging.getLogger("dlclassifier.appose.list_models")

if model_registry is None:
    task.outputs["models"] = []
else:
    models = model_registry.list_models()
    task.outputs["models"] = [
        {
            "id": m.get("id", ""),
            "name": m.get("name", ""),
            "type": m.get("type", ""),
            "path": m.get("path", ""),
        }
        for m in models
    ]
