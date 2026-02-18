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
    task.outputs["models"] = [m.to_dict() for m in models]
