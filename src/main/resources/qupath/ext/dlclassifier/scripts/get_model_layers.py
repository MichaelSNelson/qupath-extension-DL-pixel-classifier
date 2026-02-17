"""
Get model layer structure for freeze/unfreeze configuration.

Inputs:
    architecture: str
    encoder: str
    num_channels: int
    num_classes: int

Outputs:
    layers: list of dicts with name, display_name, param_count, is_encoder,
            depth, recommended_freeze, description
"""
import logging

logger = logging.getLogger("dlclassifier.appose.model_layers")

architecture = task.inputs["architecture"]
encoder = task.inputs["encoder"]
num_channels = task.inputs["num_channels"]
num_classes = task.inputs["num_classes"]

try:
    from dlclassifier_server.services.pretrained_service import PretrainedService
    pretrained = PretrainedService()
    layers = pretrained.get_model_layers(architecture, encoder, num_channels, num_classes)
    task.outputs["layers"] = layers
except Exception as e:
    logger.error("Failed to get model layers: %s", e)
    task.outputs["layers"] = []
