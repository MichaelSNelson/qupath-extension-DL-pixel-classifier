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

# Appose 0.10.0+: inputs are injected directly into script scope (task.inputs is private).
# Required inputs: architecture, encoder, num_channels, num_classes

try:
    from dlclassifier_server.services.pretrained_models import PretrainedModelsService
    pretrained = PretrainedModelsService()
    layers = pretrained.get_model_layers(architecture, encoder, num_channels, num_classes)
    task.outputs["layers"] = layers
    if not layers:
        # Empty list may indicate an upstream error that was caught and
        # logged inside get_model_layers(). Surface it so the Java side
        # can warn the user instead of silently using its local fallback.
        logger.warning(
            "get_model_layers returned 0 layers for architecture=%s encoder=%s "
            "num_channels=%d num_classes=%d. Java will fall back to built-in "
            "layer defaults. Check worker log above for the root cause.",
            architecture, encoder, num_channels, num_classes)
except Exception as e:
    # Log full traceback so a regression like the num_channels free-variable
    # bug is not silent when it recurs.
    import traceback as _tb
    logger.error("Failed to get model layers: %s\n%s", e, _tb.format_exc())
    task.outputs["layers"] = []
