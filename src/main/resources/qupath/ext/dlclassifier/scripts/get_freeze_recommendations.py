"""
Get recommended freeze settings for a dataset size.

Inputs:
    dataset_size: str ("small", "medium", "large")
    encoder: str (optional)

Outputs:
    recommendations: dict mapping depth (int as str) -> bool
"""
import logging

logger = logging.getLogger("dlclassifier.appose.freeze_recs")

# Appose 0.10.0+: inputs are injected directly into script scope (task.inputs is private).
# Required inputs: dataset_size
# Optional inputs: encoder
try:
    encoder
except NameError:
    encoder = None

try:
    from dlclassifier_server.services.pretrained_models import PretrainedModelsService
    pretrained = PretrainedModelsService()
    recs = pretrained.get_freeze_recommendations(dataset_size, encoder=encoder)
    task.outputs["recommendations"] = recs
except Exception as e:
    logger.error("Failed to get freeze recommendations: %s", e)
    task.outputs["recommendations"] = {}
