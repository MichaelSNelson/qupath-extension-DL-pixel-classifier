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

dataset_size = task.inputs["dataset_size"]
encoder = task.inputs.get("encoder", None)

try:
    from dlclassifier_server.services.pretrained_service import PretrainedService
    pretrained = PretrainedService()
    recs = pretrained.get_freeze_recommendations(dataset_size, encoder=encoder)
    task.outputs["recommendations"] = recs
except Exception as e:
    logger.error("Failed to get freeze recommendations: %s", e)
    task.outputs["recommendations"] = {}
