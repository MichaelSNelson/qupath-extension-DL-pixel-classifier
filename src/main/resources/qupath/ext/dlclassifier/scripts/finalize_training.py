"""
Finalize training from a checkpoint -- restore best model and save as final classifier.

Inputs:
    checkpoint_path: str - path to the training checkpoint file

Outputs:
    model_path: str - path to the saved model directory
    final_loss: float
    final_accuracy: float
    best_epoch: int
    best_mean_iou: float
    epochs_trained: int
"""
import torch
import logging

logger = logging.getLogger("dlclassifier.appose.finalize")

if inference_service is None:
    raise RuntimeError("Services not initialized: " + globals().get("init_error", "unknown"))

from dlclassifier_server.services.training_service import TrainingService

training_service = TrainingService(gpu_manager=gpu_manager)

logger.info("Loading checkpoint: %s", checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location=training_service.device, weights_only=False)

# Extract config from checkpoint
config = checkpoint["training_config"]
model_type = config["model_type"]
architecture = config["architecture"]
input_config = config["input_config"]
classes = config["classes"]

# Compute effective channels (handles context_scale)
context_scale = architecture.get("context_scale", 1)
base_channels = input_config["num_channels"]
effective_channels = base_channels * 2 if context_scale > 1 else base_channels

# Create model
model = training_service._create_model(model_type, architecture, effective_channels, len(classes))

# Restore best model weights
if "best_model_state" in checkpoint and checkpoint["best_model_state"] is not None:
    model.load_state_dict(checkpoint["best_model_state"])
    logger.info("Restored best model weights from checkpoint")
else:
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("No best model state in checkpoint, using last epoch weights")

model = model.to(training_service.device)

# Save as final model
model_path = training_service._save_model(
    model=model,
    model_type=model_type,
    architecture=architecture,
    input_config=input_config,
    classes=classes,
    data_path="",  # data may no longer exist
    training_history=checkpoint.get("training_history", []),
    normalization_stats=None
)

# Get metrics from checkpoint
history = checkpoint.get("training_history", [])
best_epoch = 0
best_loss = 0.0
best_accuracy = 0.0
best_mean_iou = 0.0

# Find best epoch from history (highest mean_iou)
for entry in history:
    entry_iou = entry.get("mean_iou", 0)
    if entry_iou >= best_mean_iou:
        best_mean_iou = entry_iou
        best_epoch = entry.get("epoch", 0)
        best_loss = entry.get("val_loss", 0)
        best_accuracy = entry.get("accuracy", 0)

task.outputs["model_path"] = model_path
task.outputs["final_loss"] = best_loss
task.outputs["final_accuracy"] = best_accuracy
task.outputs["best_epoch"] = best_epoch
task.outputs["best_mean_iou"] = best_mean_iou
task.outputs["epochs_trained"] = len(history)

logger.info("Training finalized from checkpoint. Model saved to %s", model_path)
