"""
Training task with progress reporting via Appose events.

Inputs:
    model_type: str
    architecture: dict
    input_config: dict
    training_params: dict
    classes: list of str
    data_path: str

Outputs:
    model_path: str
    final_loss: float
    final_accuracy: float
    epochs_trained: int
"""
import json
import threading
import time
import logging

logger = logging.getLogger("dlclassifier.appose.train")

if inference_service is None:
    raise RuntimeError("Services not initialized: " + globals().get("_init_error", "unknown"))

# Appose 0.10.0+: inputs are injected directly into script scope (task.inputs is private).
# Required inputs: model_type, architecture, input_config, training_params, classes, data_path

# Import training service (heavier import, done here rather than init)
from dlclassifier_server.services.training_service import TrainingService
training_service = TrainingService(gpu_manager=gpu_manager)

# Log device and training configuration for diagnostics
import torch
device_name = training_service.device
cuda_available = torch.cuda.is_available()
device_info = "CPU"
logger.info("Training device: %s (CUDA available: %s)", device_name, cuda_available)
if device_name == "cuda":
    device_info = torch.cuda.get_device_name(0)
    logger.info("GPU: %s", device_info)
elif device_name == "cpu":
    logger.warning("Training on CPU -- this will be very slow. Check pixi.toml CUDA configuration.")
logger.info("Model: %s, backbone: %s", model_type, architecture.get("backbone", "unknown"))
logger.info("Classes: %s", classes)
logger.info("Epochs: %s, batch_size: %s, lr: %s",
    training_params.get("epochs"), training_params.get("batch_size"), training_params.get("learning_rate"))
logger.info("Data path: %s", data_path)

# Send pre-training status update so the Java UI can show device info
total_epochs = training_params.get("epochs", 50)
task.update(
    message=json.dumps({
        "status": "initializing",
        "device": device_name,
        "device_info": device_info,
        "cuda_available": cuda_available,
        "epoch": 0,
        "total_epochs": total_epochs,
    }),
    current=0,
    maximum=total_epochs
)


def progress_callback(epoch, train_loss, val_loss, accuracy,
                       per_class_iou, per_class_loss, mean_iou):
    """Forward training progress to Appose task events."""
    total_epochs = training_params.get("epochs", 50)
    task.update(
        message=json.dumps({
            "epoch": epoch,
            "total_epochs": total_epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "mean_iou": mean_iou,
            "per_class_iou": per_class_iou if per_class_iou else {},
            "per_class_loss": per_class_loss if per_class_loss else {},
        }),
        current=epoch,
        maximum=total_epochs
    )


# Set up cancellation bridge: Appose cancel -> threading.Event
cancel_flag = threading.Event()


def watch_cancel():
    """Poll for Appose cancellation request and set the cancel flag."""
    while not cancel_flag.is_set():
        if task.cancel_requested:
            cancel_flag.set()
            logger.info("Training cancellation requested via Appose")
            break
        time.sleep(0.5)


cancel_watcher = threading.Thread(target=watch_cancel, daemon=True)
cancel_watcher.start()

# Extract frozen layers from architecture dict (Java puts them there)
frozen_layers = architecture.get("frozen_layers", None)

result = training_service.train(
    model_type=model_type,
    architecture=architecture,
    input_config=input_config,
    training_params=training_params,
    classes=classes,
    data_path=data_path,
    progress_callback=progress_callback,
    cancel_flag=cancel_flag,
    frozen_layers=frozen_layers
)

task.outputs["model_path"] = result.get("model_path", "")
task.outputs["final_loss"] = result.get("final_loss", 0.0)
task.outputs["final_accuracy"] = result.get("final_accuracy", 0.0)
task.outputs["epochs_trained"] = result.get("epochs_trained", 0)
