"""
MAE pretraining task for MuViT encoder with progress reporting via Appose.

Inputs:
    config: dict - model and training configuration
    data_path: str - directory of image tiles for pretraining
    output_dir: str - directory to save pretrained encoder weights

Outputs:
    status: str ("completed" or "cancelled")
    encoder_path: str - path to saved encoder weights
    epochs_completed: int
    final_loss: float
    best_loss: float
"""
import json
import logging
import threading
import time

logger = logging.getLogger("dlclassifier.appose.pretrain_mae")

if inference_service is None:
    raise RuntimeError(
        "Services not initialized: "
        + globals().get("init_error", "unknown"))

# Appose 0.10.0+: inputs are injected directly into script scope.
# Required: config, data_path, output_dir

import torch
from dlclassifier_server.services.mae_pretraining import MAEPretrainingService

mae_service = MAEPretrainingService(gpu_manager=gpu_manager)

# Log device info
device_name = str(mae_service.device)
cuda_available = torch.cuda.is_available()
device_info = "CPU"
logger.info("MAE pretraining device: %s (CUDA: %s)",
            device_name, cuda_available)
if mae_service._device_str == "cuda":
    device_info = torch.cuda.get_device_name(0)
    logger.info("GPU: %s", device_info)
elif mae_service._device_str == "cpu":
    logger.warning("Pretraining on CPU -- this will be extremely slow.")

total_epochs = config.get("epochs", 100)

# Initial status update
task.update(
    message=json.dumps({
        "status": "initializing",
        "device": mae_service._device_str,
        "device_info": device_info,
        "cuda_available": cuda_available,
        "epoch": 0,
        "total_epochs": total_epochs,
    }),
    current=0,
    maximum=total_epochs
)

logger.info("Config: %s", json.dumps(config, indent=2))
logger.info("Data path: %s", data_path)
logger.info("Output dir: %s", output_dir)


def setup_callback(phase, data=None):
    """Forward setup phase updates to Appose."""
    msg = {
        "status": "setup",
        "setup_phase": phase,
        "epoch": 0,
        "total_epochs": total_epochs,
    }
    if data:
        msg["config"] = data
    task.update(message=json.dumps(msg), current=0, maximum=total_epochs)


def progress_callback(epoch, total, loss, lr):
    """Forward training progress to Appose."""
    task.update(
        message=json.dumps({
            "epoch": epoch,
            "total_epochs": total,
            "train_loss": loss,
            "val_loss": loss,
            "accuracy": 0.0,
            "mean_iou": 0.0,
            "mae_lr": lr,
        }),
        current=epoch,
        maximum=total
    )


# Cancellation bridge
cancel_flag = threading.Event()


def watch_cancel():
    while not cancel_flag.is_set():
        if task.cancel_requested:
            cancel_flag.set()
            logger.info("MAE pretraining cancellation requested")
            break
        time.sleep(0.5)


cancel_watcher = threading.Thread(target=watch_cancel, daemon=True)
cancel_watcher.start()

try:
    result = mae_service.pretrain(
        config=config,
        data_path=data_path,
        output_dir=output_dir,
        progress_callback=progress_callback,
        setup_callback=setup_callback,
        cancel_flag=cancel_flag,
    )
except Exception as e:
    logger.error("MAE pretraining failed: %s", e)
    raise
finally:
    cancel_flag.set()

task.outputs["status"] = result.get("status", "completed")
task.outputs["encoder_path"] = result.get("encoder_path", "")
task.outputs["epochs_completed"] = result.get("epochs_completed", 0)
task.outputs["final_loss"] = result.get("final_loss", 0.0)
task.outputs["best_loss"] = result.get("best_loss", 0.0)
