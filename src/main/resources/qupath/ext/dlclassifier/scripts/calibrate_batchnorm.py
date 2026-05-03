"""
AdaBN ('Adaptive Batch Normalization') calibration.

Recomputes BatchNorm running statistics on a batch of tiles sampled
from the target image, so the model's encoder adapts to the new
acquisition's pixel statistics without any retraining or labels. The
mathematical update is the standard BN momentum-EMA: each forward pass
in train() mode nudges running_mean / running_var toward the batch's
empirical stats.

Inputs:
    model_path: str   -- absolute path to source model directory
    tile_paths: list  -- paths to tiles sampled from the target image
    output_dir: str   -- destination directory for calibrated model
    input_config: dict -- normalization config (same as inference)
    batch_size: int   -- forward-pass batch size (default 16)

Outputs:
    success: bool
    output_path: str  -- saved model.pt path on success
    n_tiles: int      -- number of tiles processed
    message: str      -- human-readable summary
"""
import logging
from contextlib import nullcontext

logger = logging.getLogger("dlclassifier.appose.calibrate_batchnorm")

if inference_service is None:
    raise RuntimeError(
        "Services not initialized: "
        + globals().get("init_error", "unknown"))

# Required inputs are injected directly into script scope by Appose.
_lock = globals().get("inference_lock", nullcontext())

try:
    with _lock:
        n = len(tile_paths) if tile_paths else 0
        if n == 0:
            task.outputs["success"] = False
            task.outputs["output_path"] = ""
            task.outputs["n_tiles"] = 0
            task.outputs["message"] = (
                "No tiles provided -- AdaBN needs at least a few tiles "
                "from the target image to recompute BN statistics.")
        else:
            bs = int(globals().get("batch_size", 16))
            saved = inference_service.calibrate_batchnorm(
                model_path=model_path,
                tile_paths=list(tile_paths),
                output_dir=output_dir,
                input_config=input_config or {},
                batch_size=bs,
            )
            task.outputs["success"] = True
            task.outputs["output_path"] = saved
            task.outputs["n_tiles"] = n
            task.outputs["message"] = (
                "Calibrated batchnorm on %d tiles; saved to %s" % (n, saved))
            logger.info(
                "AdaBN complete: %d tiles -> %s", n, saved)
except Exception as e:
    logger.error("AdaBN failed: %s", e)
    task.outputs["success"] = False
    task.outputs["output_path"] = ""
    task.outputs["n_tiles"] = 0
    task.outputs["message"] = "Error: %s" % str(e)
