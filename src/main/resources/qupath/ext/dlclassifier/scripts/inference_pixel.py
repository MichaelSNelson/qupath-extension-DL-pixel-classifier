"""
Per-tile pixel inference via Appose shared memory.

Inputs (from Java):
    model_path: str - path to model directory
    tile_data: NDArray - shared memory tile (H, W, C) float32
    tile_height: int
    tile_width: int
    num_channels: int
    input_config: dict - normalization config
    reflection_padding: int (default 0)

Outputs:
    probabilities: NDArray - shared memory probability map (C, H, W) float32
    num_classes: int
"""
import numpy as np
import logging

logger = logging.getLogger("dlclassifier.appose.inference")

# Access persistent globals from init script
if inference_service is None:
    raise RuntimeError("Inference service not initialized: " + globals().get("init_error", "unknown"))

# Appose 0.10.0+: inputs are injected directly into script scope (task.inputs is private).
# Required inputs: model_path, tile_data, tile_height, tile_width, num_channels, input_config
# Optional inputs: reflection_padding
tile_nd = tile_data
try:
    reflection_padding
except NameError:
    reflection_padding = 0

# Zero-copy read from shared memory NDArray.
# Copy immediately so the shared memory segment can be reused by Java.
tile_array = tile_nd.ndarray().reshape(tile_height, tile_width, num_channels).copy()

# Normalize (thread-safe, no GPU access)
tile_array = inference_service._normalize(tile_array, input_config)

# Select channels if specified
selected = input_config.get("selected_channels")
if selected:
    tile_array = tile_array[:, :, selected]

# Serialize GPU access. Appose runs each task in its own thread, so
# without this lock, 10+ overlay threads would race on model loading,
# CUDA memory, and forward passes simultaneously. The GPU can only run
# one batch at a time anyway, so serializing here prevents OOM and
# thread-safety issues (torch.compile is NOT thread-safe).
with inference_lock:
    model_tuple = inference_service._load_model(model_path)
    prob_maps = inference_service._infer_batch_spatial(
        model_tuple, [tile_array],
        reflection_padding=reflection_padding,
        gpu_batch_size=1
    )
    prob_map = prob_maps[0]  # (C, H, W) float32
    inference_service._cleanup_after_inference()

num_classes = prob_map.shape[0]

# Write result to shared memory NDArray (outside lock -- no GPU needed).
# NDArray auto-creates SharedMemory of the correct size when shm=None.
from appose import NDArray as PyNDArray

out_nd = PyNDArray(dtype="float32", shape=[num_classes, tile_height, tile_width])
np.copyto(out_nd.ndarray(), prob_map)

task.outputs["probabilities"] = out_nd
task.outputs["num_classes"] = num_classes
