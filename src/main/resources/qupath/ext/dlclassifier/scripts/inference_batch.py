"""
Batch inference via Appose shared memory.

Returns per-class average probabilities (not spatial maps).
Used for MEASUREMENTS output type.

Inputs:
    model_path: str
    tile_data: NDArray - concatenated tiles (N*H*W*C) float32
    tile_ids: list of str
    tile_height: int
    tile_width: int
    num_channels: int
    input_config: dict
    dtype: str ("uint8" or "float32")

Outputs:
    predictions: dict mapping tile_id -> list of per-class probabilities
"""
import numpy as np
import logging

logger = logging.getLogger("dlclassifier.appose.inference_batch")

if inference_service is None:
    raise RuntimeError("Inference service not initialized: " + globals().get("init_error", "unknown"))

# Appose 0.10.0+: inputs are injected directly into script scope (task.inputs is private).
# Required inputs: model_path, tile_data, tile_ids, tile_height, tile_width, num_channels, input_config
tile_nd = tile_data

num_tiles = len(tile_ids)

# Read from shared memory
raw = tile_nd.ndarray().reshape(num_tiles, tile_height, tile_width, num_channels).copy()

# Normalize and select channels
selected = input_config.get("selected_channels")
preprocessed = []
for i in range(num_tiles):
    img = inference_service._normalize(raw[i], input_config)
    if selected:
        img = img[:, :, selected]
    preprocessed.append(img)

# Serialize GPU access. Appose runs each task in its own thread, so
# without this lock, concurrent tasks would race on model loading,
# CUDA memory, and forward passes.
with inference_lock:
    model_tuple = inference_service._load_model(model_path)
    all_prob_maps = inference_service._infer_batch_spatial(model_tuple, preprocessed)
    inference_service._cleanup_after_inference()

# Average spatial dims for per-class probabilities (outside lock -- CPU-only numpy)
predictions = {}
for tile_id, prob_map in zip(tile_ids, all_prob_maps):
    class_probs = prob_map.mean(axis=(1, 2))
    predictions[tile_id] = class_probs.tolist()

task.outputs["predictions"] = predictions
