"""
Batch pixel-level inference via Appose.

Saves probability maps to disk as .bin files (same format as HTTP backend).
Used when multiple tiles need spatial probability maps and the caller
expects file-based output (e.g. InferenceWorkflow for OBJECTS output).

Inputs:
    model_path: str
    tile_data: NDArray - concatenated tiles (N*H*W*C) float32
    tile_ids: list of str
    tile_height: int
    tile_width: int
    num_channels: int
    input_config: dict
    output_dir: str
    reflection_padding: int (default 0)

Outputs:
    output_paths: dict mapping tile_id -> output file path
    num_classes: int
"""
import os
import numpy as np
import logging

logger = logging.getLogger("dlclassifier.appose.inference_pixel_batch")

if inference_service is None:
    raise RuntimeError("Inference service not initialized: " + globals().get("_init_error", "unknown"))

model_path = task.inputs["model_path"]
tile_nd = task.inputs["tile_data"]
tile_ids = task.inputs["tile_ids"]
tile_height = task.inputs["tile_height"]
tile_width = task.inputs["tile_width"]
num_channels = task.inputs["num_channels"]
input_config = task.inputs["input_config"]
output_dir = task.inputs["output_dir"]
reflection_padding = task.inputs.get("reflection_padding", 0)

num_tiles = len(tile_ids)
os.makedirs(output_dir, exist_ok=True)

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

# Batched inference with reflection padding
model_tuple = inference_service._load_model(model_path)
all_prob_maps = inference_service._infer_batch_spatial(
    model_tuple, preprocessed,
    reflection_padding=reflection_padding
)

# Save probability maps to disk
output_paths = {}
num_classes = 0
for tile_id, prob_map in zip(tile_ids, all_prob_maps):
    num_classes = prob_map.shape[0]
    output_path = os.path.join(output_dir, "%s.bin" % tile_id)
    prob_map.astype(np.float32).tofile(output_path)
    output_paths[tile_id] = output_path

task.outputs["output_paths"] = output_paths
task.outputs["num_classes"] = num_classes

inference_service._cleanup_after_inference()
