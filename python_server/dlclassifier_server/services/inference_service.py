"""Inference service for deep learning models.

Supports:
- CUDA, Apple MPS, and CPU inference
- ONNX and PyTorch model loading with caching
- Batch and pixel-level inference modes
- Multiple normalization strategies
"""

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .gpu_manager import GPUManager, get_gpu_manager

logger = logging.getLogger(__name__)


class InferenceService:
    """Service for running model inference.

    Features:
    - Automatic device selection (CUDA > MPS > CPU)
    - Model caching for efficient batch processing
    - ONNX inference with appropriate execution providers
    - PyTorch inference as fallback
    - Multiple normalization strategies
    """

    def __init__(
        self,
        device: str = "auto",
        gpu_manager: Optional[GPUManager] = None
    ):
        """Initialize inference service.

        Args:
            device: Device to use ("cuda", "mps", "cpu", or "auto")
            gpu_manager: Optional GPUManager instance (uses singleton if not provided)
        """
        self.gpu_manager = gpu_manager or get_gpu_manager()

        if device == "auto":
            self.device = self.gpu_manager.device
            self._device_str = self.gpu_manager.device_type
        else:
            self._device_str = device
            self.device = torch.device(device)

        self._model_cache: Dict[str, Tuple[str, Any]] = {}
        self._onnx_providers = self._get_onnx_providers()

        logger.info(f"InferenceService initialized on device: {self._device_str}")

    def _get_onnx_providers(self) -> List[str]:
        """Get available ONNX execution providers based on device.

        Returns:
            List of ONNX execution provider names
        """
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except ImportError:
            logger.warning("ONNX Runtime not available")
            return ["CPUExecutionProvider"]

        if self._device_str == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self._device_str == "mps" and "CoreMLExecutionProvider" in available:
            # MPS devices can use CoreML for ONNX
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        return ["CPUExecutionProvider"]

    def run_batch(
        self,
        model_path: str,
        tiles: List[Dict[str, Any]],
        input_config: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """Run inference on a batch of tiles.

        Args:
            model_path: Path to the model directory
            tiles: List of tile dictionaries with 'id' and 'data' keys
            input_config: Input configuration (channels, normalization)

        Returns:
            Dict mapping tile_id to list of per-class probabilities
        """
        # Load model
        model = self._load_model(model_path)

        # Process tiles
        predictions = {}

        for tile in tiles:
            tile_id = tile["id"]
            tile_data = tile["data"]

            # Decode tile data
            if tile_data.startswith("data:") or "/" not in tile_data:
                # Base64 encoded
                img_array = self._decode_base64(tile_data)
            else:
                # File path
                img_array = self._load_image(tile_data)

            # Normalize
            img_array = self._normalize(img_array, input_config)

            # Select channels if needed
            selected = input_config.get("selected_channels")
            if selected:
                img_array = img_array[:, :, selected]

            # Run inference
            probs = self._infer_tile(model, img_array)
            predictions[tile_id] = probs.tolist()

        return predictions

    def run_pixel_inference(
        self,
        model_path: str,
        tiles: List[Dict[str, Any]],
        input_config: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, str]:
        """Run inference returning per-pixel probability maps saved as files.

        This mode is used for pixel classification (OBJECTS/OVERLAY output types)
        where full spatial probability maps are needed for tile blending.

        Args:
            model_path: Path to the model directory
            tiles: List of tile data dicts with 'id' and 'data' (file path or base64)
            input_config: Input configuration (channels, normalization)
            output_dir: Directory to save probability map files

        Returns:
            Dict mapping tile_id to output file path
        """
        model = self._load_model(model_path)
        os.makedirs(output_dir, exist_ok=True)

        output_paths = {}

        for tile in tiles:
            tile_id = tile["id"]
            tile_data = tile["data"]

            # Load tile image
            if tile_data.startswith("data:") or "/" not in tile_data:
                img_array = self._decode_base64(tile_data)
            else:
                img_array = self._load_image(tile_data)

            # Normalize
            img_array = self._normalize(img_array, input_config)

            # Select channels if needed
            selected = input_config.get("selected_channels")
            if selected:
                img_array = img_array[:, :, selected]

            # Run inference - get full spatial probability map
            prob_map = self._infer_tile_spatial(model, img_array)

            # Save as raw float32 binary (C, H, W order) for easy Java reading
            output_path = os.path.join(output_dir, f"{tile_id}.bin")
            prob_map.astype(np.float32).tofile(output_path)
            output_paths[tile_id] = output_path

        # Clear GPU cache after batch
        self._cleanup_after_inference()

        logger.info(f"Pixel inference complete: {len(output_paths)} tiles -> {output_dir}")
        return output_paths

    def run_batch_files(
        self,
        model_path: str,
        tile_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """Run inference on tile files.

        Args:
            model_path: Path to the model directory
            tile_paths: List of paths to tile image files

        Returns:
            List of result dictionaries with path and probabilities
        """
        model = self._load_model(model_path)

        results = []
        for path in tile_paths:
            img_array = self._load_image(path)

            # Basic normalization
            img_array = img_array.astype(np.float32) / 255.0

            probs = self._infer_tile(model, img_array)

            results.append({
                "path": path,
                "probabilities": probs.tolist()
            })

        return results

    def _cleanup_after_inference(self) -> None:
        """Clear GPU cache after inference batch."""
        self.gpu_manager.clear_cache()

    def _load_model(self, model_path: str) -> Tuple[str, Any]:
        """Load a model from disk.

        Prefers ONNX models for inference efficiency, falls back to PyTorch.

        Args:
            model_path: Path to model directory

        Returns:
            Tuple of (model_type, model) where model_type is "onnx" or "pytorch"
        """
        if model_path in self._model_cache:
            return self._model_cache[model_path]

        model_dir = Path(model_path)

        # Try ONNX first
        onnx_path = model_dir / "model.onnx"
        if onnx_path.exists():
            try:
                logger.info(f"Loading ONNX model from {onnx_path}")
                import onnxruntime as ort

                session = ort.InferenceSession(
                    str(onnx_path),
                    providers=self._onnx_providers
                )
                self._model_cache[model_path] = ("onnx", session)
                return ("onnx", session)
            except Exception as e:
                logger.warning(f"ONNX loading failed, trying PyTorch: {e}")

        # Try PyTorch
        pt_path = model_dir / "model.pt"
        if pt_path.exists():
            logger.info(f"Loading PyTorch model from {pt_path}")

            # Load metadata for model creation
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Create model architecture
            import segmentation_models_pytorch as smp

            arch = metadata.get("architecture", {})
            input_config = metadata.get("input_config", {})
            model_type = arch.get("type", "unet")
            encoder_name = arch.get("backbone", "resnet34")
            num_channels = input_config.get("num_channels", 3)
            num_classes = len(metadata.get("classes", [{"index": 0}, {"index": 1}]))

            model_map = {
                "unet": smp.Unet,
                "unetplusplus": smp.UnetPlusPlus,
                "deeplabv3": smp.DeepLabV3,
                "deeplabv3plus": smp.DeepLabV3Plus,
                "fpn": smp.FPN,
                "pspnet": smp.PSPNet,
                "manet": smp.MAnet,
                "linknet": smp.Linknet,
                "pan": smp.PAN,
            }

            model_cls = model_map.get(model_type, smp.Unet)
            model = model_cls(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=num_channels,
                classes=num_classes
            )

            model.load_state_dict(
                torch.load(pt_path, map_location=self.device, weights_only=True)
            )
            model = model.to(self.device)
            model.eval()

            self._model_cache[model_path] = ("pytorch", model)
            return ("pytorch", model)

        raise FileNotFoundError(f"No model found at {model_path}")

    def _infer_tile(self, model_tuple: Tuple[str, Any], img_array: np.ndarray) -> np.ndarray:
        """Run inference on a single tile, returning per-class average probabilities.

        Args:
            model_tuple: (model_type, model) from _load_model
            img_array: Image array (H, W, C) normalized

        Returns:
            Per-class average probabilities
        """
        prob_map = self._infer_tile_spatial(model_tuple, img_array)
        # Average over spatial dimensions to get per-class probabilities
        class_probs = prob_map.mean(axis=(1, 2))
        return class_probs

    def _infer_tile_spatial(
        self,
        model_tuple: Tuple[str, Any],
        img_array: np.ndarray
    ) -> np.ndarray:
        """Run inference on a single tile, returning full spatial probability map.

        Args:
            model_tuple: (model_type, model) from _load_model
            img_array: Image array (H, W, C) normalized

        Returns:
            Probability map with shape (C, H, W) where C is num_classes
        """
        model_type, model = model_tuple

        # Convert to tensor (HWC -> NCHW)
        if img_array.ndim == 2:
            img_array = img_array[..., np.newaxis]

        img_tensor = img_array.transpose(2, 0, 1)[np.newaxis, ...]
        img_tensor = img_tensor.astype(np.float32)

        if model_type == "onnx":
            # ONNX inference
            input_name = model.get_inputs()[0].name
            outputs = model.run(None, {input_name: img_tensor})
            logits = outputs[0]
        else:
            # PyTorch inference
            with torch.no_grad():
                tensor = torch.from_numpy(img_tensor).to(self.device)
                outputs = model(tensor)
                logits = outputs.cpu().numpy()

        # Softmax to get probabilities (C, H, W)
        probs = self._softmax(logits[0])

        return probs

    def _softmax(self, x: np.ndarray, axis: int = 0) -> np.ndarray:
        """Compute softmax.

        Args:
            x: Input logits
            axis: Axis along which to compute softmax

        Returns:
            Softmax probabilities
        """
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def _decode_base64(self, data: str) -> np.ndarray:
        """Decode base64 image data.

        Args:
            data: Base64 encoded image (with or without data URL prefix)

        Returns:
            Image as numpy array
        """
        # Remove data URL prefix if present
        if data.startswith("data:"):
            data = data.split(",")[1]

        img_bytes = base64.b64decode(data)
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img, dtype=np.float32)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from file.

        Args:
            path: Path to image file

        Returns:
            Image as numpy array
        """
        img = Image.open(path)
        return np.array(img, dtype=np.float32)

    def _normalize(self, img: np.ndarray, input_config: Dict[str, Any]) -> np.ndarray:
        """Normalize image data.

        Args:
            img: Input image array
            input_config: Configuration with normalization settings

        Returns:
            Normalized image array
        """
        norm_config = input_config.get("normalization", {})
        strategy = norm_config.get("strategy", "percentile_99")
        per_channel = norm_config.get("per_channel", False)

        if per_channel and img.ndim == 3 and img.shape[2] > 1:
            # Normalize each channel independently
            for c in range(img.shape[2]):
                img[..., c] = self._normalize_single(img[..., c], norm_config, strategy)
        else:
            img = self._normalize_single(img, norm_config, strategy)

        return img

    def _normalize_single(
        self,
        img: np.ndarray,
        norm_config: Dict[str, Any],
        strategy: str
    ) -> np.ndarray:
        """Normalize a single image or channel.

        Args:
            img: Image or channel array
            norm_config: Normalization configuration
            strategy: Normalization strategy name

        Returns:
            Normalized array
        """
        if strategy == "percentile_99":
            percentile = norm_config.get("clip_percentile", 99.0)
            p_min = np.percentile(img, 100 - percentile)
            p_max = np.percentile(img, percentile)
            img = np.clip(img, p_min, p_max)
            if p_max > p_min:
                img = (img - p_min) / (p_max - p_min)

        elif strategy == "min_max":
            i_min, i_max = img.min(), img.max()
            if i_max > i_min:
                img = (img - i_min) / (i_max - i_min)

        elif strategy == "z_score":
            mean, std = img.mean(), img.std()
            if std > 0:
                img = (img - mean) / std
                img = np.clip(img, -5, 5)
                # Rescale to 0-1 for model compatibility
                img = (img + 5) / 10

        elif strategy == "fixed_range":
            fixed_min = norm_config.get("min", 0)
            fixed_max = norm_config.get("max", 255)
            img = np.clip(img, fixed_min, fixed_max)
            if fixed_max > fixed_min:
                img = (img - fixed_min) / (fixed_max - fixed_min)

        return img

    def clear_model_cache(self) -> None:
        """Clear the model cache to free memory."""
        self._model_cache.clear()
        self._cleanup_after_inference()
        logger.info("Model cache cleared")
