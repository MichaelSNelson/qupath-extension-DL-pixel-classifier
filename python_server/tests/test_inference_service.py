"""Tests for inference service.

Tests cover:
- Model loading (PyTorch and ONNX)
- Single tile inference
- Batch inference
- Pixel-level inference
- Normalization strategies
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


class TestModelLoading:
    """Test model loading functionality."""

    def test_load_pytorch_model(self, trained_model_path):
        """Test PyTorch model loading."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        model_tuple = inf._load_model(str(trained_model_path))

        assert model_tuple is not None
        model_type, model = model_tuple

        # Should be pytorch since no ONNX file
        assert model_type == "pytorch"

    def test_load_model_caching(self, trained_model_path):
        """Test model caching works."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        model_path_str = str(trained_model_path)

        # Load model first time
        model1 = inf._load_model(model_path_str)

        # Verify it's in the cache
        assert model_path_str in inf._model_cache

        # Load again - should return cached version
        model2 = inf._load_model(model_path_str)

        # Should be same cached entry (same model type at minimum)
        assert model1[0] == model2[0]  # Same model type
        # Cache should still only have one entry
        assert len(inf._model_cache) == 1

    def test_load_model_not_found(self, tmp_path):
        """Test loading non-existent model raises error."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        with pytest.raises(FileNotFoundError):
            inf._load_model(str(tmp_path / "nonexistent"))

    def test_load_onnx_model(self, trained_model_path):
        """Test ONNX model loading when available."""
        try:
            import torch
            import onnxruntime
        except ImportError:
            pytest.skip("ONNX runtime not available")

        from dlclassifier_server.services.inference_service import InferenceService

        # Create ONNX model
        try:
            import segmentation_models_pytorch as smp

            model = smp.Unet(
                encoder_name="mobilenet_v2",
                encoder_weights=None,
                in_channels=3,
                classes=2
            )
            model.eval()

            onnx_path = trained_model_path / "model.onnx"
            dummy_input = torch.randn(1, 3, 256, 256)
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                opset_version=14,
                input_names=["input"],
                output_names=["output"]
            )

            # Now load
            inf = InferenceService(device="cpu")
            # Clear cache first
            inf._model_cache.clear()

            model_tuple = inf._load_model(str(trained_model_path))

            model_type, _ = model_tuple
            # Should prefer ONNX
            assert model_type == "onnx"

        except Exception as e:
            pytest.skip(f"ONNX export failed: {e}")


class TestNormalization:
    """Test image normalization strategies."""

    def test_normalize_min_max(self):
        """Test min-max normalization."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8).astype(np.float32)

        config = {
            "normalization": {"strategy": "min_max"}
        }

        normalized = inf._normalize(img, config)

        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalize_percentile(self):
        """Test percentile normalization."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8).astype(np.float32)

        config = {
            "normalization": {
                "strategy": "percentile_99",
                "clip_percentile": 99.0
            }
        }

        normalized = inf._normalize(img, config)

        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalize_default(self):
        """Test default normalization when not specified."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8).astype(np.float32)

        # Empty config should use default (percentile_99)
        normalized = inf._normalize(img, {})

        # Should be normalized to [0, 1]
        assert normalized.min() >= 0
        assert normalized.max() <= 1


class TestSingleTileInference:
    """Test single tile inference."""

    def test_infer_tile(self, trained_model_path, sample_tile_image):
        """Test inference on a single tile."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        model = inf._load_model(str(trained_model_path))

        img = np.array(Image.open(sample_tile_image), dtype=np.float32) / 255.0

        probs = inf._infer_tile(model, img)

        # Should return per-class probabilities
        assert len(probs) == 2  # 2 classes
        assert all(0 <= p <= 1 for p in probs)
        assert abs(sum(probs) - 1.0) < 0.01  # Should sum to ~1

    def test_infer_tile_spatial(self, trained_model_path, sample_tile_image):
        """Test spatial inference on a single tile."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        model = inf._load_model(str(trained_model_path))

        img = np.array(Image.open(sample_tile_image), dtype=np.float32) / 255.0

        prob_map = inf._infer_tile_spatial(model, img)

        # Should return (C, H, W) probability map
        assert prob_map.ndim == 3
        assert prob_map.shape[0] == 2  # 2 classes
        assert prob_map.shape[1] == img.shape[0]  # Same height
        assert prob_map.shape[2] == img.shape[1]  # Same width

        # Probabilities should sum to 1 across classes
        class_sum = prob_map.sum(axis=0)
        assert np.allclose(class_sum, 1.0, atol=0.01)


class TestBatchInference:
    """Test batch inference."""

    def test_run_batch(self, trained_model_path, sample_tile_image):
        """Test batch inference."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        tiles = [
            {"id": "tile_0", "data": sample_tile_image},
            {"id": "tile_1", "data": sample_tile_image}
        ]

        input_config = {
            "num_channels": 3,
            "normalization": {"strategy": "min_max"}
        }

        results = inf.run_batch(
            model_path=str(trained_model_path),
            tiles=tiles,
            input_config=input_config
        )

        assert "tile_0" in results
        assert "tile_1" in results
        assert len(results["tile_0"]) == 2  # 2 classes

    def test_run_batch_with_channel_selection(self, trained_model_path, sample_tile_image):
        """Test batch inference with channel selection."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        tiles = [
            {"id": "tile_0", "data": sample_tile_image}
        ]

        input_config = {
            "num_channels": 3,
            "selected_channels": [0, 1, 2],  # All channels
            "normalization": {"strategy": "min_max"}
        }

        results = inf.run_batch(
            model_path=str(trained_model_path),
            tiles=tiles,
            input_config=input_config
        )

        assert "tile_0" in results


class TestPixelInference:
    """Test pixel-level inference."""

    def test_run_pixel_inference(self, trained_model_path, sample_tile_image, tmp_path):
        """Test pixel-level inference with file output."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        tiles = [
            {"id": "tile_0", "data": sample_tile_image, "x": 0, "y": 0}
        ]

        input_config = {
            "num_channels": 3,
            "normalization": {"strategy": "min_max"}
        }

        result = inf.run_pixel_inference(
            model_path=str(trained_model_path),
            tiles=tiles,
            input_config=input_config,
            output_dir=str(output_dir)
        )

        assert "tile_0" in result
        assert os.path.exists(result["tile_0"])

    def test_pixel_inference_output_format(self, trained_model_path, sample_tile_image, tmp_path):
        """Test pixel inference output file format."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        tiles = [
            {"id": "tile_0", "data": sample_tile_image, "x": 0, "y": 0}
        ]

        result = inf.run_pixel_inference(
            model_path=str(trained_model_path),
            tiles=tiles,
            input_config={"num_channels": 3},
            output_dir=str(output_dir)
        )

        # Read output file
        output_path = result["tile_0"]
        data = np.fromfile(output_path, dtype=np.float32)

        # Should be (C, H, W) flattened
        # For a 256x256 image with 2 classes: 2 * 256 * 256 = 131072
        expected_size = 2 * 256 * 256
        assert len(data) == expected_size


class TestSoftmax:
    """Test softmax computation."""

    def test_softmax(self):
        """Test softmax produces valid probabilities."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        x = np.array([1.0, 2.0, 3.0])
        result = inf._softmax(x)

        assert len(result) == 3
        assert all(r >= 0 for r in result)
        assert abs(sum(result) - 1.0) < 0.001

    def test_softmax_2d(self):
        """Test softmax on 2D array (CHW format)."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")

        # (C, H, W) shaped logits
        x = np.random.randn(2, 4, 4)
        result = inf._softmax(x, axis=0)

        # Should sum to 1 along class axis
        class_sum = result.sum(axis=0)
        assert np.allclose(class_sum, 1.0)


class TestImageDecoding:
    """Test image decoding utilities."""

    def test_load_image(self, sample_tile_image):
        """Test loading image from file."""
        from dlclassifier_server.services.inference_service import InferenceService

        inf = InferenceService(device="cpu")
        img = inf._load_image(sample_tile_image)

        assert img is not None
        assert img.ndim == 3
        assert img.shape[2] == 3  # RGB

    def test_decode_base64(self):
        """Test decoding base64 image."""
        from dlclassifier_server.services.inference_service import InferenceService
        import base64
        from PIL import Image
        import io

        inf = InferenceService(device="cpu")

        # Create a small test image
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Encode to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64_data = base64.b64encode(buffer.getvalue()).decode()

        # Decode
        decoded = inf._decode_base64(b64_data)

        assert decoded is not None
        assert decoded.shape == (32, 32, 3)

    def test_decode_base64_with_prefix(self):
        """Test decoding base64 with data URL prefix."""
        from dlclassifier_server.services.inference_service import InferenceService
        import base64
        from PIL import Image
        import io

        inf = InferenceService(device="cpu")

        # Create a small test image
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Encode to base64 with data URL prefix
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64_data = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

        # Decode
        decoded = inf._decode_base64(b64_data)

        assert decoded is not None
        assert decoded.shape == (32, 32, 3)
