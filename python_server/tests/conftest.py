"""Shared fixtures for dlclassifier_server tests.

This module provides pytest fixtures for:
- Test client for FastAPI endpoints
- Sample training data (synthetic images and masks)
- Sample trained model for inference tests
- Temporary directories for test outputs
"""

import json
import os
import shutil
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from PIL import Image

# Import test client only when available
try:
    from fastapi.testclient import TestClient
    from dlclassifier_server.main import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.fixture
def client():
    """FastAPI test client fixture with lifespan context.

    Uses context manager to properly initialize app state (model_registry, etc.)
    """
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI test client not available")

    # Use context manager to invoke lifespan
    with TestClient(app) as client:
        yield client


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_training_data(tmp_path) -> Path:
    """Generate synthetic training data for testing.

    Creates a minimal dataset with 6 images split into train/validation:
    - 4 training images with masks
    - 2 validation images with masks
    - config.json with class weights

    Returns:
        Path to the training data directory
    """
    data_dir = tmp_path / "training_data"

    # Create directory structure
    (data_dir / "train" / "images").mkdir(parents=True)
    (data_dir / "train" / "masks").mkdir(parents=True)
    (data_dir / "validation" / "images").mkdir(parents=True)
    (data_dir / "validation" / "masks").mkdir(parents=True)

    # Generate 6 synthetic images
    np.random.seed(42)  # Reproducible test data

    for i in range(6):
        # Create image with random background and circles
        img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
        mask = np.full((256, 256), 255, dtype=np.uint8)  # Start unlabeled

        # Add random circles (class 1 = foreground)
        for _ in range(np.random.randint(2, 5)):
            cx, cy = np.random.randint(30, 226, 2)
            r = np.random.randint(15, 40)
            y, x = np.ogrid[:256, :256]
            circle_mask = (x - cx)**2 + (y - cy)**2 <= r**2
            img[circle_mask] = [200, 100, 100]  # Reddish foreground
            mask[circle_mask] = 1

        # Add background annotation region (class 0)
        mask[0:50, 0:256] = 0  # Top strip is background

        # Determine split
        split = "train" if i < 4 else "validation"

        # Save image and mask
        Image.fromarray(img).save(data_dir / split / "images" / f"patch_{i:04d}.tiff")
        Image.fromarray(mask).save(data_dir / split / "masks" / f"patch_{i:04d}.png")

    # Create config.json
    config = {
        "classes": ["Background", "Foreground"],
        "class_weights": [1.0, 2.0],
        "unlabeled_index": 255
    }
    with open(data_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return data_dir


@pytest.fixture
def sample_tile_image(tmp_path) -> str:
    """Create a single sample tile image for inference testing.

    Returns:
        Path to the test tile image
    """
    np.random.seed(42)
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)

    # Add some structure
    img[100:150, 100:150] = [200, 100, 100]  # Red square

    tile_path = tmp_path / "test_tile.tiff"
    Image.fromarray(img).save(tile_path)

    return str(tile_path)


@pytest.fixture
def trained_model_path(tmp_path) -> Path:
    """Create a minimal trained model for inference testing.

    Creates a small mobilenet_v2-based UNet model with 2 classes.
    This is used for testing inference without requiring actual training.

    Returns:
        Path to the model directory
    """
    try:
        import torch
        import segmentation_models_pytorch as smp
    except ImportError:
        pytest.skip("PyTorch or smp not available")

    model_dir = tmp_path / "test_model"
    model_dir.mkdir()

    # Create a small model
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,  # No pretrained weights for faster test
        in_channels=3,
        classes=2
    )

    # Save PyTorch model
    torch.save(model.state_dict(), model_dir / "model.pt")

    # Create metadata
    metadata = {
        "id": "test_model",
        "name": "Test Classifier",
        "architecture": {
            "type": "unet",
            "backbone": "mobilenet_v2",
            "use_pretrained": False
        },
        "input_config": {
            "num_channels": 3,
            "normalization": {
                "strategy": "min_max"
            }
        },
        "classes": [
            {"index": 0, "name": "Background"},
            {"index": 1, "name": "Foreground"}
        ]
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return model_dir


@pytest.fixture
def training_config(sample_training_data) -> dict:
    """Create a minimal training configuration for testing.

    Uses mobilenet_v2 backbone for fast training and 2 epochs.

    Args:
        sample_training_data: Fixture providing training data path

    Returns:
        Training configuration dictionary
    """
    return {
        "model_type": "unet",
        "architecture": {
            "backbone": "mobilenet_v2",
            "use_pretrained": False  # Faster for testing
        },
        "input_config": {
            "num_channels": 3,
            "normalization": {
                "strategy": "min_max"
            }
        },
        "training_params": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "augmentation": False,  # Faster for testing
            "early_stopping": False,
            "scheduler": "none"
        },
        "classes": ["Background", "Foreground"],
        "data_path": str(sample_training_data)
    }


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU (CUDA or MPS) is available.

    Returns:
        True if GPU is available, False otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True
        return False
    except ImportError:
        return False


@pytest.fixture
def skip_without_gpu(gpu_available):
    """Skip test if no GPU is available."""
    if not gpu_available:
        pytest.skip("GPU not available")


@pytest.fixture
def skip_without_cuda():
    """Skip test if CUDA is not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def skip_without_mps():
    """Skip test if Apple MPS is not available."""
    try:
        import torch
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            pytest.skip("Apple MPS not available")
    except ImportError:
        pytest.skip("PyTorch not available")
