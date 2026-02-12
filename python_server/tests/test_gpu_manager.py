"""Tests for GPU manager service.

Tests cover:
- Device detection (CUDA, MPS, CPU)
- Memory information retrieval
- Cache clearing
- Model memory estimation
- API info structure
"""

import pytest


class TestGPUManager:
    """Test suite for GPUManager class."""

    def test_device_detection(self):
        """Verify correct device type detection."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()

        # Should detect one of the valid device types
        assert gm.device_type in ["cuda", "mps", "cpu"]

        # Device should be created
        import torch
        assert isinstance(gm.device, torch.device)

        # Device string should match type
        assert gm.get_device() == gm.device_type

    def test_is_available(self):
        """Test GPU availability check."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()

        if gm.device_type in ["cuda", "mps"]:
            assert gm.is_available() is True
        else:
            assert gm.is_available() is False

    def test_device_name(self):
        """Test device name retrieval."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()
        name = gm.get_device_name()

        assert isinstance(name, str)
        assert len(name) > 0

        # Name should make sense for device type
        if gm.device_type == "cpu":
            assert "CPU" in name
        elif gm.device_type == "mps":
            assert "MPS" in name or "Apple" in name

    def test_memory_info_structure(self):
        """Test memory info returns correct structure."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()
        mem_info = gm.get_memory_info()

        assert isinstance(mem_info, dict)
        assert "device" in mem_info

        if gm.device_type == "cuda":
            assert "allocated_mb" in mem_info
            assert "total_mb" in mem_info
            assert mem_info["allocated_mb"] >= 0
            assert mem_info["total_mb"] >= 0

    def test_memory_info_cuda(self, skip_without_cuda):
        """Test CUDA-specific memory reporting."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()
        info = gm.get_memory_info()

        assert info["device"] == "cuda"
        assert "allocated_mb" in info
        assert "reserved_mb" in info
        assert "max_allocated_mb" in info
        assert "total_mb" in info
        assert "utilization_percent" in info

        # All values should be non-negative
        assert info["allocated_mb"] >= 0
        assert info["reserved_mb"] >= 0
        assert info["total_mb"] > 0  # Should have some memory

    def test_cache_clear_no_raise(self):
        """Test GPU cache clearing doesn't raise exceptions."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()

        # Should not raise regardless of device type
        gm.clear_cache()

    def test_model_memory_estimate(self):
        """Test model memory estimation."""
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            pytest.skip("segmentation_models_pytorch not available")

        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()

        # Create a small model
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=2
        )

        mem_mb = gm.estimate_model_memory(model)

        assert mem_mb > 0
        # mobilenet_v2 UNet should be roughly 10-20MB
        assert mem_mb < 100

    def test_model_memory_estimate_larger_model(self):
        """Test memory estimation with larger model."""
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            pytest.skip("segmentation_models_pytorch not available")

        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()

        # Create a larger model
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=10
        )

        mem_mb = gm.estimate_model_memory(model)

        # resnet50 UNet should be larger than mobilenet_v2
        assert mem_mb > 50

    def test_get_info_structure(self):
        """Test GPU info API response structure."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()
        info = gm.get_info()

        # Required fields for all device types
        assert "available" in info
        assert "device_type" in info
        assert "device_string" in info
        assert "name" in info

        assert isinstance(info["available"], bool)
        assert info["device_type"] in ["cuda", "mps", "cpu"]

    def test_get_info_cuda_fields(self, skip_without_cuda):
        """Test CUDA-specific info fields."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()
        info = gm.get_info()

        assert info["device_type"] == "cuda"
        assert "cuda_version" in info
        assert "compute_capability" in info
        assert "total_memory_mb" in info
        assert info["total_memory_mb"] > 0

    def test_get_info_mps_fields(self, skip_without_mps):
        """Test MPS-specific info fields."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()
        info = gm.get_info()

        assert info["device_type"] == "mps"
        assert info["mps_available"] is True
        assert "backend" in info

    def test_log_memory_status_no_raise(self):
        """Test memory status logging doesn't raise."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()

        # Should not raise regardless of device type
        gm.log_memory_status()
        gm.log_memory_status(prefix="Test: ")

    def test_singleton_get_gpu_manager(self):
        """Test singleton pattern for get_gpu_manager."""
        from dlclassifier_server.services.gpu_manager import get_gpu_manager

        gm1 = get_gpu_manager()
        gm2 = get_gpu_manager()

        # Should return the same instance
        assert gm1 is gm2

    def test_memory_mb_cuda(self, skip_without_cuda):
        """Test CUDA memory MB value."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()
        mem_mb = gm.get_memory_mb()

        assert mem_mb > 0
        # Modern GPUs have at least 2GB
        assert mem_mb >= 2000

    def test_cuda_version(self, skip_without_cuda):
        """Test CUDA version retrieval."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()
        version = gm.get_cuda_version()

        assert version is not None
        assert isinstance(version, str)
        # CUDA version should be like "11.8" or "12.1"
        assert "." in version


class TestGPUManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_estimate_memory_empty_model(self):
        """Test memory estimation with model with no parameters."""
        import torch.nn as nn
        from dlclassifier_server.services.gpu_manager import GPUManager

        class EmptyModel(nn.Module):
            def forward(self, x):
                return x

        gm = GPUManager()
        model = EmptyModel()

        mem_mb = gm.estimate_model_memory(model)
        assert mem_mb == 0.0

    def test_device_type_property(self):
        """Test device_type property."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()

        # Should be accessible as property
        dt = gm.device_type
        assert dt == gm._device_type

    def test_device_property_lazy_init(self):
        """Test device property lazy initialization."""
        from dlclassifier_server.services.gpu_manager import GPUManager

        gm = GPUManager()

        # Access device - should work even if _device was None
        device = gm.device
        assert device is not None
