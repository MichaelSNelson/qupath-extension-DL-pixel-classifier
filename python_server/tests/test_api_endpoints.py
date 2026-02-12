"""Tests for FastAPI endpoints.

Tests cover:
- Health check endpoints
- GPU status endpoint
- Pretrained model endpoints
- Training endpoints (validation only)
- Inference endpoints (validation only)
"""

import pytest

# Skip all tests if FastAPI test client not available
try:
    from fastapi.testclient import TestClient
    from dlclassifier_server.main import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None
    app = None


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI test client not available")
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_endpoint(self, client):
        """Test /health endpoint returns 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    @pytest.mark.skip(reason="Requires app.state.gpu_manager from lifespan")
    def test_gpu_endpoint(self, client):
        """Test /health/gpu endpoint."""
        response = client.get("/api/v1/gpu")
        assert response.status_code == 200

        data = response.json()
        assert "available" in data
        assert "device_type" in data
        assert data["device_type"] in ["cuda", "mps", "cpu"]

    @pytest.mark.skip(reason="Requires app.state.gpu_manager from lifespan")
    def test_gpu_endpoint_structure(self, client):
        """Test GPU endpoint returns correct structure."""
        response = client.get("/api/v1/gpu")
        data = response.json()

        # Required fields
        assert isinstance(data["available"], bool)
        assert "device_string" in data or "device_type" in data
        assert "name" in data


class TestPretrainedEndpoints:
    """Test pretrained model catalog endpoints."""

    def test_get_encoders(self, client):
        """Test /pretrained/encoders endpoint."""
        response = client.get("/api/v1/pretrained/encoders")
        assert response.status_code == 200

        encoders = response.json()
        assert isinstance(encoders, list)
        assert len(encoders) > 0

        # Check for common encoders
        encoder_names = [e["name"] for e in encoders]
        assert "resnet34" in encoder_names
        assert "mobilenet_v2" in encoder_names

    def test_get_architectures(self, client):
        """Test /pretrained/architectures endpoint."""
        response = client.get("/api/v1/pretrained/architectures")
        assert response.status_code == 200

        archs = response.json()
        assert isinstance(archs, list)
        assert len(archs) > 0

        # Check for common architectures
        arch_names = [a["name"] for a in archs]
        assert "unet" in arch_names

    def test_get_encoder_details(self, client):
        """Test /pretrained/encoder/{name} endpoint."""
        response = client.get("/api/v1/pretrained/encoder/resnet34")
        assert response.status_code == 200

        encoder = response.json()
        assert "name" in encoder
        assert encoder["name"] == "resnet34"

    def test_get_encoder_not_found(self, client):
        """Test encoder not found returns 404."""
        response = client.get("/api/v1/pretrained/encoder/nonexistent_encoder")
        assert response.status_code == 404

    def test_get_architecture_details(self, client):
        """Test /pretrained/architecture/{name} endpoint."""
        response = client.get("/api/v1/pretrained/architecture/unet")
        assert response.status_code == 200

        arch = response.json()
        assert "name" in arch
        assert arch["name"] == "unet"

    def test_get_freeze_recommendations(self, client):
        """Test /pretrained/freeze-recommendations endpoint."""
        response = client.get("/api/v1/pretrained/freeze-recommendations/small")
        assert response.status_code == 200

        recommendations = response.json()
        # Should return some form of recommendation
        assert recommendations is not None


class TestTrainingEndpoints:
    """Test training API endpoints (validation only, no actual training)."""

    def test_training_endpoint_validation(self, client):
        """Test training endpoint validates input."""
        # Empty body should fail validation
        response = client.post("/api/v1/train", json={})
        assert response.status_code == 422  # Validation error

    def test_training_requires_model_type(self, client):
        """Test training requires model_type field."""
        response = client.post("/api/v1/train", json={
            "architecture": {"backbone": "resnet34"},
            "classes": ["a", "b"]
        })
        # Should fail validation
        assert response.status_code == 422

    def test_training_requires_classes(self, client):
        """Test training requires classes field."""
        response = client.post("/api/v1/train", json={
            "model_type": "unet",
            "architecture": {"backbone": "resnet34"}
        })
        # Should fail validation
        assert response.status_code == 422


class TestInferenceEndpoints:
    """Test inference API endpoints (validation only)."""

    def test_inference_endpoint_validation(self, client):
        """Test inference endpoint validates input."""
        # Empty body should fail validation
        response = client.post("/api/v1/inference", json={})
        assert response.status_code == 422  # Validation error

    def test_inference_requires_model_path(self, client):
        """Test inference requires model_path field."""
        response = client.post("/api/v1/inference", json={
            "tiles": [{"id": "tile_0", "data": "xxx"}]
        })
        # Should fail validation
        assert response.status_code == 422

    def test_inference_requires_tiles(self, client):
        """Test inference requires tiles field."""
        response = client.post("/api/v1/inference", json={
            "model_path": "/path/to/model"
        })
        # Should fail validation
        assert response.status_code == 422

    def test_pixel_inference_endpoint_validation(self, client):
        """Test pixel inference endpoint validates input."""
        response = client.post("/api/v1/inference/pixel", json={})
        assert response.status_code == 422


class TestModelsEndpoints:
    """Test model management endpoints."""

    @pytest.mark.skip(reason="Requires app.state.model_registry from lifespan")
    def test_list_models(self, client):
        """Test /models endpoint returns list."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200

        models = response.json()
        assert isinstance(models, list)


class TestCORS:
    """Test CORS configuration."""

    def test_options_request(self, client):
        """Test OPTIONS request for CORS preflight."""
        response = client.options("/api/v1/health")
        # Should not error
        assert response.status_code in [200, 405]


class TestAPIVersioning:
    """Test API versioning."""

    def test_api_v1_prefix(self, client):
        """Test all endpoints use /api/v1 prefix."""
        # Health without prefix should 404
        response = client.get("/health")
        assert response.status_code == 404

        # With prefix should work
        response = client.get("/api/v1/health")
        assert response.status_code == 200


class TestErrorHandling:
    """Test API error handling."""

    def test_invalid_json(self, client):
        """Test invalid JSON returns proper error."""
        response = client.post(
            "/api/v1/train",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_method_not_allowed(self, client):
        """Test wrong HTTP method returns 405."""
        response = client.delete("/api/v1/health")
        assert response.status_code == 405
