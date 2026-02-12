"""Run integration test steps 2-4 using the model from Step 1.

This script uses the trained model saved by Step 1 to run:
- Step 2: Model prediction verification (direct service call)
- Step 3: FastAPI server training endpoint test (via TestClient)
- Step 4: FastAPI server inference endpoint test (via TestClient)

Usage:
    python tests/run_integration_steps_2_4.py <model_path>

Example:
    python tests/run_integration_steps_2_4.py ~/.dlclassifier/models/unet_20260211_172752
"""

import base64
import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("integration_test")

# Training data path
DATA_DIR = Path(__file__).parent.parent.parent / "dl_training" / "-CTRL_PknRNAi_108870_0001"
CLASSES = ["Ignore*", "hair", "intervein", "vein"]
NUM_CLASSES = 4

passed = 0
failed = 0


def encode_tile(img_path: Path) -> str:
    """Encode a tile image as base64 with data URI prefix."""
    img = Image.open(img_path)
    buffer = io.BytesIO()
    img.save(buffer, format="TIFF")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/tiff;base64,{b64}"


def run_step_2(model_path: str):
    """Step 2: Model prediction verification."""
    global passed, failed
    from dlclassifier_server.services.inference_service import InferenceService

    logger.info("=" * 70)
    logger.info("STEP 2: Model Prediction Verification")
    logger.info("=" * 70)
    logger.info(f"Model path: {model_path}")

    try:
        service = InferenceService(device="cpu")

        # Load all validation tiles
        val_images_dir = DATA_DIR / "validation" / "images"
        val_images = sorted(val_images_dir.glob("*.tiff"))
        logger.info(f"Found {len(val_images)} validation tiles")

        input_config = {
            "num_channels": 3,
            "normalization": {
                "strategy": "percentile_99",
                "per_channel": False,
                "clip_percentile": 99.0
            }
        }

        # Prepare tiles
        tiles = []
        for i, img_path in enumerate(val_images):
            b64_data = encode_tile(img_path)
            tiles.append({"id": f"val_{i:04d}", "data": b64_data, "x": 0, "y": 0})

        # Batch inference
        predictions = service.run_batch(
            model_path=model_path,
            tiles=tiles,
            input_config=input_config
        )

        assert len(predictions) == len(val_images), \
            f"Expected {len(val_images)} predictions, got {len(predictions)}"

        argmax_classes = []
        for tile_id, probs in predictions.items():
            probs = np.array(probs)

            # Shape check
            assert probs.shape == (NUM_CLASSES,), \
                f"Tile {tile_id}: expected ({NUM_CLASSES},), got {probs.shape}"

            # Softmax check
            prob_sum = probs.sum()
            assert abs(prob_sum - 1.0) < 0.01, \
                f"Tile {tile_id}: probs sum to {prob_sum:.4f}"

            # Valid range
            assert np.all(probs >= 0) and np.all(probs <= 1)
            assert np.all(np.isfinite(probs))

            argmax_class = int(np.argmax(probs))
            argmax_classes.append(argmax_class)
            logger.info(f"  {tile_id}: {probs.round(4).tolist()} "
                       f"-> {CLASSES[argmax_class]}")

        unique = set(argmax_classes)
        logger.info(f"\nClass diversity: {len(unique)} unique classes: "
                    f"{[CLASSES[c] for c in unique]}")

        # Step 2b: Spatial predictions
        logger.info("\nStep 2b: Spatial prediction test...")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "pixel_output")
            tile = [{"id": "spatial_test", "data": encode_tile(val_images[0]),
                      "x": 0, "y": 0}]

            output_paths = service.run_pixel_inference(
                model_path=model_path,
                tiles=tile,
                input_config=input_config,
                output_dir=output_dir
            )

            assert "spatial_test" in output_paths
            output_file = output_paths["spatial_test"]
            assert os.path.exists(output_file)

            prob_map = np.fromfile(output_file, dtype=np.float32)
            expected_size = NUM_CLASSES * 512 * 512
            assert prob_map.size == expected_size, \
                f"Expected {expected_size} elements, got {prob_map.size}"

            prob_map = prob_map.reshape(NUM_CLASSES, 512, 512)
            pixel_sums = prob_map.sum(axis=0)
            assert np.allclose(pixel_sums, 1.0, atol=0.01), \
                f"Pixel sums: [{pixel_sums.min():.4f}, {pixel_sums.max():.4f}]"

            argmax_map = np.argmax(prob_map, axis=0)
            unique_spatial = np.unique(argmax_map)
            logger.info(f"  Spatial output: {prob_map.shape}, "
                       f"classes: {[CLASSES[c] for c in unique_spatial]}")
            logger.info(f"  Pixel sum range: "
                       f"[{pixel_sums.min():.4f}, {pixel_sums.max():.4f}]")

        logger.info("[OK] Step 2 PASSED")
        passed += 1

    except Exception as e:
        logger.error(f"[FAIL] Step 2 FAILED: {e}", exc_info=True)
        failed += 1


def run_step_3():
    """Step 3: FastAPI server training endpoint test."""
    global passed, failed
    from fastapi.testclient import TestClient
    from dlclassifier_server.main import app

    logger.info("=" * 70)
    logger.info("STEP 3: FastAPI Server Training Endpoint Test")
    logger.info("=" * 70)

    try:
        with TestClient(app) as client:
            # Health check
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            logger.info(f"  Health: {response.json()}")

            # GPU info
            response = client.get("/api/v1/gpu")
            assert response.status_code == 200
            logger.info(f"  GPU: {response.json()}")

            # Submit training job (3 epochs, fast)
            train_request = {
                "model_type": "unet",
                "architecture": {
                    "backbone": "mobilenet_v2",
                    "input_size": [512, 512],
                    "use_pretrained": False
                },
                "input_config": {
                    "num_channels": 3,
                    "channel_names": ["Red", "Green", "Blue"],
                    "bit_depth": 8,
                    "normalization": {
                        "strategy": "percentile_99",
                        "per_channel": False,
                        "clip_percentile": 99.0
                    }
                },
                "training": {
                    "epochs": 3,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 1e-4,
                    "validation_split": 0.2,
                    "augmentation": False
                },
                "classes": CLASSES,
                "data_path": str(DATA_DIR)
            }

            logger.info("  Submitting training job (3 epochs)...")
            response = client.post("/api/v1/train", json=train_request)
            assert response.status_code == 200, \
                f"POST /train failed: {response.status_code} {response.text}"

            data = response.json()
            assert "job_id" in data
            assert data["status"] == "started"
            job_id = data["job_id"]
            logger.info(f"  Job created: {job_id}")

            # Poll for completion
            max_wait = 300
            poll_interval = 5
            elapsed = 0
            final_status = None

            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval

                response = client.get(f"/api/v1/train/{job_id}/status")
                assert response.status_code == 200
                status = response.json()

                if status["status"] == "training":
                    logger.info(f"  [{elapsed}s] Training: epoch={status.get('epoch')}, "
                               f"loss={status.get('loss')}, acc={status.get('accuracy')}")
                elif status["status"] == "completed":
                    final_status = status
                    logger.info(f"  [{elapsed}s] Completed!")
                    break
                elif status["status"] == "failed":
                    raise AssertionError(f"Training failed: {status.get('error')}")
                else:
                    logger.info(f"  [{elapsed}s] Status: {status['status']}")

            assert final_status is not None, f"Training didn't complete in {max_wait}s"
            assert final_status["status"] == "completed"
            assert final_status.get("model_path") is not None

            server_model_path = final_status["model_path"]
            logger.info(f"  Model: {server_model_path}")
            logger.info(f"  Loss: {final_status.get('final_loss')}")
            logger.info(f"  Accuracy: {final_status.get('final_accuracy')}")

            # Check models list
            response = client.get("/api/v1/models")
            assert response.status_code == 200
            logger.info(f"  Models endpoint: {response.status_code}")

        logger.info("[OK] Step 3 PASSED")
        passed += 1
        return server_model_path

    except Exception as e:
        logger.error(f"[FAIL] Step 3 FAILED: {e}", exc_info=True)
        failed += 1
        return None


def run_step_4(model_path: str):
    """Step 4: FastAPI server inference endpoint test."""
    global passed, failed
    from fastapi.testclient import TestClient
    from dlclassifier_server.main import app

    logger.info("=" * 70)
    logger.info("STEP 4: FastAPI Server Inference Endpoint Test")
    logger.info("=" * 70)
    logger.info(f"  Using model: {model_path}")

    try:
        with TestClient(app) as client:
            # Load 3 validation tiles
            val_images = sorted(
                (DATA_DIR / "validation" / "images").glob("*.tiff")
            )[:3]

            tiles = []
            for i, img_path in enumerate(val_images):
                tiles.append({
                    "id": f"tile_{i}",
                    "data": encode_tile(img_path),
                    "x": i * 512,
                    "y": 0
                })

            # 4a: Batch inference
            inference_request = {
                "model_path": model_path,
                "input_config": {
                    "num_channels": 3,
                    "selected_channels": [0, 1, 2],
                    "normalization": {
                        "strategy": "percentile_99",
                        "per_channel": False,
                        "clip_percentile": 99.0
                    }
                },
                "tiles": tiles,
                "options": {
                    "use_gpu": False,
                    "blend_mode": "linear"
                }
            }

            logger.info(f"  Sending {len(tiles)} tiles for batch inference...")
            response = client.post("/api/v1/inference", json=inference_request)
            assert response.status_code == 200, \
                f"POST /inference failed: {response.status_code} {response.text}"

            data = response.json()
            assert "predictions" in data
            predictions = data["predictions"]
            assert len(predictions) == len(tiles)

            for tile_id, probs in predictions.items():
                probs = np.array(probs)
                assert probs.shape == (NUM_CLASSES,)
                assert abs(probs.sum() - 1.0) < 0.01
                assert np.all(probs >= 0) and np.all(probs <= 1)
                assert np.all(np.isfinite(probs))
                logger.info(f"  {tile_id}: {probs.round(4).tolist()} "
                           f"-> {CLASSES[int(np.argmax(probs))]}")

            # 4b: Pixel inference
            logger.info("\n  Step 4b: Pixel inference endpoint...")
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = os.path.join(tmpdir, "server_pixel_out")
                pixel_request = {
                    "model_path": model_path,
                    "input_config": {
                        "num_channels": 3,
                        "selected_channels": [0, 1, 2],
                        "normalization": {
                            "strategy": "percentile_99",
                            "per_channel": False,
                            "clip_percentile": 99.0
                        }
                    },
                    "tiles": [{
                        "id": "pixel_0",
                        "data": encode_tile(val_images[0]),
                        "x": 0,
                        "y": 0
                    }],
                    "output_dir": output_dir,
                    "options": {
                        "use_gpu": False,
                        "blend_mode": "linear"
                    }
                }

                response = client.post("/api/v1/inference/pixel", json=pixel_request)
                assert response.status_code == 200, \
                    f"Pixel inference failed: {response.status_code} {response.text}"

                data = response.json()
                assert "output_paths" in data
                assert "num_classes" in data
                assert data["num_classes"] == NUM_CLASSES

                output_file = data["output_paths"]["pixel_0"]
                assert os.path.exists(output_file)

                prob_map = np.fromfile(output_file, dtype=np.float32)
                prob_map = prob_map.reshape(NUM_CLASSES, 512, 512)
                pixel_sums = prob_map.sum(axis=0)
                assert np.allclose(pixel_sums, 1.0, atol=0.01)

                logger.info(f"  Pixel output: {prob_map.shape}, "
                           f"sum range: [{pixel_sums.min():.4f}, {pixel_sums.max():.4f}]")

        logger.info("[OK] Step 4 PASSED")
        passed += 1

    except Exception as e:
        logger.error(f"[FAIL] Step 4 FAILED: {e}", exc_info=True)
        failed += 1


def main():
    global passed, failed

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_path>")
        print(f"Example: {sys.argv[0]} ~/.dlclassifier/models/unet_20260211_172752")
        sys.exit(1)

    model_path = os.path.expanduser(sys.argv[1])
    if not os.path.isdir(model_path):
        print(f"Error: model directory not found: {model_path}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("DL Pixel Classifier Integration Tests - Steps 2-4")
    logger.info("=" * 70)
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {DATA_DIR}")
    logger.info("")

    # Step 2: Direct inference verification
    run_step_2(model_path)

    # Step 3: Server training endpoint
    server_model = run_step_3()

    # Step 4: Server inference endpoint (use Step 1 model, it's better trained)
    run_step_4(model_path)

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Passed: {passed}")
    logger.info(f"  Failed: {failed}")
    logger.info("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
