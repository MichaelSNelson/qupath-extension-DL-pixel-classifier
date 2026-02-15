# DL Classifier Server

Python FastAPI server for deep learning pixel classification, designed to work with the QuPath DL Pixel Classifier extension.

## Features

- **Multi-device support**: CUDA, Apple Silicon (MPS), and CPU fallback
- **GPU memory management**: Automatic cache clearing and memory monitoring
- **Multiple architectures**: UNet, UNet++, DeepLabV3+, FPN, PSPNet, MANet, LinkNet, PAN
- **Pretrained encoders**: ResNet, EfficientNet, MobileNet, DenseNet, VGG, and more
- **Histology-pretrained encoders**: ResNet-50 models pretrained on TCGA/Lunit/Kather100K tissue patches via timm
- **Transfer learning**: Layer-level freeze/unfreeze control with encoder-aware recommendations
- **ONNX export**: Automatic export for deployment without Python server
- **Sparse annotation support**: Works with line/brush annotations (UNLABELED_INDEX=255)

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA-capable GPU (recommended, but not required)

## Installation

### Step 1: Navigate to the python_server directory

```bash
cd python_server
```

### Step 2: Create a Virtual Environment (Recommended)

Using a virtual environment avoids conflicts with system packages and is required on some Linux distributions.

**Linux/macOS:**

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
```

```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
python -m venv venv
```

```powershell
venv\Scripts\Activate.ps1
```

> **Note (Windows PowerShell):** If you get an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3: Install the Package

**Standard installation:**

```bash
pip install -e .
```

**With CUDA support (NVIDIA GPUs):**

```bash
pip install -e ".[cuda]"
```

**Development installation (includes test dependencies):**

```bash
pip install -e ".[dev]"
```

### Platform-Specific Notes

| Platform | Notes |
|----------|-------|
| **Linux** | Some distributions (Ubuntu 23.04+, Fedora 38+) require virtual environments due to PEP 668. Use `python3` instead of `python`. |
| **macOS** | Apple Silicon (M1/M2/M3) uses MPS acceleration, requires macOS 12.3+. Use `python3` or install Python via Homebrew. |
| **Windows** | Use `python` (not `python3`). Ensure Python is added to PATH during installation. |

### PyTorch with Specific CUDA Versions

If you need a specific CUDA version, install PyTorch first before installing this package:

**CUDA 11.8:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the server package:

```bash
pip install -e .
```

## Running the Server

```bash
dlclassifier-server
```

The server starts on `http://localhost:8765` by default.

### Verify Server Status

**Health check:**

```bash
curl http://localhost:8765/api/v1/health
```

**GPU info:**

```bash
curl http://localhost:8765/api/v1/gpu
```

> **Note (Windows):** If `curl` is not available, use PowerShell:
> ```powershell
> Invoke-WebRequest -Uri http://localhost:8765/api/v1/health
> ```
> Or open `http://localhost:8765/docs` in a browser to use the interactive Swagger UI.

## Testing

### Prerequisites

Ensure you have installed with dev dependencies:

```bash
pip install -e ".[dev]"
```

### Running All Tests

```bash
pytest tests/ -v
```

### Running Specific Test Files

**GPU manager tests:**

```bash
pytest tests/test_gpu_manager.py -v
```

**Training service tests:**

```bash
pytest tests/test_training_service.py -v
```

**Inference service tests:**

```bash
pytest tests/test_inference_service.py -v
```

**API endpoint tests:**

```bash
pytest tests/test_api_endpoints.py -v
```

### Running with Coverage

```bash
pytest tests/ --cov=dlclassifier_server --cov-report=html
```

Coverage report will be generated in `htmlcov/index.html`.

### Test Status

Current test suite: **78 passed, 5 skipped**

Skipped tests require app.state initialization from FastAPI lifespan context (GPU endpoint tests that need full server startup).

## Training Data Format

### Directory Structure

Training data must be organized in the following structure:

```
training_data/
  config.json                   # Configuration file (see below)
  train/
    images/
      patch_0000.tiff           # RGB or multi-channel TIFF images
      patch_0001.tiff
      ...
    masks/
      patch_0000.png            # Single-channel masks
      patch_0001.png
      ...
  validation/
    images/
      patch_0004.tiff
      ...
    masks/
      patch_0004.png
      ...
```

### config.json Format

```json
{
  "classes": ["Background", "Foreground"],
  "class_weights": [1.0, 2.0],
  "unlabeled_index": 255
}
```

| Field | Description |
|-------|-------------|
| `classes` | List of class names (order = index) |
| `class_weights` | Optional inverse-frequency weights for class balancing |
| `unlabeled_index` | Pixel value for unlabeled regions (typically 255) |

### Image Requirements

| Property | Specification |
|----------|---------------|
| **Format** | TIFF (preferred), PNG, or JPEG |
| **Size** | Typically 256x256 or 512x512 pixels |
| **Channels** | 1-N (grayscale, RGB, or multi-channel) |
| **Bit depth** | 8-bit or 16-bit |

### Mask Requirements

| Property | Specification |
|----------|---------------|
| **Format** | PNG (single-channel) |
| **Size** | Must match corresponding image |
| **Values** | 0 = class 0, 1 = class 1, ..., 255 = unlabeled |
| **Type** | uint8 |

### Sparse Annotation Support

The training system supports sparse annotations where only part of each image is labeled:

- **Unlabeled pixels**: Set to 255 (or value specified in `unlabeled_index`)
- **Loss computation**: Uses `ignore_index=255` in CrossEntropyLoss
- **Class weights**: Calculated from labeled pixels only

This allows training from line/brush annotations without requiring full image segmentation.

### Generating Synthetic Test Data

For testing, you can generate synthetic data.

**Linux/macOS:**

```bash
python3 tests/generate_test_data.py
```

**Windows:**

```cmd
python tests/generate_test_data.py
```

This creates 6 synthetic images (256x256 RGB) with random circles in `tests/test_data/`. The data includes:
- 4 training images with masks
- 2 validation images with masks
- `config.json` with class weights

## Training Request Format

### POST /api/v1/train

```json
{
  "model_type": "unet",
  "architecture": {
    "backbone": "resnet34",
    "use_pretrained": true,
    "frozen_layers": ["encoder.conv1", "encoder.layer1"]
  },
  "input_config": {
    "num_channels": 3,
    "normalization": {
      "strategy": "percentile_99",
      "per_channel": true,
      "clip_percentile": 99.0
    }
  },
  "training_params": {
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "augmentation": true,
    "early_stopping": true,
    "patience": 10,
    "scheduler": "cosine"
  },
  "classes": ["Background", "Foreground"],
  "data_path": "/path/to/training_data"
}
```

### Available Options

**Architectures** (`model_type`):
- `unet`, `unet++`, `deeplabv3`, `deeplabv3+`, `fpn`, `pspnet`, `manet`, `linknet`, `pan`

**Encoders** (`backbone`):
- ImageNet-pretrained: `resnet34`, `resnet50`, `efficientnet-b0`, `efficientnet-b4`, `mobilenet_v2`, `densenet121`, `vgg16`, `dpn68`, `resnext50_32x4d`, `se_resnet50`, `timm-efficientnet-b3`, `mit_b2`
- Histology-pretrained (ResNet-50, downloaded from HuggingFace via timm on first use, ~100MB each):
  - `resnet50_lunit-swav` -- Lunit SwAV, 19M TCGA patches (non-commercial)
  - `resnet50_lunit-bt` -- Lunit Barlow Twins, 19M TCGA patches (non-commercial)
  - `resnet50_kather100k` -- Kather100K colorectal tissue (CC-BY-4.0)
  - `resnet50_tcga-brca` -- TCGA-BRCA breast cancer SimCLR (GPLv3)

**Normalization Strategies**:
- `min_max` - Scale to [0, 1] using min/max values
- `percentile_99` - Clip at 99th percentile, then normalize
- `z_score` - Standard normalization (x - mean) / std
- `fixed_range` - User-specified min/max values

**Learning Rate Schedulers** (`scheduler`):
- `none` - Constant learning rate
- `cosine` - Cosine annealing
- `step` - Step decay
- `one_cycle` - One-cycle policy

## Inference Request Format

### POST /api/v1/inference (Classification Averages)

Returns per-tile class probability averages:

```json
{
  "model_path": "/path/to/model_dir",
  "tiles": [
    {"id": "tile_0", "data": "/path/to/tile_0.tiff"},
    {"id": "tile_1", "data": "/path/to/tile_1.tiff"}
  ],
  "input_config": {
    "num_channels": 3,
    "normalization": {"strategy": "percentile_99"}
  }
}
```

### POST /api/v1/inference/pixel (Pixel-Level Maps)

Returns file paths to raw probability maps:

```json
{
  "model_path": "/path/to/model_dir",
  "tiles": [
    {"id": "tile_0", "data": "/path/to/tile_0.tiff", "x": 0, "y": 0},
    {"id": "tile_1", "data": "/path/to/tile_1.tiff", "x": 448, "y": 0}
  ],
  "input_config": {
    "num_channels": 3,
    "normalization": {"strategy": "percentile_99"}
  },
  "output_dir": "/tmp/inference_output"
}
```

Response:
```json
{
  "output_paths": {
    "tile_0": "/tmp/inference_output/tile_0.bin",
    "tile_1": "/tmp/inference_output/tile_1.bin"
  }
}
```

**Output file format**: Raw float32 binary, CHW order (channels, height, width)

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Server health check |
| GET | `/api/v1/gpu` | GPU availability and info |
| POST | `/api/v1/gpu/clear` | Force-clear all GPU memory |
| GET | `/api/v1/models` | List available models |
| POST | `/api/v1/train` | Start training job |
| GET | `/api/v1/train/{job_id}/status` | Training progress |
| POST | `/api/v1/inference` | Run inference (class averages) |
| POST | `/api/v1/inference/pixel` | Run pixel inference (probability maps) |
| GET | `/api/v1/pretrained/encoders` | List pretrained encoders |
| GET | `/api/v1/pretrained/architectures` | List architectures |
| POST | `/api/v1/pretrained/layers` | Inspect model layers |
| GET | `/api/v1/pretrained/freeze-recommendations/{size}?encoder=...` | Layer freeze recommendations (encoder-aware) |
| GET | `/api/v1/pretrained/encoder/{name}` | Encoder details |
| GET | `/api/v1/pretrained/architecture/{name}` | Architecture details |

## GPU Support

### Device Priority

1. **CUDA** (NVIDIA GPUs) - Highest priority
2. **MPS** (Apple Silicon) - Second priority
3. **CPU** - Fallback

### Memory Management

The server includes automatic GPU memory management:

- **Cache clearing**: Automatically clears GPU cache between epochs
- **Memory monitoring**: Logs memory usage during training (CUDA)
- **Memory estimation**: Reports model memory requirements
- **Training cleanup**: `try/finally` guarantees GPU memory is freed after training completes, fails, or is cancelled
- **Force clear**: `POST /api/v1/gpu/clear` cancels running jobs, clears model caches, and frees all GPU memory

### Checking GPU Status

```bash
curl http://localhost:8765/api/v1/gpu
```

### Force-Clearing GPU Memory

If GPU memory is not released after a crash or failed training, force-clear it:

```bash
curl -X POST http://localhost:8765/api/v1/gpu/clear
```

This cancels running training jobs, clears cached models, runs garbage collection, and frees GPU memory. Returns before/after memory stats. Also available via the "Free GPU Memory" button in QuPath (Extensions > DL Pixel Classifier > Utilities).

Example response (CUDA):
```json
{
  "available": true,
  "device_type": "cuda",
  "device_string": "cuda:0",
  "name": "NVIDIA GeForce RTX 3080",
  "cuda_version": "12.1",
  "compute_capability": [8, 6],
  "allocated_mb": 0.0,
  "reserved_mb": 0.0,
  "total_mb": 10240.0
}
```

Example response (Apple Silicon):
```json
{
  "available": true,
  "device_type": "mps",
  "device_string": "mps",
  "name": "Apple Silicon (MPS)",
  "mps_available": true
}
```

## Model Output Structure

After training, models are saved with:

```
model_dir/
  model.pt              # PyTorch state dict
  model.onnx            # ONNX export (auto-generated)
  metadata.json         # Model metadata
```

### metadata.json Format

```json
{
  "id": "unet_20260123_141500",
  "name": "UNET Classifier",
  "architecture": {
    "type": "unet",
    "backbone": "resnet34",
    "use_pretrained": true
  },
  "input_config": {
    "num_channels": 3,
    "normalization": {
      "strategy": "percentile_99",
      "per_channel": true,
      "clip_percentile": 99.0
    }
  },
  "classes": [
    {"index": 0, "name": "Background"},
    {"index": 1, "name": "Foreground"}
  ]
}
```

## Development

### Project Structure

```
python_server/
  dlclassifier_server/
    main.py                    # FastAPI application
    routers/
      health.py                # /health, /gpu, /gpu/clear endpoints
      training.py              # /train endpoints
      inference.py             # /inference endpoints
      pretrained.py            # /pretrained/* endpoints
    services/
      training_service.py      # PyTorch training loop
      inference_service.py     # ONNX + PyTorch inference
      pretrained_models.py     # Encoder/architecture catalog
      job_manager.py           # Async job tracking
      gpu_manager.py           # CUDA/MPS/CPU detection
  tests/
    conftest.py                # Shared fixtures
    generate_test_data.py      # Synthetic data generator
    test_gpu_manager.py        # GPU manager tests
    test_training_service.py   # Training service tests
    test_inference_service.py  # Inference service tests
    test_api_endpoints.py      # API endpoint tests
```

### Running Development Server

For development with auto-reload on code changes:

```bash
uvicorn dlclassifier_server.main:app --reload --host 0.0.0.0 --port 8765
```

### API Documentation

When the server is running, visit:
- **Swagger UI**: http://localhost:8765/docs (interactive testing)
- **ReDoc**: http://localhost:8765/redoc (documentation)

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `externally-managed-environment` error | Use a virtual environment (see Installation Step 2) |
| `ModuleNotFoundError: No module named 'torch'` | Install PyTorch first (see PyTorch with Specific CUDA Versions) |
| `python: command not found` | Use `python3` on Linux/macOS |
| CUDA out of memory | Reduce batch size in training params, or use a smaller encoder (e.g., `mobilenet_v2`) |
| GPU memory not freed after crash | Use `POST /api/v1/gpu/clear` or "Free GPU Memory" button in QuPath Utilities menu |
| MPS not detected on Mac | Requires macOS 12.3+ and PyTorch 2.0+ |
| Server won't start on port 8765 | Check if another process is using the port, or specify a different port with `--port` |

### Checking Your Environment

**Verify Python version:**

```bash
python3 --version
```

**Verify PyTorch and GPU:**

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available()}')"
```

**Windows version:**

```cmd
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## License

Apache License 2.0
