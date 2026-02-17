# QuPath Deep Learning Pixel Classifier Extension

A QuPath extension for deep learning-based pixel classification, supporting both brightfield (RGB) and multi-channel fluorescence/spectral images.

## Features

- **Train custom classifiers** from annotated regions using sparse annotations
- **Multi-image training** from multiple project images in a single training run
- **Multi-channel support** with per-channel normalization
- **Real-time progress** with separate train/val loss charting
- **Multiple output types**: Measurements, detection objects, or classification overlays
- **Pixel-level inference** for OBJECTS and OVERLAY output types with full per-pixel probability maps
- **Dialog preference persistence** -- training and inference settings are remembered across sessions
- **Combined CE + Dice loss** for improved segmentation quality (default)
- **IoU-based early stopping** monitors mean IoU instead of validation loss
- **Mixed precision training** (AMP) for ~2x speedup on CUDA GPUs
- **Configurable training strategy** via collapsed "Training Strategy" section in the training dialog (scheduler, loss function, early stopping metric/patience, mixed precision)
- **Histology-pretrained encoders** from TCGA/Lunit/Kather100K for better tissue feature extraction
- **Pluggable architecture** supporting UNet, SegFormer, and custom ONNX models
- **REST API communication** with Python deep learning server
- **Groovy scripting API** for batch processing
- **Headless builder API** for running workflows without GUI
- **"Copy as Script" buttons** in dialogs for reproducible workflows
- **Hierarchical geometry union** for efficient ROI merging

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/INSTALLATION.md) | Full setup: Java extension + Python server + GPU configuration |
| [Training Guide](docs/TRAINING_GUIDE.md) | Step-by-step training workflow how-to |
| [Inference Guide](docs/INFERENCE_GUIDE.md) | Step-by-step inference workflow how-to |
| [Parameters](docs/PARAMETERS.md) | Every parameter with defaults, ranges, and ML guidance |
| [Scripting](docs/SCRIPTING.md) | Groovy API, builder pattern, batch processing, Copy as Script |
| [Best Practices](docs/BEST_PRACTICES.md) | Backbone selection, annotation strategy, hyperparameter tuning |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Server issues, training issues, GPU issues, diagnostics |
| [Preferences](docs/PREFERENCES.md) | All persistent preferences with defaults and keys |
| [Quickstart](QUICKSTART.md) | Zero-to-classifier in 10 minutes |
| [Python Server](python_server/README.md) | Detailed Python server documentation |

## Requirements

- QuPath 0.6.0+, Java 21+
- CUDA-capable GPU recommended (CPU and Apple Silicon MPS also supported)
- Internet connection for first-time environment setup (~2-4 GB download)

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for full setup instructions.

## Quick Start

```bash
# Build Java extension
./gradlew build
# Copy JAR from build/libs/ to QuPath extensions directory
```

Then in QuPath:

1. **Extensions > DL Pixel Classifier > Setup DL Environment...** (first time only)
2. Click **Begin Setup** to download the Python environment (~2-4 GB)
3. Once complete, **Train Classifier...** and other menu items appear automatically

The extension uses [Appose](https://github.com/apposed/appose) to manage an embedded Python environment with PyTorch -- no separate Python installation or server management required.

See [QUICKSTART.md](QUICKSTART.md) for the complete walkthrough.

## Usage Overview

**Training**: Create annotations with 2+ classes, open the training dialog, configure parameters, and click Start Training. See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

**Inference**: Select a trained classifier, choose output type (measurements/objects/overlay), and click Apply. See [docs/INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md).

**Scripting**: Use the Simple API or Builder API for batch processing. Both dialogs include a "Copy as Script" button. See [docs/SCRIPTING.md](docs/SCRIPTING.md).

## Testing

```bash
cd python_server
pip install -e ".[dev]"
pytest tests/ -v
```

Current status: **78 tests passing, 5 skipped**

## Backend Modes

The extension supports two backend modes for running the Python deep learning engine:

### Appose (Default -- Embedded Python)

The default mode uses [Appose](https://github.com/apposed/appose) to manage an embedded Python environment with PyTorch, eliminating the need for a separate Python installation or server process.

- **First-time setup**: A guided setup wizard (**Extensions > DL Pixel Classifier > Setup DL Environment...**) downloads and configures the environment (~2-4 GB). ONNX export support can be optionally excluded to reduce download size.
- **Environment location**: `~/.appose/pixi/dl-pixel-classifier/`
- **Recovery**: Use **Utilities > Rebuild DL Environment...** to delete and re-download the environment if it becomes corrupted.
- Communication uses Appose's shared-memory IPC (no network sockets).

### HTTP (External Server)

For advanced setups (e.g., remote GPU workstations), disable Appose in **Edit > Preferences > DL Pixel Classifier** and run the Python server separately:

```bash
cd python_server && pip install -e ".[cuda]"
dlclassifier-server
```

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for full HTTP server setup instructions.

## Architecture

```
qupath-extension-DL-pixel-classifier/
├── src/main/java/qupath/ext/dlclassifier/
│   ├── SetupDLClassifier.java       # Extension entry point & menu management
│   ├── classifier/                   # Classifier type system
│   │   ├── ClassifierHandler.java
│   │   ├── ClassifierRegistry.java
│   │   └── handlers/
│   │       └── UNetHandler.java
│   ├── controller/                   # Workflow orchestration
│   │   ├── DLClassifierController.java
│   │   ├── TrainingWorkflow.java
│   │   └── InferenceWorkflow.java
│   ├── service/                      # Backend services
│   │   ├── ApposeService.java       # Appose embedded Python management
│   │   ├── ClassifierClient.java    # HTTP client (external server mode)
│   │   ├── BackendFactory.java      # Backend selection (Appose vs HTTP)
│   │   ├── ModelManager.java
│   │   └── OverlayService.java
│   ├── model/                        # Data objects
│   │   ├── TrainingConfig.java
│   │   ├── InferenceConfig.java
│   │   ├── ChannelConfiguration.java
│   │   └── ClassifierMetadata.java
│   ├── utilities/                    # Processing utilities
│   │   ├── AnnotationExtractor.java # Training data export (single + multi-image)
│   │   ├── TileProcessor.java
│   │   ├── ChannelNormalizer.java
│   │   └── OutputGenerator.java
│   ├── ui/                           # UI components
│   │   ├── SetupEnvironmentDialog.java  # First-time setup wizard
│   │   └── TooltipHelper.java
│   ├── scripting/
│   │   ├── DLClassifierScripts.java  # Groovy API
│   │   └── ScriptGenerator.java      # Dialog-to-script generation
│   └── preferences/
│       └── DLClassifierPreferences.java
│
├── python_server/                    # Python DL server (HTTP mode)
│   └── dlclassifier_server/
│       ├── main.py                   # FastAPI application
│       ├── routers/                  # API endpoints
│       └── services/                 # Training/inference services
```

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/health | Server health check |
| GET | /api/v1/gpu | GPU availability info |
| GET | /api/v1/models | List available models |
| POST | /api/v1/train | Start training job |
| GET | /api/v1/train/{job_id}/status | Training progress (train_loss, val_loss, accuracy) |
| POST | /api/v1/inference | Run tile-level aggregated inference (for MEASUREMENTS) |
| POST | /api/v1/inference/pixel | Run pixel-level inference returning probability maps (for OBJECTS/OVERLAY) |

## Supported Image Types

| Image Type | Channels | Bit Depth | Strategy |
|------------|----------|-----------|----------|
| Brightfield RGB | 3 | 8-bit | Direct input |
| Immunofluorescence | 2-8+ | 8/12/16-bit | Channel selection + normalization |
| Spectral/Hyperspectral | 10-100+ | 16-bit | Channel grouping |

## License

Apache License 2.0

## Contributing

Contributions welcome! Please open issues or pull requests on GitHub.
