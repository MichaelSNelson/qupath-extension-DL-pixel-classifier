# QuPath
# Deep Learning
# Pixel Classifier
# Extension

A QuPath extension for deep learning-based pixel classification, supporting both brightfield (RGB) and multi-channel fluorescence/spectral images.

## Features

- **Train custom classifiers** from annotated regions using sparse annotations
- **Multi-image training** from multiple project images in a single training run
- **Multi-channel support** with per-channel normalization
- **Real-time progress** with separate train/val loss charting
- **Multiple output types**: Measurements, detection objects, or classification overlays
- **Pixel-level inference** for OBJECTS and OVERLAY output types with full per-pixel probability maps
- **Image-level normalization** eliminates tile boundary artifacts by computing consistent statistics across the entire image
- **Dialog preference persistence** -- training and inference settings are remembered across sessions
- **Combined CE + Dice loss** for improved segmentation quality (default)
- **IoU-based early stopping** monitors mean IoU instead of validation loss
- **Mixed precision training** (AMP) for ~2x speedup on CUDA GPUs
- **Configurable training strategy** via collapsed "Training Strategy" section in the training dialog (scheduler, loss function, early stopping metric/patience, mixed precision)
- **Histology-pretrained encoders** from TCGA/Lunit/Kather100K for better tissue feature extraction
- **Pluggable architecture** supporting UNet and custom ONNX models
- **Appose shared-memory IPC** for embedded Python inference (REST API available for external server mode)
- **Groovy scripting API** for batch processing
- **Headless builder API** for running workflows without GUI
- **"Copy as Script" buttons** in dialogs for reproducible workflows
- **Hierarchical geometry union** for efficient ROI merging

## Installation

1. **Download** the latest JAR from the [GitHub Releases](https://github.com/MichaelSNelson/qupath-extension-DL-pixel-classifier/releases) page
2. **Copy** the JAR to your QuPath extensions directory:

| OS | Extensions path |
|----|----------------|
| Windows | `C:\Users\<you>\AppData\Local\QuPath\v0.6\extensions\` |
| macOS | `~/Library/Application Support/QuPath/v0.6/extensions/` |
| Linux | `~/.local/share/QuPath/v0.6/extensions/` |

3. **Restart QuPath** -- the extension appears under **Extensions > DL Pixel Classifier**

> **Tip:** In QuPath, **Edit > Preferences > Extensions** shows the extensions directory path.

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed instructions and GPU configuration.

## Getting Started

1. **Set up the Python environment** -- **Extensions > DL Pixel Classifier > Setup DL Environment...** downloads and configures everything automatically (~2-4 GB, first time only)
2. **Train a classifier** -- create annotations, open **Train Classifier...**, and click Start Training
3. **Apply the classifier** -- open **Apply Classifier...**, select a model, choose an output type, and click Apply

See [QUICKSTART.md](QUICKSTART.md) for a complete walkthrough (zero to classifier in ~10 minutes).

## Requirements

- QuPath 0.6.0+
- NVIDIA GPU with CUDA recommended (CPU and Apple Silicon MPS also supported)
- Internet connection for first-time environment setup (~2-4 GB download)

> **Note:** A separate Python installation is **not** required. The extension manages its own embedded Python environment via [Appose](https://github.com/apposed/appose).

## Documentation

| Guide | Description |
|-------|-------------|
| [Quickstart](QUICKSTART.md) | Zero-to-classifier in 10 minutes |
| [Installation](docs/INSTALLATION.md) | Full setup: extension install + Python environment + GPU configuration |
| [Training Guide](docs/TRAINING_GUIDE.md) | Step-by-step training workflow how-to |
| [Inference Guide](docs/INFERENCE_GUIDE.md) | Step-by-step inference workflow how-to |
| [Parameters](docs/PARAMETERS.md) | Every parameter with defaults, ranges, and ML guidance |
| [Scripting](docs/SCRIPTING.md) | Groovy API, builder pattern, batch processing, Copy as Script |
| [Best Practices](docs/BEST_PRACTICES.md) | Backbone selection, annotation strategy, hyperparameter tuning |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Environment issues, GPU issues, training/inference issues, diagnostics |
| [Preferences](docs/PREFERENCES.md) | All persistent preferences with defaults and keys |
| [Python Server](python_server/README.md) | Detailed Python server documentation (HTTP mode) |

## GPU Support

The extension automatically detects and uses available GPU hardware:

- **NVIDIA GPUs (CUDA)** -- auto-detected on Windows and Linux. Requires NVIDIA drivers to be installed.
- **Apple Silicon (MPS)** -- auto-detected on macOS with M-series chips.
- **CPU fallback** -- automatic when no GPU is available. Training will be slower but functional.

The setup wizard reports which GPU backend was detected at completion. See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) if GPU is not detected.

## Backend Modes

The extension supports two backend modes for running the Python deep learning engine:

### Appose (Default -- Embedded Python)

The default mode uses [Appose](https://github.com/apposed/appose) to manage an embedded Python environment with PyTorch, eliminating the need for a separate Python installation or server process.

- **First-time setup**: A guided setup wizard (**Extensions > DL Pixel Classifier > Setup DL Environment...**) downloads and configures the environment (~2-4 GB). ONNX export support can be optionally excluded to reduce download size.
- **Environment location**: `~/.local/share/appose/dl-pixel-classifier/`
- **Recovery**: Use **Utilities > Rebuild DL Environment...** to delete and re-download the environment if it becomes corrupted.
- Communication uses Appose's shared-memory IPC (no network sockets).

### HTTP (External Server)

For advanced setups (e.g., remote GPU workstations), disable Appose in **Edit > Preferences > DL Pixel Classifier** and run the Python server separately. See [docs/INSTALLATION.md](docs/INSTALLATION.md) for full HTTP server setup instructions.

## Supported Image Types

| Image Type | Channels | Bit Depth | Strategy |
|------------|----------|-----------|----------|
| Brightfield RGB | 3 | 8-bit | Direct input |
| Immunofluorescence | 2-8+ | 8/12/16-bit | Channel selection + normalization |
| Spectral/Hyperspectral | 10-100+ | 16-bit | Channel grouping |

## Architecture

```
qupath-extension-DL-pixel-classifier/
├── src/main/java/qupath/ext/dlclassifier/
│   ├── SetupDLClassifier.java        # Extension entry point & menu management
│   ├── DLClassifierChecks.java       # Startup validation
│   ├── classifier/                    # Classifier type system
│   │   ├── ClassifierHandler.java
│   │   ├── ClassifierRegistry.java
│   │   └── handlers/
│   │       ├── UNetHandler.java
│   │       └── CustomONNXHandler.java
│   ├── controller/                    # Workflow orchestration
│   │   ├── DLClassifierController.java
│   │   ├── TrainingWorkflow.java
│   │   ├── InferenceWorkflow.java
│   │   └── ModelManagementWorkflow.java
│   ├── service/                       # Backend services
│   │   ├── ApposeService.java        # Appose embedded Python management
│   │   ├── ApposeClassifierBackend.java  # Appose backend implementation
│   │   ├── HttpClassifierBackend.java    # HTTP backend implementation
│   │   ├── ClassifierBackend.java    # Backend interface
│   │   ├── ClassifierClient.java     # HTTP client (external server mode)
│   │   ├── BackendFactory.java       # Backend selection (Appose vs HTTP)
│   │   ├── DLPixelClassifier.java    # QuPath PixelClassifier integration
│   │   ├── ModelManager.java
│   │   └── OverlayService.java
│   ├── model/                         # Data objects
│   │   ├── TrainingConfig.java
│   │   ├── InferenceConfig.java
│   │   ├── ChannelConfiguration.java
│   │   └── ClassifierMetadata.java
│   ├── utilities/                     # Processing utilities
│   │   ├── AnnotationExtractor.java  # Training data export (single + multi-image)
│   │   ├── TileProcessor.java
│   │   ├── ChannelNormalizer.java
│   │   ├── BitDepthConverter.java
│   │   └── OutputGenerator.java
│   ├── ui/                            # UI components
│   │   ├── TrainingDialog.java
│   │   ├── InferenceDialog.java
│   │   ├── SetupEnvironmentDialog.java   # First-time setup wizard
│   │   ├── ChannelSelectionPanel.java
│   │   ├── LayerFreezePanel.java
│   │   ├── ProgressMonitorController.java
│   │   ├── PythonConsoleWindow.java
│   │   └── TooltipHelper.java
│   ├── scripting/
│   │   ├── DLClassifierScripts.java   # Groovy API
│   │   └── ScriptGenerator.java       # Dialog-to-script generation
│   └── preferences/
│       └── DLClassifierPreferences.java
│
├── python_server/                     # Python DL server (HTTP mode)
│   └── dlclassifier_server/
│       ├── main.py                    # FastAPI application
│       ├── routers/                   # API endpoints
│       ├── services/                  # Training/inference services
│       └── utils/                     # Shared utilities (normalization, etc.)
```

## REST API (HTTP Mode Only)

> These endpoints apply when using the external Python server (HTTP mode). In the default Appose mode, communication uses shared-memory IPC and no HTTP server is involved.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/health | Server health check |
| GET | /api/v1/gpu | GPU availability info |
| GET | /api/v1/models | List available models |
| POST | /api/v1/train | Start training job |
| GET | /api/v1/train/{job_id}/status | Training progress (train_loss, val_loss, accuracy) |
| POST | /api/v1/inference | Run tile-level aggregated inference (for MEASUREMENTS) |
| POST | /api/v1/inference/pixel | Run pixel-level inference returning probability maps (for OBJECTS/OVERLAY) |

## For Developers

### Building from source

```bash
git clone https://github.com/MichaelSNelson/qupath-extension-DL-pixel-classifier.git
cd qupath-extension-DL-pixel-classifier
./gradlew build
```

This produces a JAR file in `build/libs/`. Copy it to your QuPath extensions directory and restart QuPath.

For a shadow JAR that bundles all dependencies:

```bash
./gradlew shadowJar
```

### Running Python tests

```bash
cd python_server
pip install -e ".[dev]"
pytest tests/ -v
```

Current status: **78 tests passing, 5 skipped**

## License

Apache License 2.0

## Contributing

Contributions welcome! Please open issues or pull requests on GitHub.
