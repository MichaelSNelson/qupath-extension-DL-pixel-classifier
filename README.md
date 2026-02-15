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
- **Histology-pretrained encoders** from TCGA/Lunit/Kather100K for better tissue feature extraction
- **Pluggable architecture** supporting UNet, SegFormer, and custom ONNX models
- **REST API communication** with Python deep learning server
- **Groovy scripting API** for batch processing
- **Headless builder API** for running workflows without GUI
- **"Copy as Script" buttons** in dialogs for reproducible workflows
- **Hierarchical geometry union** for efficient ROI merging

## Requirements

### QuPath Extension
- QuPath 0.6.0 or later
- Java 21+

### Python Server
- Python 3.10+
- PyTorch 2.1+
- CUDA-capable GPU (recommended)

## Installation

### QuPath Extension

1. Build the extension:
   ```bash
   cd qupath-extension-DL-pixel-classifier
   ./gradlew build
   ```

2. Copy the JAR from `build/libs/` to your QuPath extensions directory.

### Python Server

1. Install dependencies:
   ```bash
   cd python_server
   pip install -e .
   ```

2. Start the server:
   ```bash
   dlclassifier-server
   ```

For detailed Python server documentation, see [python_server/README.md](python_server/README.md).

## Testing

### Python Server Tests

The Python server includes a comprehensive test suite covering GPU management, training, inference, and API endpoints.

```bash
cd python_server

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install with test dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=dlclassifier_server --cov-report=html
```

Current status: **78 tests passing, 5 skipped**

### Generating Test Data

To generate synthetic training data for testing:

```bash
cd python_server
python tests/generate_test_data.py
```

This creates 6 synthetic images with masks for training validation.

## Usage

### Training a Classifier

1. Open an image in QuPath
2. Create annotations with classification (e.g., "Foreground", "Background")
3. Go to **Extensions > DL Pixel Classifier > Train Classifier...**
4. Configure training parameters (model, epochs, tile size, etc.)
5. Select training data source:
   - **Current image only** (default) - train from annotations on the open image
   - **Selected project images** - train from annotations across multiple project images
6. Click **Start Training**
7. Monitor progress with separate train/val loss charting

### Applying a Classifier

1. Open an image in QuPath
2. Go to **Extensions > DL Pixel Classifier > Apply Classifier...**
3. Select a trained classifier
4. Choose output type (measurements, objects, or overlay)
5. Click **Apply**

### Scripting

**Simple API** (convenience methods):

```groovy
// Load classifier
def classifier = DLClassifierScripts.loadClassifier("my_classifier_id")

// Apply to annotations
def annotations = getAnnotationObjects()
DLClassifierScripts.classifyRegions(classifier, annotations)

// Or with specific output type
DLClassifierScripts.classifyRegions(classifier, annotations, "objects")

// Batch process project
for (entry in getProject().getImageList()) {
    def imageData = entry.readImageData()
    DLClassifierScripts.classifyRegions(classifier, imageData.getAnnotationObjects())
    entry.saveImageData(imageData)
}
```

**Builder API** (full control, headless-compatible):

```groovy
import qupath.ext.dlclassifier.controller.InferenceWorkflow
import qupath.ext.dlclassifier.model.*
import qupath.ext.dlclassifier.scripting.DLClassifierScripts

def classifier = DLClassifierScripts.loadClassifier("my_classifier_id")

def inferenceConfig = InferenceConfig.builder()
        .tileSize(512)
        .overlap(64)
        .blendMode(InferenceConfig.BlendMode.LINEAR)
        .outputType(InferenceConfig.OutputType.MEASUREMENTS)
        .useGPU(true)
        .build()

def channelConfig = ChannelConfiguration.builder()
        .selectedChannels([0, 1, 2])
        .channelNames(["Red", "Green", "Blue"])
        .bitDepth(8)
        .normalizationStrategy(ChannelConfiguration.NormalizationStrategy.PERCENTILE_99)
        .build()

def result = InferenceWorkflow.builder()
        .classifier(classifier)
        .config(inferenceConfig)
        .channels(channelConfig)
        .annotations(getAnnotationObjects())
        .build()
        .run()

println "Processed ${result.processedAnnotations()} annotations, ${result.processedTiles()} tiles"
```

Training follows the same pattern via `TrainingWorkflow.builder()`.

### Copy as Script

Both the training and inference dialogs include a **"Copy as Script"** button that generates
a runnable Groovy script from the current dialog settings and copies it to the clipboard.
This enables reproducible workflows -- configure in the GUI, then paste into QuPath's
Script Editor for batch use.

## Architecture

```
qupath-extension-DL-pixel-classifier/
├── src/main/java/qupath/ext/dlclassifier/
│   ├── SetupDLClassifier.java       # Extension entry point
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
│   │   ├── ClassifierClient.java    # HTTP client
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
│   ├── scripting/
│   │   ├── DLClassifierScripts.java  # Groovy API
│   │   └── ScriptGenerator.java      # Dialog-to-script generation
│   └── preferences/
│       └── DLClassifierPreferences.java
│
├── python_server/                    # Python DL server
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
