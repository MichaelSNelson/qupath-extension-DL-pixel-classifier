# Quickstart Guide

Get from zero to your first trained pixel classifier in about 10 minutes.

---

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| QuPath | 0.6.0 or later |
| Java JDK | 21+ (for building the extension) |
| GPU | NVIDIA GPU with CUDA recommended; Apple Silicon (MPS) also works; CPU fallback available |
| Internet | Required for first-time environment setup (~2-4 GB download) |

> **Note:** A separate Python installation is **not** required. The extension manages its own embedded Python environment via [Appose](https://github.com/apposed/appose).

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/MichaelSNelson/qupath-extension-DL-pixel-classifier.git
cd qupath-extension-DL-pixel-classifier
```

---

## Step 2: Build the Java Extension

```bash
./gradlew build
```

This produces a JAR file in `build/libs/`. Copy it to your QuPath extensions directory:

| OS | Typical extensions path |
|----|------------------------|
| Windows | `C:\Users\<you>\AppData\Local\QuPath\v0.6\extensions\` |
| macOS | `~/Library/Application Support/QuPath/v0.6/extensions/` |
| Linux | `~/.local/share/QuPath/v0.6/extensions/` |

Alternatively, in QuPath: **Edit > Preferences > Extensions** shows the extensions directory path. Drop the JAR there and restart QuPath.

---

## Step 3: Set Up the Python Environment

On first launch after installing the extension, only **Setup DL Environment...** will be visible in the menu.

1. Go to **Extensions > DL Pixel Classifier > Setup DL Environment...**
2. Review the download size warning (~2-4 GB)
3. Optionally uncheck **ONNX export support** (~200 MB savings) if you don't need it
4. Click **Begin Setup**
5. Wait for the download and configuration to complete (the dialog shows progress)
6. Click **Close** when done

The training and inference menu items now appear automatically. On subsequent launches, the environment is detected on disk and everything is ready immediately.

> **Alternative: External Python Server**
>
> If you prefer to run the Python backend on a separate machine (e.g., a GPU workstation), disable Appose in **Edit > Preferences > DL Pixel Classifier** and set up the server manually. See [docs/INSTALLATION.md](docs/INSTALLATION.md) for details.

---

## Step 4: Train Your First Classifier

### 4a. Prepare annotations in QuPath

1. Open an image in QuPath
2. Create at least two annotation classes (e.g., right-click in the annotation class list to add "Foreground" and "Background")
3. Draw annotations on the image using the brush or polygon tools
4. Assign each annotation to a class (right-click the annotation > Set class)

> **Minimum requirement:** At least one annotation per class. More annotations = better results. Line/brush annotations work well -- you don't need to label every pixel.

### 4b. Open the training dialog

**Extensions > DL Pixel Classifier > Train Classifier...**

### 4c. Configure training

The dialog has collapsible sections. For a quick first test:

| Setting | Recommended first-run value |
|---------|-----------------------------|
| **Classifier Name** | `test_classifier_v1` |
| **Training Data Source** | Current image only (default) |
| **Architecture** | `unet` |
| **Backbone** | `resnet34` (or a histology backbone -- see below) |
| **Epochs** | `3` (just to verify it works) |
| **Tile Size** | `256` or `512` |
| **Use Pretrained Weights** | checked |

Leave everything else at defaults.

### 4d. Start training

Click **Start Training**. A progress window appears showing:
- Export progress (extracting patches from annotations)
- Training progress with epoch-by-epoch train loss and validation loss
- A live loss chart

A 3-epoch test run should complete in under a minute on GPU.

### 4e. Verify the result

When training completes, the classifier is saved to your QuPath project under `classifiers/`. You can see it via **Extensions > DL Pixel Classifier > Manage Models...**

---

## Step 5: Apply the Classifier

1. Open an image (same one or a different one)
2. Create annotation(s) around the region(s) you want to classify
3. **Extensions > DL Pixel Classifier > Apply Classifier...**
4. Select your trained classifier
5. Choose an output type:
   - **Measurements** -- adds class probabilities as annotation measurements
   - **Objects** -- creates detection objects from the classification map
   - **Overlay** -- renders a color overlay on the viewer
6. Click **Apply**

---

## Multi-Image Training

To train from annotations across multiple project images:

1. Open a QuPath project with multiple annotated images
2. Open the training dialog (**Train Classifier...**)
3. Under **TRAINING DATA SOURCE**, select **Selected project images**
4. Check/uncheck images in the list (only images with classified annotations appear)
5. Train as usual -- patches from all selected images are combined into one training set

---

## Project Layout

```
qupath-extension-DL-pixel-classifier/
|
|-- src/                      # Java extension source
|   |-- main/java/qupath/ext/dlclassifier/
|       |-- controller/       # Workflow orchestration
|       |-- model/            # Data objects (TrainingConfig, etc.)
|       |-- service/          # ApposeService, ClassifierClient, ModelManager
|       |-- ui/               # Dialogs, setup wizard, progress UI
|       |-- utilities/        # AnnotationExtractor, TileProcessor
|       +-- scripting/        # Groovy API, script generation
|
|-- python_server/            # Python backend (HTTP mode only)
|   |-- dlclassifier_server/
|       |-- main.py           # FastAPI app
|       |-- routers/          # API endpoints
|       +-- services/         # Training, inference, job management
|
|-- scripts/examples/         # Groovy script examples
|-- build.gradle.kts          # Java build config
+-- QUICKSTART.md             # This file
```

---

## Configuration

The extension stores preferences via QuPath's preference system. Defaults:

**Backend & Server:**

| Preference | Default | Where to change |
|-----------|---------|-----------------|
| Use Appose (Embedded Python) | `true` | Edit > Preferences > DL Pixel Classifier |
| Server host | `localhost` | Extensions > DL Pixel Classifier > Utilities > Server Settings |
| Server port | `8765` | Extensions > DL Pixel Classifier > Utilities > Server Settings |
| Default tile size | `512` | Training / Inference dialog |
| Default epochs | `50` | Training dialog |
| Default batch size | `8` | Training dialog |

**Training Dialog (remembered across sessions):**

| Preference | Default | Description |
|-----------|---------|-------------|
| Architecture | `unet` | Last used model architecture |
| Backbone | `resnet34` | Last used encoder backbone |
| Validation split | `20%` | Percentage of data held for validation |
| Horizontal flip | `true` | Augmentation: random horizontal flip |
| Vertical flip | `true` | Augmentation: random vertical flip |
| Rotation | `true` | Augmentation: random 90-degree rotation |
| Color jitter | `false` | Augmentation: brightness/contrast variation |
| Elastic deformation | `false` | Augmentation: elastic distortion |

**Training Strategy (remembered across sessions):**

These are in the collapsed **"TRAINING STRATEGY"** section of the training dialog.

| Preference | Default | Description |
|-----------|---------|-------------|
| LR Scheduler | `One Cycle` | Learning rate schedule (One Cycle / Cosine Annealing / Step Decay / None) |
| Loss Function | `Cross Entropy + Dice` | Loss function (CE+Dice optimizes region overlap; CE alone over-weights easy pixels) |
| Early Stop Metric | `Mean IoU` | Metric for early stopping (Mean IoU / Validation Loss) |
| Early Stop Patience | `15` | Epochs to wait without improvement before stopping |
| Mixed Precision | `true` | Use FP16/FP32 automatic mixed precision on CUDA GPUs (~2x speedup) |

**Inference Dialog (remembered across sessions):**

| Preference | Default | Description |
|-----------|---------|-------------|
| Output type | `MEASUREMENTS` | How to output classification results |
| Blend mode | `LINEAR` | How overlapping tiles are merged |
| Smoothing | `1.0` | Boundary smoothing amount |
| Apply to selected | `true` | Apply to selected annotations only |
| Create backup | `false` | Back up measurements before overwriting |

> **Note:** Dialog settings are saved automatically when you click "Start Training" or "Apply". The next time you open the dialog, your previous settings are restored.

If the Python server runs on a different machine (e.g., a GPU workstation), change the host/port in Server Settings to point to that machine's IP and port.

---

## Scripting (Headless / Batch)

Both dialogs have a **Copy as Script** button that generates a runnable Groovy script from the current settings.

**Quick example -- apply a classifier to all project images:**

```groovy
import qupath.ext.dlclassifier.scripting.DLClassifierScripts

def classifier = DLClassifierScripts.loadClassifier("test_classifier_v1")

for (entry in getProject().getImageList()) {
    def imageData = entry.readImageData()
    DLClassifierScripts.classifyRegions(classifier, imageData.getAnnotationObjects())
    entry.saveImageData(imageData)
}
println "Done"
```

See `scripts/examples/` for more examples.

---

## Troubleshooting

### Environment setup fails or stalls

| Symptom | Fix |
|---------|-----|
| Setup dialog shows error | Check internet connection; try again with **Retry** |
| Download is very slow | The initial download is ~2-4 GB; expect several minutes on slower connections |
| Environment corrupted | Use **Utilities > Rebuild DL Environment...** to delete and re-download |
| Menu items don't appear after setup | Close and reopen QuPath; verify `~/.local/share/appose/dl-pixel-classifier/.pixi/` exists |

### Training fails immediately

- Make sure you have annotations with assigned classes
- The selected classes in the dialog must match annotation classes in the image
- Check the QuPath log (**View > Show log**) for detailed error messages

### Out of memory

- Reduce **batch size** (try 4 or 2)
- Use a smaller backbone (`mobilenet_v2` instead of `resnet50`)
- Reduce **tile size** (256 instead of 512)

### HTTP mode: Server won't start

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'torch'` | Activate your venv first, then `pip install -e .` |
| Port 8765 already in use | Kill the other process, or start with `dlclassifier-server --port 8766` and update QuPath's Server Settings |
| QuPath can't connect | Verify server is running, check Server Settings, check firewall rules |

---

## Next Steps

- **Increase epochs** once you've verified the pipeline works (50-100 for real training)
- **Try histology-pretrained backbones** -- select a backbone ending in "(Histology)" for weights pretrained on tissue patches instead of ImageNet. These produce better features for tissue classification and need less layer freezing. ~100MB download on first use (cached afterward).
- **Try transfer learning** -- freeze early encoder layers for faster convergence on small datasets
- **Experiment with backbones** -- try a larger backbone (resnet50) or a histology-pretrained backbone for tissue classification, or import a custom ONNX model
- **Multi-image training** -- combine annotations from several images for a more robust classifier
- **Tune training strategy** -- expand the "TRAINING STRATEGY" section in the training dialog to adjust the LR scheduler, loss function, early stopping metric/patience, and mixed precision
- See the full [README](README.md) and [Python server docs](python_server/README.md) for the complete API reference
