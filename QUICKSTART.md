# Quickstart Guide

Get from zero to your first trained pixel classifier in about 10 minutes.

This guide covers both halves of the system:
- **Java extension** (runs inside QuPath)
- **Python server** (runs the deep learning backend)

---

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| QuPath | 0.6.0 or later |
| Java JDK | 21+ (for building the extension) |
| Python | 3.10+ |
| GPU | NVIDIA GPU with CUDA recommended; Apple Silicon (MPS) also works; CPU fallback available |

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

## Step 3: Set Up the Python Server

### 3a. Create a virtual environment

**Windows (Command Prompt):**
```cmd
cd python_server
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
cd python_server
python -m venv venv
venv\Scripts\Activate.ps1
```

> If PowerShell blocks activation, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**macOS / Linux:**
```bash
cd python_server
python3 -m venv venv
source venv/bin/activate
```

### 3b. Install dependencies

**With NVIDIA GPU (recommended):**
```bash
pip install -e ".[cuda]"
```

**CPU only or Apple Silicon:**
```bash
pip install -e .
```

> **Specific CUDA version?** Install PyTorch first, then the server package:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install -e .
> ```

### 3c. Start the server

```bash
dlclassifier-server
```

You should see output like:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8765
```

### 3d. Verify it works

Open a second terminal and run:

```bash
curl http://localhost:8765/api/v1/health
```

Expected response:
```json
{"status": "healthy"}
```

Check GPU detection:
```bash
curl http://localhost:8765/api/v1/gpu
```

> **Windows without curl:** Open `http://localhost:8765/docs` in a browser for the interactive Swagger UI.

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
| **Backbone** | `resnet34` |
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
|       |-- service/          # ClassifierClient (HTTP), ModelManager
|       |-- ui/               # Dialogs and progress UI
|       |-- utilities/        # AnnotationExtractor, TileProcessor
|       +-- scripting/        # Groovy API, script generation
|
|-- python_server/            # Python backend
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

| Preference | Default | Where to change |
|-----------|---------|-----------------|
| Server host | `localhost` | Extensions > DL Pixel Classifier > Utilities > Server Settings |
| Server port | `8765` | Extensions > DL Pixel Classifier > Utilities > Server Settings |
| Default tile size | `512` | Training dialog |
| Default epochs | `50` | Training dialog |
| Default batch size | `8` | Training dialog |

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

### Server won't start

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'torch'` | Activate your venv first, then `pip install -e .` |
| Port 8765 already in use | Kill the other process, or start with `dlclassifier-server --port 8766` and update QuPath's Server Settings |
| `externally-managed-environment` | Use a virtual environment (Step 3a) |

### QuPath can't connect to server

1. Verify the server is running (`curl http://localhost:8765/api/v1/health`)
2. Check Server Settings in QuPath match the server's host and port
3. If the server is on another machine, check firewall rules

### Training fails immediately

- Make sure you have annotations with assigned classes
- The selected classes in the dialog must match annotation classes in the image
- Check the QuPath log (**View > Show log**) for detailed error messages

### Out of memory

- Reduce **batch size** (try 4 or 2)
- Use a smaller backbone (`mobilenet_v2` instead of `resnet50`)
- Reduce **tile size** (256 instead of 512)

### Verify your Python environment

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Next Steps

- **Increase epochs** once you've verified the pipeline works (50-100 for real training)
- **Try transfer learning** -- freeze early encoder layers for faster convergence on small datasets
- **Experiment with architectures** -- UNet++ and DeepLabV3+ often outperform vanilla UNet
- **Multi-image training** -- combine annotations from several images for a more robust classifier
- See the full [README](README.md) and [Python server docs](python_server/README.md) for the complete API reference
