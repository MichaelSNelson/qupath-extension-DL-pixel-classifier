# Installation Guide

Complete setup instructions for the DL Pixel Classifier extension.

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| QuPath | 0.6.0 or later |
| GPU | NVIDIA GPU with CUDA recommended; Apple Silicon (MPS) also works; CPU fallback available |
| Internet | Required for first-time environment setup (~2-4 GB download) |

> **Note:** A separate Python installation is **not** required for the default Appose backend. The extension manages its own embedded Python environment.

## Part 1: Install the Extension

### Download the JAR

Download the latest release JAR from the [GitHub Releases](https://github.com/MichaelSNelson/qupath-extension-DL-pixel-classifier/releases) page.

### Copy to QuPath extensions directory

Copy the JAR to your QuPath extensions directory:

| OS | Typical extensions path |
|----|------------------------|
| Windows | `C:\Users\<you>\AppData\Local\QuPath\v0.6\extensions\` |
| macOS | `~/Library/Application Support/QuPath/v0.6/extensions/` |
| Linux | `~/.local/share/QuPath/v0.6/extensions/` |

> **Tip:** In QuPath, **Edit > Preferences > Extensions** shows the extensions directory path. You can drag and drop the JAR there.

### Verify installation

Restart QuPath. You should see **Extensions > DL Pixel Classifier** in the menu bar. On first launch, only **Setup DL Environment...** and **Utilities** will be visible -- this is normal.

## Part 2: Python Environment Setup (Appose -- Default)

The extension uses [Appose](https://github.com/apposed/appose) to automatically manage an embedded Python environment. No manual Python setup is needed.

### First-time setup

1. Open QuPath with the extension installed
2. Go to **Extensions > DL Pixel Classifier**
3. Click **Setup DL Environment...**
4. Review the download size warning (~2-4 GB)
5. Optionally uncheck **ONNX export support** to reduce download size (~200 MB savings)
6. Click **Begin Setup**
7. Wait for the environment to download and configure (may take several minutes depending on connection speed)
8. When complete, click **Close** -- the training and inference menu items will appear automatically

### What gets downloaded

The setup wizard uses [pixi](https://pixi.sh/) (via Appose) to create an isolated Python environment containing:

- Python 3.10
- PyTorch 2.1+ (with CUDA support on Windows/Linux)
- segmentation-models-pytorch
- NumPy, Pillow, scikit-image
- ONNX and ONNX Runtime (optional)

### Environment location

The environment is stored at:

| OS | Path |
|----|------|
| Windows | `C:\Users\<you>\.local\share\appose\dl-pixel-classifier\` |
| macOS | `~/.local/share/appose/dl-pixel-classifier/` |
| Linux | `~/.local/share/appose/dl-pixel-classifier/` |

### Rebuilding the environment

If the environment becomes corrupted or you want a fresh install:

1. Go to **Extensions > DL Pixel Classifier > Utilities > Rebuild DL Environment...**
2. Confirm the rebuild (this deletes the existing environment)
3. Complete the setup wizard again

## Part 3: GPU Configuration

### NVIDIA GPU (CUDA)

1. Install the latest NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers)
   - **Windows:** Install "Game Ready" or "Studio" drivers
   - **Linux:** Install via your distribution's package manager or NVIDIA's `.run` installer
2. **Important:** NVIDIA drivers must be installed **before** running the environment setup. If you installed drivers after setup, use **Utilities > Rebuild DL Environment...** to reinstall.

### Verifying GPU detection (Appose mode)

After completing the setup wizard, verify that the GPU was detected:

1. **Setup dialog completion message** -- the dialog reports which GPU backend was found (CUDA, MPS, or CPU)
2. **Python Console** -- go to **Extensions > DL Pixel Classifier > Utilities > Python Console** and look for:
   - `CUDA available: True` (NVIDIA GPU)
   - `MPS available: True` (Apple Silicon)
3. **System Info** -- go to **Extensions > DL Pixel Classifier > Utilities > System Info** for a full diagnostic dump including PyTorch version, CUDA version, and GPU details

### Apple Silicon (MPS)

MPS support is automatic on macOS with Apple Silicon (M-series chips). No additional configuration needed.

### CPU fallback

If no GPU is detected, the backend automatically falls back to CPU. Training will be slower but functional.

## Part 4: Verifying the Complete Setup

### Appose mode (default)

1. Open QuPath with the extension installed
2. You should see **Extensions > DL Pixel Classifier** in the menu bar
3. If this is first time: only **Setup DL Environment...** and **Utilities > Server Settings** are visible
4. After running setup: all workflow items (Train, Apply, Toggle Prediction Overlay, etc.) appear
5. Open the **Python Console** (Utilities menu) to verify GPU status

### HTTP mode

1. Start the Python server
2. Open QuPath with the extension installed
3. Disable Appose in preferences
4. All workflow items should be visible immediately
5. Use **Utilities > Server Settings** to test the connection

If issues occur, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Part 5: Building from Source (for Developers)

> This section is for **developers** who want to build the extension from source. End-users should download the pre-built JAR from GitHub Releases (see Part 1).

### Build the extension

```bash
git clone https://github.com/MichaelSNelson/qupath-extension-DL-pixel-classifier.git
cd qupath-extension-DL-pixel-classifier
./gradlew build
```

This produces a JAR file in `build/libs/`. Copy it to your QuPath extensions directory and restart QuPath.

### Shadow JAR (bundled dependencies)

For a self-contained JAR that includes all dependencies:

```bash
./gradlew shadowJar
```

### Running Python tests

```bash
cd python_server
pip install -e ".[dev]"
pytest tests/ -v
```

### Java build requirements

- Java JDK 21+
- Gradle (wrapper included in the repository)

## Part 6: Alternative -- External Python Server (HTTP Mode)

> This is for **advanced setups** where the Python backend runs on a different machine (e.g., a dedicated GPU workstation). Most users should use the default Appose mode (Part 2).

### 6a. Disable Appose in QuPath

1. Go to **Edit > Preferences > DL Pixel Classifier**
2. Uncheck **Use Appose (Embedded Python)**
3. All workflow menu items will appear immediately (no environment setup needed)

### 6b. Set up the Python server

On the machine that will run the server:

**Create a virtual environment:**

```bash
cd python_server
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# or: venv\Scripts\activate.bat  (Windows CMD)
# or: venv\Scripts\Activate.ps1  (Windows PowerShell)
```

**Install dependencies:**

```bash
# With NVIDIA GPU (recommended):
pip install -e ".[cuda]"

# CPU only or Apple Silicon:
pip install -e .
```

> **Specific CUDA version?** Install PyTorch first, then the server package:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install -e .
> ```

### 6c. Start the server

```bash
dlclassifier-server
```

You should see output like:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8765
```

### 6d. Configure QuPath

1. In QuPath, go to **Extensions > DL Pixel Classifier > Utilities > Server Settings**
2. Set the host to the server machine's IP address (or `localhost` for same machine)
3. Set the port to `8765` (default)
4. Ensure firewall rules allow traffic on the configured port

### 6e. Verify the connection

```bash
curl http://localhost:8765/api/v1/health
```

Expected response:
```json
{"status": "healthy"}
```

> **Windows without curl:** Open `http://localhost:8765/docs` in a browser for the interactive Swagger UI.
