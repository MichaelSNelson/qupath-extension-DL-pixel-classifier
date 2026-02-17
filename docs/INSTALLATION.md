# Installation Guide

Complete setup instructions for the DL Pixel Classifier extension.

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| QuPath | 0.6.0 or later |
| Java JDK | 21+ (for building the extension) |
| GPU | NVIDIA GPU with CUDA recommended; Apple Silicon (MPS) also works; CPU fallback available |
| Internet | Required for first-time environment setup (~2-4 GB download) |

> **Note:** A separate Python installation is **not** required for the default Appose backend. The extension manages its own embedded Python environment.

## Part 1: Java Extension

### Build from source

```bash
git clone https://github.com/MichaelSNelson/qupath-extension-DL-pixel-classifier.git
cd qupath-extension-DL-pixel-classifier
./gradlew build
```

This produces a JAR file in `build/libs/`.

### Install into QuPath

Copy the JAR to your QuPath extensions directory:

| OS | Typical extensions path |
|----|------------------------|
| Windows | `C:\Users\<you>\AppData\Local\QuPath\v0.6\extensions\` |
| macOS | `~/Library/Application Support/QuPath/v0.6/extensions/` |
| Linux | `~/.local/share/QuPath/v0.6/extensions/` |

Alternatively, in QuPath: **Edit > Preferences > Extensions** shows the extensions directory path. Drop the JAR there and restart QuPath.

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
| Windows | `C:\Users\<you>\.appose\pixi\dl-pixel-classifier\` |
| macOS | `~/.appose/pixi/dl-pixel-classifier/` |
| Linux | `~/.appose/pixi/dl-pixel-classifier/` |

### Rebuilding the environment

If the environment becomes corrupted or you want a fresh install:

1. Go to **Extensions > DL Pixel Classifier > Utilities > Rebuild DL Environment...**
2. Confirm the rebuild (this deletes the existing environment)
3. Complete the setup wizard again

## Part 3: Alternative -- External Python Server (HTTP Mode)

For advanced setups where the Python backend runs on a different machine (e.g., a dedicated GPU workstation), you can disable Appose and connect to an external server instead.

### 3a. Disable Appose in QuPath

1. Go to **Edit > Preferences > DL Pixel Classifier**
2. Uncheck **Use Appose (Embedded Python)**
3. All workflow menu items will appear immediately (no environment setup needed)

### 3b. Set up the Python server

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

### 3c. Start the server

```bash
dlclassifier-server
```

You should see output like:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8765
```

### 3d. Configure QuPath

1. In QuPath, go to **Extensions > DL Pixel Classifier > Utilities > Server Settings**
2. Set the host to the server machine's IP address (or `localhost` for same machine)
3. Set the port to `8765` (default)
4. Ensure firewall rules allow traffic on the configured port

### 3e. Verify the connection

```bash
curl http://localhost:8765/api/v1/health
```

Expected response:
```json
{"status": "healthy"}
```

> **Windows without curl:** Open `http://localhost:8765/docs` in a browser for the interactive Swagger UI.

## Part 4: GPU Configuration

### NVIDIA GPU (CUDA)

1. Install the latest NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers)
2. Verify CUDA availability:
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
   ```
3. For best performance, ensure the PyTorch CUDA version matches your driver's CUDA version

> With Appose mode, the embedded environment automatically installs CUDA-compatible PyTorch on Windows and Linux.

### Apple Silicon (MPS)

MPS support is automatic on macOS with Apple Silicon. Verify:
```bash
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### CPU fallback

If no GPU is detected, the backend automatically falls back to CPU. Training will be slower but functional.

## Verifying the Complete Setup

### Appose mode (default)

1. Open QuPath with the extension installed
2. You should see **Extensions > DL Pixel Classifier** in the menu bar
3. If this is first time: only **Setup DL Environment...** and **Utilities > Server Settings** are visible
4. After running setup: all workflow items (Train, Apply, Live Prediction, etc.) appear

### HTTP mode

1. Start the Python server
2. Open QuPath with the extension installed
3. Disable Appose in preferences
4. All workflow items should be visible immediately
5. Use **Utilities > Server Settings** to test the connection

If issues occur, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
