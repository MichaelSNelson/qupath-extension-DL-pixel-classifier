# Installation Guide

Complete setup instructions for the DL Pixel Classifier extension, covering the Java extension, Python server, and GPU configuration.

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| QuPath | 0.6.0 or later |
| Java JDK | 21+ (for building the extension) |
| Python | 3.10+ |
| GPU | NVIDIA GPU with CUDA recommended; Apple Silicon (MPS) also works; CPU fallback available |

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

## Part 2: Python Server

### 2a. Create a virtual environment

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

### 2b. Install dependencies

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

### 2c. Start the server

```bash
dlclassifier-server
```

You should see output like:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8765
```

### 2d. Verify it works

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

## Part 3: GPU Configuration

### NVIDIA GPU (CUDA)

1. Install the latest NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers)
2. Verify CUDA availability:
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
   ```
3. For best performance, ensure the PyTorch CUDA version matches your driver's CUDA version

### Apple Silicon (MPS)

MPS support is automatic on macOS with Apple Silicon. Verify:
```bash
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### CPU fallback

If no GPU is detected, the server automatically falls back to CPU. Training will be slower but functional.

## Part 4: Remote Server Setup

If the Python server runs on a different machine (e.g., a GPU workstation):

1. Start the server on the remote machine:
   ```bash
   dlclassifier-server --host 0.0.0.0 --port 8765
   ```
2. In QuPath, go to **Extensions > DL Pixel Classifier > Utilities > Server Settings**
3. Set the host to the remote machine's IP address and the port to 8765
4. Ensure firewall rules allow traffic on port 8765

## Verifying the Complete Setup

1. Start the Python server
2. Open QuPath with the extension installed
3. You should see **Extensions > DL Pixel Classifier** in the menu bar
4. If the server is not running, a warning notification will appear

If issues occur, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
