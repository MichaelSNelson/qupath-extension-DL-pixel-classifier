# Troubleshooting

Common issues and solutions for the DL Pixel Classifier extension.

## Viewing Logs and Diagnostics

Before diving into specific issues, know where to find diagnostic information:

### QuPath log

**View > Show log** in QuPath displays Java-side messages from the extension, including backend initialization, error messages, and workflow progress.

### Python Console

**Extensions > DL Pixel Classifier > Utilities > Python Console** shows all Python-side output (stderr) in real time:

- GPU initialization messages (e.g., "CUDA available: True")
- Model loading and inference progress
- Python errors and stack traces
- Normalization statistics and processing details

Use **Copy to Clipboard** to capture the full log for bug reports.

> **Tip:** Keep the Python Console open during training and inference to monitor the Python backend's activity.

### System Info

**Extensions > DL Pixel Classifier > Utilities > System Info** provides a complete diagnostic dump:

- PyTorch version and build info
- CUDA version and GPU details (name, VRAM, driver version)
- Installed Python package versions
- Backend mode (Appose or HTTP)

Use **Copy to Clipboard** to share the full output when reporting bugs.

## Environment Setup Issues (Appose Mode)

### Setup dialog shows an error

| Symptom | Cause | Fix |
|---------|-------|-----|
| Network error during setup | No internet connection | Connect to the internet and click **Retry** |
| Download stalls or times out | Slow/unstable connection | Cancel and retry; the ~2-4 GB download may take several minutes |
| "Resource not found" | JAR may be corrupted | Re-download the JAR from GitHub Releases and reinstall |

### Setup completes but GPU not detected

If the setup wizard reports CPU-only but you have an NVIDIA GPU:

1. **Check the Python Console** -- look for "CUDA available: False"
2. **Verify NVIDIA drivers are installed** -- open a terminal and run `nvidia-smi`. If this fails, install drivers first.
3. **Drivers must be installed before environment setup** -- the Appose environment installs PyTorch with CUDA support, but it needs to detect your GPU drivers during setup
4. **If drivers were installed after setup**: Go to **Utilities > Rebuild DL Environment...** to delete and re-create the environment with GPU support
5. **Windows-specific**: Install "Game Ready" or "Studio" drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers)

### Menu items don't appear after setup

1. Verify the environment directory exists:
   - Windows: `C:\Users\<you>\.local\share\appose\dl-pixel-classifier\.pixi\`
   - macOS/Linux: `~/.local/share/appose/dl-pixel-classifier/.pixi/`
2. Close and reopen QuPath
3. Check the QuPath log (**View > Show log**) for errors

### Environment seems corrupted

Use **Extensions > DL Pixel Classifier > Utilities > Rebuild DL Environment...** to delete the environment and re-run setup.

### Only "Setup DL Environment..." is visible

This is normal on first launch before the environment has been set up. Click it to begin the setup wizard.

If you want to use an external Python server instead, disable Appose in **Edit > Preferences > DL Pixel Classifier** (uncheck "Use Appose"). All workflow menu items will appear immediately.

## HTTP Server Issues (External Server Mode)

> These issues only apply when Appose is **disabled** and you are connecting to an external Python server.

### Server won't start

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch not installed | Activate your venv first, then `pip install -e .` |
| Port 8765 already in use | Another process using the port | Kill the other process, or start with `dlclassifier-server --port 8766` and update QuPath Server Settings |
| `externally-managed-environment` | System Python is locked | Use a virtual environment (see [INSTALLATION.md](INSTALLATION.md)) |
| Permission denied | No execute permission | Run `chmod +x` on the script, or use `python -m dlclassifier_server` |

### QuPath can't connect to server

1. Verify the server is running:
   ```bash
   curl http://localhost:8765/api/v1/health
   ```
2. Check **Extensions > DL Pixel Classifier > Utilities > Server Settings** matches the server's host and port
3. If the server is on another machine, check firewall rules allow port 8765
4. Try restarting the server

### Server crashes during training/inference

- Check the server terminal for error messages
- Use **Extensions > DL Pixel Classifier > Utilities > Free GPU Memory** to clear state
- Restart the server if it becomes unresponsive

## GPU Issues

### GPU not detected (Appose mode)

1. Open the **Python Console** (Utilities menu) and look for "CUDA available" or "MPS available"
2. Run **System Info** (Utilities menu) and check the CUDA/GPU section
3. Verify NVIDIA drivers are installed by running `nvidia-smi` in a terminal
4. If drivers were installed **after** environment setup: use **Utilities > Rebuild DL Environment...** to reinstall
5. If still not detected: check that your NVIDIA driver's CUDA version is >= the PyTorch CUDA version shown in System Info
6. **Apple Silicon:** MPS should be auto-detected. If not, verify you are running macOS 12.3+ and check System Info output

### GPU not detected (HTTP mode)

1. Check the server's startup output for GPU detection messages
2. Query the GPU endpoint:
   ```bash
   curl http://localhost:8765/api/v1/gpu
   ```
3. If CUDA is not detected:
   - Verify the correct PyTorch version is installed: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
   - Verify NVIDIA drivers: `nvidia-smi`
   - Ensure PyTorch CUDA version matches your driver

### CUDA out of memory

| Fix | Description |
|-----|-------------|
| Reduce batch size | Try 4, then 2, then 1 |
| Reduce tile size | 256 instead of 512 |
| Use smaller backbone | efficientnet-b0 instead of resnet50 |
| Free GPU memory | Use the "Free GPU Memory" utility menu item |
| Close other GPU programs | Other applications may be using VRAM |
| Disable mixed precision | Rarely helps, but try if nothing else works |

### GPU memory not freed after training

Use **Extensions > DL Pixel Classifier > Utilities > Free GPU Memory** to force-clear all GPU state. This cancels running jobs, clears cached models, and frees VRAM.

## Training Issues

### Training fails immediately

| Symptom | Cause | Fix |
|---------|-------|-----|
| "No annotations found" | No classified annotations | Create annotations and assign them to classes |
| "At least 2 classes required" | Only one class annotated | Add annotations for a second class |
| "Server error" | Server connection failed | Check server is running and accessible |
| Dialog won't open | No image loaded | Open an image first |

### Training is very slow

| Cause | Fix |
|-------|-----|
| Running on CPU | **Appose mode:** Open the Python Console and look for "CUDA available: True". If False, install NVIDIA drivers and rebuild the environment. **HTTP mode:** `curl http://localhost:8765/api/v1/gpu` |
| Mixed precision disabled | Enable in Training Strategy section |
| Very large tile size | Reduce tile size (256 instead of 512) |
| Very large batch size | Reduce batch size |
| Too much augmentation | Disable elastic deformation (slowest augmentation) |

### Training produces poor results

See [BEST_PRACTICES.md](BEST_PRACTICES.md) for detailed guidance on improving results.

Quick checklist:
- [ ] Annotations are accurate and consistent
- [ ] At least 2 classes with sufficient annotations
- [ ] Pretrained weights enabled
- [ ] Appropriate backbone for dataset size
- [ ] Augmentation enabled (at least flips and rotation)

## Inference Issues

### Inference produces blank/uniform results

- The classifier may not have trained well -- check training loss curves
- Channel configuration may not match training -- verify channel order and count
- Resolution (downsample) may differ from training

### Tile seams visible in output

Image-level normalization (enabled by default) should eliminate most tile boundary artifacts. If seams are still visible:

- **Re-train the model** -- newly trained models save training dataset normalization statistics in metadata, giving the best cross-tile consistency
- Increase tile overlap percentage (10-15% recommended)
- Use LINEAR or GAUSSIAN blend mode instead of NONE
- Verify overlap is not 0%
- For overlays, adjust the **Overlay Overlap (um)** preference (Edit > Preferences) to increase physical overlap distance

### Objects are fragmented or too small

- Increase hole filling threshold
- Decrease min object size threshold
- Increase boundary smoothing
- Consider using a larger tile size for more context

### Inference is very slow

- Enable GPU in processing options
- Reduce tile overlap (but quality may suffer)
- Use NONE blend mode for fastest processing (quality trade-off)
- Process selected annotations first to estimate total time

## Extension Issues

### Menu items are hidden (not visible)

- **First launch (Appose mode):** Only **Setup DL Environment...** and **Utilities > Server Settings** are visible until you complete the environment setup
- **After setup completes:** All workflow items should appear. If not, restart QuPath
- **HTTP mode:** Disable Appose in preferences to see all menu items immediately

### Menu items are grayed out (visible but disabled)

Menu items like Train and Apply require an open image/project. Open an image first, then the menu items will become active.

### "No classifiers available"

- Train a classifier first, or check that classifiers are saved in the project's `classifiers/` directory
- Verify the backend is running and the model storage path is accessible

### Preferences not saving

Preferences are saved automatically when you click "Start Training" or "Apply" in the dialogs. They persist across QuPath sessions via QuPath's preference system. If preferences are not saving:
- Check QuPath's preferences directory is writable
- Verify no other QuPath instance is locking the preferences file

## Advanced Diagnostics

### Appose environment details

The Appose environment is a self-contained Python installation managed by [pixi](https://pixi.sh/):

| Item | Location |
|------|----------|
| Environment root | `~/.local/share/appose/dl-pixel-classifier/` (see OS-specific paths above) |
| pixi.toml | `<env root>/pixi.toml` -- defines all Python dependencies |
| pixi.lock | `<env root>/pixi.lock` -- resolved dependency versions |
| Python installation | `<env root>/.pixi/envs/default/` |

**When to rebuild vs fresh install:**
- **Rebuild** (Utilities > Rebuild DL Environment): Deletes the existing environment and runs setup again. Use when the environment is corrupted or after driver changes.
- **Fresh install**: Manually delete the entire environment directory, then run Setup DL Environment. Use as a last resort if Rebuild does not work.

### Known transient behaviors

- **Brief pause on first inference**: The model loads into GPU memory on the first inference call. Subsequent calls are faster.
- **Occasional "thread death" during overlay rendering**: This is a transient error that auto-retries. If the overlay eventually renders correctly, no action is needed.
- **Large initial memory spike during training**: PyTorch allocates GPU memory aggressively on the first batch, then stabilizes. Monitor GPU memory over several epochs before reducing batch size.

### Reporting bugs

When filing a bug report, please include:

1. **System Info output** -- Utilities > System Info > Copy to Clipboard
2. **Python Console log** -- Utilities > Python Console > Copy to Clipboard
3. **QuPath log** -- View > Show log (copy relevant error messages)
4. **Steps to reproduce** -- what you did, what you expected, what happened
5. **Image details** -- image type (brightfield/fluorescence), channel count, bit depth

File issues at the [GitHub repository](https://github.com/MichaelSNelson/qupath-extension-DL-pixel-classifier/issues).

## Diagnostic Commands

### Appose mode diagnostics

Use the built-in tools (no terminal required):

| Tool | Menu path | What it shows |
|------|-----------|---------------|
| **Python Console** | Utilities > Python Console | Real-time Python output, GPU init, errors |
| **System Info** | Utilities > System Info | PyTorch/CUDA versions, GPU details, packages |
| **QuPath log** | View > Show log | Java-side extension messages |

### HTTP mode diagnostics

These commands apply when using the external Python server:

```bash
# Health check
curl http://localhost:8765/api/v1/health

# GPU info
curl http://localhost:8765/api/v1/gpu

# List models
curl http://localhost:8765/api/v1/models

# Interactive API docs
# Open in browser: http://localhost:8765/docs
```

### Terminal GPU verification (both modes)

If you suspect GPU driver issues, verify from a terminal:

```bash
# Check NVIDIA driver and GPU
nvidia-smi

# Check PyTorch GPU access (if Python is available)
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
```

> **Note:** For Appose mode, the Python Console and System Info utilities provide the same information without requiring terminal access.
