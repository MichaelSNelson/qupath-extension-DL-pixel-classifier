"""
Comprehensive system information for the DL Pixel Classifier.

Collects GPU status, Python package versions, platform details, and
threading configuration. Results returned as task outputs for display
in a copyable text dialog.

Outputs:
    info_text: str  (pre-formatted multi-line report)
"""
import sys
import os
import platform
import logging

logger = logging.getLogger("dlclassifier.appose.sysinfo")

lines = []

# --- Python ---
lines.append("=== Python ===")
lines.append("Python version: %s" % platform.python_version())
lines.append("Python executable: %s" % sys.executable)
lines.append("Platform: %s" % platform.platform())
lines.append("")

# --- GPU / Device ---
lines.append("=== GPU / Device ===")
try:
    import torch
    lines.append("PyTorch version: %s" % torch.__version__)

    if torch.cuda.is_available():
        lines.append("CUDA available: True")
        lines.append("CUDA version: %s" % torch.version.cuda)
        lines.append("cuDNN version: %s" % torch.backends.cudnn.version())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            lines.append("GPU %d: %s" % (i, props.name))
            lines.append("  Memory: %d MB" % (props.total_mem // (1024 * 1024)))
            lines.append("  Compute capability: %d.%d" % (props.major, props.minor))
    else:
        lines.append("CUDA available: False")

    mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    lines.append("MPS available: %s" % mps)
    if mps:
        lines.append("MPS device: Apple Silicon (Metal Performance Shaders)")

    if not torch.cuda.is_available() and not mps:
        lines.append("Active device: CPU (no GPU acceleration)")
    elif torch.cuda.is_available():
        lines.append("Active device: cuda")
    else:
        lines.append("Active device: mps")
except ImportError:
    lines.append("PyTorch: NOT INSTALLED")
except Exception as e:
    lines.append("PyTorch error: %s" % e)

lines.append("")

# --- Package Versions ---
lines.append("=== Python Packages ===")
packages = [
    "torch", "torchvision", "segmentation_models_pytorch",
    "albumentations", "timm", "ttach", "numpy", "PIL", "skimage",
    "onnx", "onnxruntime", "pydantic", "tifffile",
    "imagecodecs", "appose"
]
for pkg_name in packages:
    try:
        mod = __import__(pkg_name)
        version = getattr(mod, "__version__", "installed (version unknown)")
        # Some packages use different import names
        display_name = pkg_name
        if pkg_name == "PIL":
            display_name = "Pillow"
        elif pkg_name == "skimage":
            display_name = "scikit-image"
        lines.append("%s: %s" % (display_name, version))
    except ImportError:
        lines.append("%s: NOT INSTALLED" % pkg_name)
    except Exception as e:
        lines.append("%s: error (%s)" % (pkg_name, e))

lines.append("")

# --- DL Classifier Server ---
lines.append("=== DL Classifier Server ===")
try:
    import dlclassifier_server
    version = getattr(dlclassifier_server, "__version__", "unknown")
    lines.append("dlclassifier-server version: %s" % version)
except ImportError:
    lines.append("dlclassifier-server: NOT INSTALLED")
except Exception as e:
    lines.append("dlclassifier-server: error (%s)" % e)

# Service status
if inference_service is not None:
    lines.append("InferenceService: initialized")
    lines.append("InferenceService device: %s" % inference_service._device_str)
else:
    err = globals().get("init_error", "unknown")
    lines.append("InferenceService: NOT initialized (%s)" % err)

if gpu_manager is not None:
    info = gpu_manager.get_info()
    lines.append("GPUManager device: %s" % info.get("device_string", "unknown"))
else:
    lines.append("GPUManager: NOT initialized")

if model_registry is not None:
    models = model_registry.list_models()
    lines.append("Cached models: %d" % len(models))
else:
    lines.append("ModelRegistry: NOT initialized")

lines.append("")

# --- Threading / CPU ---
lines.append("=== CPU / Threading ===")
lines.append("CPU count: %d" % os.cpu_count())
try:
    import torch
    lines.append("PyTorch threads (intra-op): %d" % torch.get_num_threads())
    lines.append("PyTorch threads (inter-op): %d" % torch.get_num_interop_threads())
except Exception:
    pass

info_text = "\n".join(lines)
task.outputs["info_text"] = info_text
logger.info("System info collected successfully")
