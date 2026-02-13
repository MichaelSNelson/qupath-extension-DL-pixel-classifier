#!/bin/bash
# ===========================================================================
#  DL Pixel Classifier - Python Server Setup
#
#  GPU auto-detection: automatically detects NVIDIA GPU and installs the
#  correct CUDA-enabled PyTorch. Use --cpu to force CPU-only mode.
#
#  Usage:
#    ./setup_server.sh                         (venv, auto-detect GPU)
#    ./setup_server.sh /fast/dl-env            (venv at custom path)
#    ./setup_server.sh --cpu                   (force CPU-only)
#    ./setup_server.sh --cuda                  (force CUDA)
#    ./setup_server.sh /fast/dl-env --cuda     (custom path, force CUDA)
# ===========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/python_server"
VENV_PATH=""
USE_CUDA=0
FORCE_CPU=0
FORCE_CUDA=0
GPU_NAME=""
CUDA_VER=""
CUDA_MAJOR=""
CUDA_INDEX_URL=""
PYTORCH_CUDA_TAG=""
CONFIG_FILE="$SCRIPT_DIR/.server_config"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)    FORCE_CUDA=1; shift ;;
        --cpu)     FORCE_CPU=1; shift ;;
        --help|-h)
            echo ""
            echo "Usage: $0 [VENV_PATH] [--cuda] [--cpu]"
            echo ""
            echo "GPU Detection:"
            echo "  By default, the script auto-detects NVIDIA GPUs via nvidia-smi."
            echo "  If a GPU is found, CUDA-enabled PyTorch is installed automatically."
            echo "  If no GPU is found, CPU-only PyTorch is installed."
            echo ""
            echo "  VENV_PATH    Path for the virtual environment (default: python_server/venv)"
            echo "  --cuda       Force CUDA install (error if no GPU detected)"
            echo "  --cpu        Force CPU-only install (skip GPU detection)"
            echo ""
            echo "Examples:"
            echo "  $0                              # Auto-detect GPU, default location"
            echo "  $0 --cpu                        # Force CPU, default location"
            echo "  $0 --cuda                       # Force CUDA, default location"
            echo "  $0 /fast/dl-env                 # Auto-detect, custom path"
            echo "  $0 /fast/dl-env --cuda          # Custom path, force CUDA"
            echo ""
            exit 0
            ;;
        *)         VENV_PATH="$1"; shift ;;
    esac
done

# Validate flag combinations
if [ "$FORCE_CPU" -eq 1 ] && [ "$FORCE_CUDA" -eq 1 ]; then
    echo ""
    echo " ERROR: Cannot use both --cpu and --cuda flags."
    exit 1
fi

# Default venv path
if [ -z "$VENV_PATH" ]; then
    VENV_PATH="$SERVER_DIR/venv"
fi

# ---- GPU auto-detection ----
detect_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo ""
        echo " No NVIDIA GPU detected (nvidia-smi not found). Installing CPU-only PyTorch."
        USE_CUDA=0
        CUDA_INDEX_URL=""
        return
    fi

    # Check if nvidia-smi actually runs (driver might be missing)
    if ! nvidia-smi &>/dev/null; then
        echo ""
        echo " No NVIDIA GPU detected (nvidia-smi failed). Installing CPU-only PyTorch."
        USE_CUDA=0
        CUDA_INDEX_URL=""
        return
    fi

    # Parse GPU name
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)

    # Parse CUDA version from nvidia-smi header
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/')

    # Extract major version
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)

    echo ""
    echo " Detected GPU: $GPU_NAME"
    echo " CUDA driver version: $CUDA_VER"

    if [ "$CUDA_MAJOR" = "12" ]; then
        USE_CUDA=1
        CUDA_INDEX_URL="https://download.pytorch.org/whl/cu124"
        PYTORCH_CUDA_TAG="12.4"
        echo " PyTorch CUDA target: 12.4"
    elif [ "$CUDA_MAJOR" = "11" ]; then
        USE_CUDA=1
        CUDA_INDEX_URL="https://download.pytorch.org/whl/cu118"
        PYTORCH_CUDA_TAG="11.8"
        echo " PyTorch CUDA target: 11.8"
    else
        echo " WARNING: CUDA $CUDA_VER is too old for current PyTorch. Installing CPU-only."
        USE_CUDA=0
        CUDA_INDEX_URL=""
    fi
}

if [ "$FORCE_CPU" -eq 1 ]; then
    echo ""
    echo " CPU-only mode requested (--cpu flag)."
    USE_CUDA=0
else
    detect_gpu

    if [ "$FORCE_CUDA" -eq 1 ] && [ "$USE_CUDA" -eq 0 ]; then
        echo ""
        echo " ERROR: --cuda flag specified but no NVIDIA GPU was detected."
        echo " If you have an NVIDIA GPU, ensure drivers are installed and nvidia-smi is in PATH."
        exit 1
    fi
fi

# ---- Display setup info ----
echo ""
echo " DL Pixel Classifier - Server Setup"
echo " ==================================="
echo ""
echo " Server source: $SERVER_DIR"
echo " Virtual env:   $VENV_PATH"
if [ "$USE_CUDA" -eq 1 ]; then
    echo " GPU support:   CUDA $PYTORCH_CUDA_TAG"
else
    echo " GPU support:   CPU only"
fi
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo " ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi

python3 --version

# Create virtual environment
if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
    echo " Virtual environment already exists at $VENV_PATH"
    read -p " Recreate it? (y/N): " RECREATE
    if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
        echo " Removing old environment..."
        rm -rf "$VENV_PATH"
    else
        echo " Reusing existing environment."
    fi
fi

if [ ! -d "$VENV_PATH" ]; then
    echo " Creating virtual environment at $VENV_PATH ..."
    python3 -m venv "$VENV_PATH"
    echo " Virtual environment created."
fi

echo ""
echo " Installing dependencies..."
echo " (This may take several minutes for PyTorch download)"
echo ""

source "$VENV_PATH/bin/activate"
pip install --upgrade pip > /dev/null 2>&1

if [ "$USE_CUDA" -eq 1 ]; then
    echo " Installing PyTorch with CUDA $PYTORCH_CUDA_TAG support..."
    pip install torch torchvision --index-url "$CUDA_INDEX_URL"
    echo ""
    echo " Installing server package with GPU extras..."
    pip install -e "$SERVER_DIR[cuda]"
else
    echo " Installing CPU-only..."
    pip install -e "$SERVER_DIR"
fi

# Save config
cat > "$CONFIG_FILE" <<CONF
VENV_PATH=$VENV_PATH
SERVER_DIR=$SERVER_DIR
CONF

# Create start script
cat > "$SCRIPT_DIR/start_server.sh" <<LAUNCHER
#!/bin/bash
# Auto-generated by setup_server.sh
source "$VENV_PATH/bin/activate"
echo "Starting DL Pixel Classifier server..."
echo "Press Ctrl+C to stop."
echo ""
dlclassifier-server
LAUNCHER
chmod +x "$SCRIPT_DIR/start_server.sh"

echo ""
echo " ==================================="
echo " Setup complete!"
echo " ==================================="
echo ""
echo " To start the server:"
echo "   ./start_server.sh"
echo ""

# GPU check
python3 -c "import torch; print(f' PyTorch {torch.__version__}'); print(f' CUDA: {torch.cuda.is_available()}'); print(f' Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else ' (CPU mode)')" 2>/dev/null || echo " (PyTorch GPU check skipped)"
echo ""

read -p " Start server now? (Y/n): " START_NOW
if [[ ! "$START_NOW" =~ ^[Nn]$ ]]; then
    dlclassifier-server
fi
