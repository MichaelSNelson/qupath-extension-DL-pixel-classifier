"""
Appose worker initialization script.

Called once via pythonService.init() when the worker process starts.
Sets up persistent globals that remain available across all task() calls.

CRITICAL: All output must go to sys.stderr, NOT sys.stdout.
Appose uses stdout for its JSON-based IPC protocol.
Any print() call corrupts the protocol and crashes communication.
"""
import sys
import os
import logging
import threading
import time

# Configure logging to stderr (stdout is reserved for Appose JSON protocol)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("dlclassifier.appose")

# The dlclassifier_server package is installed in the pixi environment
# via the git URL in pixi.toml. No sys.path manipulation needed.

# Import and initialize persistent services.
# ImportError is NOT caught here -- if critical packages (torch, smp, etc.)
# are missing, the init must fail loudly so the Java side knows the
# environment is broken. Runtime errors (e.g. no GPU) are caught and
# handled gracefully so the service can still run in CPU mode.
from dlclassifier_server.services.gpu_manager import GPUManager
from dlclassifier_server.services.inference_service import InferenceService
from dlclassifier_server.services.model_registry import ModelRegistry

try:
    gpu_manager = GPUManager()
    inference_service = InferenceService(device="auto", gpu_manager=gpu_manager)
    model_registry = ModelRegistry()

    logger.info("DL classifier services initialized successfully")
    logger.info("Device: %s", inference_service._device_str)

    # Threading lock for GPU operations.
    # Appose runs each task in its own thread. Without serialization,
    # concurrent tile inference tasks race on model loading, CUDA memory
    # allocation, and forward passes. PyTorch CUDA ops are thread-safe
    # but concurrent batches can OOM and torch.compile is NOT thread-safe.
    _inference_lock = threading.Lock()

except Exception as e:
    logger.error("Failed to initialize DL classifier services: %s", e)
    # Store error so tasks can report it -- imports succeeded but
    # runtime initialization failed (e.g. GPU not available)
    _init_error = str(e)
    gpu_manager = None
    inference_service = None
    model_registry = None


# --- Parent process watcher ---
# Safety net: if QuPath is force-killed (Task Manager, kill -9), the JVM
# shutdown hook never runs, so stdin never closes and this process lives
# forever. This daemon thread polls the parent PID and exits if it dies.

def _parent_alive(pid):
    """Check if a process with the given PID is still running."""
    if sys.platform == 'win32':
        # os.kill(pid, 0) does NOT work on Windows -- signal 0 maps to
        # CTRL_C_EVENT which crashes the process. Use the Win32 API instead.
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)  # Signal 0 = existence check (Unix only)
            return True
        except PermissionError:
            return True  # Process exists but no permission
        except OSError:
            return False  # Process does not exist


def _watch_parent():
    """Exit if parent process (Java/QuPath) dies."""
    ppid = os.getppid()
    if ppid <= 1:
        return  # No meaningful parent to watch (already orphaned or init)
    logger.info("Parent process watcher started (parent PID: %d)", ppid)
    while True:
        time.sleep(3)
        try:
            # Check 1: parent PID changed (Linux reparents to init/systemd)
            current_ppid = os.getppid()
            if current_ppid != ppid:
                logger.warning("Parent process changed (%d -> %d), exiting",
                               ppid, current_ppid)
                os._exit(1)
            # Check 2: parent PID no longer exists
            if not _parent_alive(ppid):
                logger.warning("Parent process %d no longer exists, exiting",
                               ppid)
                os._exit(1)
        except Exception as e:
            # Never crash the thread -- log and keep watching
            logger.debug("Parent watcher check error: %s", e)


_parent_watcher = threading.Thread(target=_watch_parent, daemon=True)
_parent_watcher.start()
