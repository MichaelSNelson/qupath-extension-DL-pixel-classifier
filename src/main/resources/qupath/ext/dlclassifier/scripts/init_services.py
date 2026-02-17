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

# Configure logging to stderr (stdout is reserved for Appose JSON protocol)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("dlclassifier.appose")

# Add the dlclassifier_server package to sys.path if not already importable.
# This allows Appose to import the existing Python services without requiring
# the package to be installed in the pixi environment.
# The server_package_path is set by ApposeClassifierBackend before init.
_server_path = globals().get("server_package_path", "")
if _server_path and os.path.isdir(_server_path):
    parent = os.path.dirname(_server_path)
    if parent not in sys.path:
        sys.path.insert(0, parent)
        logger.info("Added %s to sys.path for dlclassifier_server imports", parent)

# Import and initialize persistent services
try:
    from dlclassifier_server.services.gpu_manager import GPUManager
    from dlclassifier_server.services.inference_service import InferenceService
    from dlclassifier_server.services.model_registry import ModelRegistry

    gpu_manager = GPUManager()
    inference_service = InferenceService(device="auto", gpu_manager=gpu_manager)
    model_registry = ModelRegistry()

    logger.info("DL classifier services initialized successfully")
    logger.info("Device: %s", inference_service._device_str)

except Exception as e:
    logger.error("Failed to initialize DL classifier services: %s", e)
    # Store error so tasks can report it
    _init_error = str(e)
    gpu_manager = None
    inference_service = None
    model_registry = None
