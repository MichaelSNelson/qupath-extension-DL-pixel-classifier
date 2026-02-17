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

# The dlclassifier_server package is installed in the pixi environment
# via the git URL in pixi.toml. No sys.path manipulation needed.

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
