"""Job management service for training jobs."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Training job status."""
    PENDING = "pending"
    TRAINING = "training"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """A training job."""
    job_id: str
    model_type: str
    architecture: Dict[str, Any]
    input_config: Dict[str, Any]
    training_params: Dict[str, Any]
    classes: List[str]
    data_path: str
    device: str
    model_registry: Optional[Any] = None

    status: JobStatus = JobStatus.PENDING
    current_epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    train_loss: float = 0.0
    accuracy: float = 0.0
    per_class_iou: Dict[str, float] = field(default_factory=dict)
    per_class_loss: Dict[str, float] = field(default_factory=dict)
    mean_iou: float = 0.0
    model_path: Optional[str] = None
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None

    _cancel_flag: threading.Event = field(default_factory=threading.Event)
    _pause_flag: threading.Event = field(default_factory=threading.Event)

    def run(self, checkpoint_path: Optional[str] = None, start_epoch: int = 0):
        """Execute the training job.

        Args:
            checkpoint_path: Optional checkpoint path for resuming training
            start_epoch: Epoch to start from when resuming (0-based)
        """
        try:
            logger.info(f"Starting training job {self.job_id}")
            self.status = JobStatus.TRAINING
            self.total_epochs = self.training_params.get("epochs", 50)

            # Import training service
            from .training_service import TrainingService

            trainer = TrainingService(device=self.device)

            # Run training with progress callback
            def progress_callback(epoch, train_loss, val_loss, accuracy,
                                  per_class_iou, per_class_loss, mean_iou):
                self.current_epoch = epoch
                self.train_loss = train_loss
                self.loss = val_loss
                self.accuracy = accuracy
                self.per_class_iou = per_class_iou
                self.per_class_loss = per_class_loss
                self.mean_iou = mean_iou

            # Extract frozen layers from architecture config
            frozen_layers = self.architecture.get("frozen_layers", None)

            result = trainer.train(
                model_type=self.model_type,
                architecture=self.architecture,
                input_config=self.input_config,
                training_params=self.training_params,
                classes=self.classes,
                data_path=self.data_path,
                progress_callback=progress_callback,
                cancel_flag=self._cancel_flag,
                frozen_layers=frozen_layers,
                pause_flag=self._pause_flag,
                checkpoint_path=checkpoint_path,
                start_epoch=start_epoch
            )

            if self._cancel_flag.is_set():
                self.status = JobStatus.CANCELLED
                logger.info(f"Training job {self.job_id} cancelled")
            elif result.get("status") == "paused":
                self.status = JobStatus.PAUSED
                self.checkpoint_path = result["checkpoint_path"]
                logger.info(f"Training job {self.job_id} paused at epoch {result['epoch']}")
            else:
                self.status = JobStatus.COMPLETED
                self.model_path = result["model_path"]
                self.loss = result["final_loss"]
                self.accuracy = result["final_accuracy"]
                logger.info(f"Training job {self.job_id} completed")

                # Register the new model so it is immediately visible via the API
                self._register_model(result["model_path"])

        except Exception as e:
            logger.error(f"Training job {self.job_id} failed: {e}")
            self.status = JobStatus.FAILED
            self.error = str(e)

    def _register_model(self, model_path: str) -> None:
        """Register a newly trained model in the model registry."""
        if self.model_registry is None:
            return

        try:
            metadata_path = Path(model_path) / "metadata.json"
            if not metadata_path.exists():
                logger.warning(f"No metadata.json at {model_path}, skipping registration")
                return

            with open(metadata_path) as f:
                metadata = json.load(f)

            model_id = metadata.get("id", Path(model_path).name)
            model_name = metadata.get("name", model_id)
            model_type = metadata.get("architecture", {}).get("type", "unknown")

            self.model_registry.register_model(
                model_id=model_id,
                name=model_name,
                model_type=model_type,
                model_path=model_path,
                metadata=metadata
            )
            logger.info(f"Registered trained model: {model_id}")

        except Exception as e:
            logger.warning(f"Failed to register model after training: {e}")

    def cancel(self):
        """Cancel the training job."""
        self._cancel_flag.set()

    def pause(self):
        """Request training to pause at the end of the current epoch."""
        self._pause_flag.set()

    def resume(self, data_path: Optional[str] = None,
               training_params_overrides: Optional[Dict[str, Any]] = None):
        """Resume a paused training job.

        Args:
            data_path: Optional new data path (for re-exported annotations)
            training_params_overrides: Optional dict of training param overrides
                (epochs, learning_rate, batch_size)
        """
        if self.status != JobStatus.PAUSED:
            raise ValueError(f"Cannot resume job in state {self.status.value}")

        # Apply overrides
        if data_path:
            self.data_path = data_path
        if training_params_overrides:
            self.training_params.update(training_params_overrides)
            self.total_epochs = self.training_params.get("epochs", self.total_epochs)

        # Clear pause flag for next use
        self._pause_flag.clear()
        self._cancel_flag = threading.Event()

        start_epoch = self.current_epoch
        checkpoint = self.checkpoint_path
        self.checkpoint_path = None
        self.status = JobStatus.TRAINING

        self.run(checkpoint_path=checkpoint, start_epoch=start_epoch)

    def get_status(self) -> Dict[str, Any]:
        """Get job status as dictionary."""
        result = {
            "status": self.status.value,
        }

        if self.status == JobStatus.TRAINING:
            result["epoch"] = self.current_epoch
            result["total_epochs"] = self.total_epochs
            result["loss"] = self.loss
            result["train_loss"] = self.train_loss
            result["accuracy"] = self.accuracy
            result["mean_iou"] = self.mean_iou
            result["per_class_iou"] = self.per_class_iou
            result["per_class_loss"] = self.per_class_loss

        elif self.status == JobStatus.PAUSED:
            result["epoch"] = self.current_epoch
            result["total_epochs"] = self.total_epochs
            result["loss"] = self.loss
            result["train_loss"] = self.train_loss
            result["accuracy"] = self.accuracy
            result["mean_iou"] = self.mean_iou
            result["per_class_iou"] = self.per_class_iou
            result["per_class_loss"] = self.per_class_loss
            result["checkpoint_path"] = self.checkpoint_path

        elif self.status == JobStatus.COMPLETED:
            result["model_path"] = self.model_path
            result["final_loss"] = self.loss
            result["final_accuracy"] = self.accuracy

        elif self.status == JobStatus.FAILED:
            result["error"] = self.error

        return result


class JobManager:
    """Manages training jobs."""

    def __init__(
        self,
        max_workers: int = 2,
        model_registry: Optional["ModelRegistry"] = None
    ):
        self._jobs: Dict[str, TrainingJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._model_registry = model_registry
        logger.info(f"JobManager initialized with {max_workers} workers")

    def create_training_job(
        self,
        job_id: str,
        model_type: str,
        architecture: Dict[str, Any],
        input_config: Dict[str, Any],
        training_params: Dict[str, Any],
        classes: List[str],
        data_path: str,
        device: str
    ) -> TrainingJob:
        """Create a new training job."""
        job = TrainingJob(
            job_id=job_id,
            model_type=model_type,
            architecture=architecture,
            input_config=input_config,
            training_params=training_params,
            classes=classes,
            data_path=data_path,
            device=device,
            model_registry=self._model_registry
        )

        self._jobs[job_id] = job
        logger.info(f"Created training job: {job_id}")
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[TrainingJob]:
        """List all jobs."""
        return list(self._jobs.values())

    def shutdown(self):
        """Shutdown the job manager."""
        logger.info("Shutting down JobManager")
        self._executor.shutdown(wait=False)
