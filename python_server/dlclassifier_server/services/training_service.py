"""Training service for deep learning models.

Includes:
- Data augmentation via albumentations
- Learning rate scheduling (cosine annealing with warm restarts)
- Early stopping with patience
- GPU memory monitoring and management
- Support for CUDA, Apple MPS, and CPU training
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, OneCycleLR
from PIL import Image

from .gpu_manager import GPUManager, get_gpu_manager

logger = logging.getLogger(__name__)

# Try to import albumentations for augmentation
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logger.warning("albumentations not available - augmentation will be disabled")


def get_training_augmentation(
    image_size: int = 512,
    p_flip: float = 0.5,
    p_rotate: float = 0.5,
    p_elastic: float = 0.3,
    p_color: float = 0.3,
    p_noise: float = 0.2
) -> Optional[A.Compose]:
    """Create training augmentation pipeline.

    Args:
        image_size: Expected image size (for padding/cropping)
        p_flip: Probability of flip transforms
        p_rotate: Probability of rotation
        p_elastic: Probability of elastic deformation
        p_color: Probability of color jitter
        p_noise: Probability of noise addition

    Returns:
        Albumentations Compose object or None if not available
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    return A.Compose([
        # Spatial transforms (applied to both image and mask)
        A.HorizontalFlip(p=p_flip),
        A.VerticalFlip(p=p_flip),
        A.RandomRotate90(p=p_rotate),

        # Rotation with interpolation (fills with reflection)
        A.Rotate(
            limit=45,
            interpolation=1,  # INTER_LINEAR
            border_mode=2,    # BORDER_REFLECT
            p=p_rotate * 0.5
        ),

        # Elastic deformation - good for biological tissue
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            p=p_elastic
        ),

        # Grid distortion - another spatial augmentation
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=p_elastic * 0.5
        ),

        # Color/intensity transforms (image only, not mask)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=p_color),

        # Blur - simulates slight defocus
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.1),

        # Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=p_noise),

    ], additional_targets={})


def get_validation_transform() -> Optional[A.Compose]:
    """Create validation transform (no augmentation, just normalization)."""
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    # No transforms for validation - just return as-is
    return None


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        restore_best_weights: Whether to restore best model weights when stopping
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, epoch: int, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            epoch: Current epoch number
            val_loss: Current validation loss
            model: The model being trained

        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0

            if self.restore_best_weights:
                # Save best model state
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            logger.debug(f"Early stopping: new best loss {val_loss:.4f} at epoch {epoch}")
            return False
        else:
            # No improvement
            self.counter += 1
            logger.debug(f"Early stopping: no improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}. "
                           f"Best loss: {self.best_loss:.4f} at epoch {self.best_epoch}")
                return True

        return False

    def restore_best(self, model: nn.Module) -> None:
        """Restore best model weights."""
        if self.best_state is not None and self.restore_best_weights:
            model.load_state_dict(self.best_state)
            logger.info(f"Restored best model weights from epoch {self.best_epoch}")


class SegmentationDataset(Dataset):
    """Dataset for segmentation training with augmentation support."""

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        input_config: Dict[str, Any],
        augment: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.input_config = input_config
        self.augment = augment and ALBUMENTATIONS_AVAILABLE

        # Find all images
        self.image_files = sorted(list(self.images_dir.glob("*.tiff")) +
                                  list(self.images_dir.glob("*.tif")) +
                                  list(self.images_dir.glob("*.png")))
        logger.info(f"Found {len(self.image_files)} images in {images_dir}")

        # Setup augmentation
        if self.augment:
            aug_config = augmentation_config or {}
            self.transform = get_training_augmentation(
                image_size=aug_config.get("image_size", 512),
                p_flip=aug_config.get("p_flip", 0.5),
                p_rotate=aug_config.get("p_rotate", 0.5),
                p_elastic=aug_config.get("p_elastic", 0.3),
                p_color=aug_config.get("p_color", 0.3),
                p_noise=aug_config.get("p_noise", 0.2)
            )
            logger.info("Augmentation enabled")
        else:
            self.transform = None
            if augment and not ALBUMENTATIONS_AVAILABLE:
                logger.warning("Augmentation requested but albumentations not available")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = Image.open(img_path)
        img_array = np.array(img, dtype=np.float32)

        # Handle channels
        if img_array.ndim == 2:
            img_array = img_array[..., np.newaxis]

        # Normalize BEFORE augmentation
        img_array = self._normalize(img_array)

        # Load mask
        mask_name = img_path.stem + ".png"
        mask_path = self.masks_dir / mask_name
        if mask_path.exists():
            mask = Image.open(mask_path)
            mask_array = np.array(mask, dtype=np.int64)
        else:
            # No mask - create empty
            mask_array = np.zeros(img_array.shape[:2], dtype=np.int64)

        # Apply augmentation
        if self.transform is not None:
            # Albumentations expects uint8 or float32 in [0, 1] for image
            # and int for mask
            transformed = self.transform(image=img_array, mask=mask_array)
            img_array = transformed["image"]
            mask_array = transformed["mask"]

        # Convert to tensors (HWC -> CHW for image)
        if img_array.ndim == 2:
            img_array = img_array[..., np.newaxis]
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1).astype(np.float32))
        mask_tensor = torch.from_numpy(mask_array.astype(np.int64))

        return img_tensor, mask_tensor

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image data."""
        norm_config = self.input_config.get("normalization", {})
        strategy = norm_config.get("strategy", "percentile_99")
        per_channel = norm_config.get("per_channel", False)

        if per_channel and img.ndim == 3 and img.shape[2] > 1:
            # Normalize each channel independently
            for c in range(img.shape[2]):
                img[..., c] = self._normalize_single(img[..., c], norm_config, strategy)
        else:
            img = self._normalize_single(img, norm_config, strategy)

        return img

    def _normalize_single(
        self,
        img: np.ndarray,
        norm_config: Dict[str, Any],
        strategy: str
    ) -> np.ndarray:
        """Normalize a single image or channel."""
        if strategy == "percentile_99":
            percentile = norm_config.get("clip_percentile", 99.0)
            p_min = np.percentile(img, 100 - percentile)
            p_max = np.percentile(img, percentile)
            img = np.clip(img, p_min, p_max)
            if p_max > p_min:
                img = (img - p_min) / (p_max - p_min)
        elif strategy == "min_max":
            i_min, i_max = img.min(), img.max()
            if i_max > i_min:
                img = (img - i_min) / (i_max - i_min)
        elif strategy == "z_score":
            mean, std = img.mean(), img.std()
            if std > 0:
                img = (img - mean) / std
                img = np.clip(img, -5, 5)
                # Rescale to 0-1 for augmentation compatibility
                img = (img + 5) / 10

        return img


class TrainingService:
    """Service for training deep learning models.

    Features:
    - Multiple model architectures via segmentation-models-pytorch
    - Data augmentation via albumentations
    - Learning rate scheduling (cosine annealing, step decay, one-cycle)
    - Early stopping with patience
    - Transfer learning with layer freezing
    - Class weighting for imbalanced datasets
    - GPU memory monitoring and management
    - Support for CUDA, Apple MPS, and CPU
    """

    def __init__(
        self,
        device: str = "auto",
        gpu_manager: Optional[GPUManager] = None
    ):
        """Initialize training service.

        Args:
            device: Device to use ("cuda", "mps", "cpu", or "auto")
            gpu_manager: Optional GPUManager instance (uses singleton if not provided)
        """
        self.gpu_manager = gpu_manager or get_gpu_manager()

        if device == "auto":
            self.device = self.gpu_manager.device_type
        else:
            self.device = device

        logger.info(f"TrainingService initialized on device: {self.device}")

    def train(
        self,
        model_type: str,
        architecture: Dict[str, Any],
        input_config: Dict[str, Any],
        training_params: Dict[str, Any],
        classes: List[str],
        data_path: str,
        progress_callback: Optional[Callable[[int, float, float, float], None]] = None,
        cancel_flag: Optional[threading.Event] = None,
        frozen_layers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train a model.

        Args:
            model_type: Type of model architecture (e.g., "unet", "deeplabv3plus")
            architecture: Architecture configuration dict
            input_config: Input configuration (channels, normalization)
            training_params: Training hyperparameters
            classes: List of class names
            data_path: Path to training data
            progress_callback: Optional callback for progress updates
            cancel_flag: Optional threading event for cancellation
            frozen_layers: Optional list of layer names to freeze for transfer learning

        Training params can include:
            - epochs: Number of training epochs (default: 50)
            - batch_size: Batch size (default: 8)
            - learning_rate: Initial learning rate (default: 0.001)
            - weight_decay: L2 regularization (default: 1e-4)
            - augmentation: Enable data augmentation (default: True)
            - scheduler: Learning rate scheduler type ("cosine", "step", "onecycle", "none")
            - scheduler_config: Scheduler-specific parameters
            - early_stopping: Enable early stopping (default: True)
            - early_stopping_patience: Epochs to wait (default: 10)
            - early_stopping_min_delta: Minimum improvement (default: 0.001)
        """
        logger.info(f"Starting training: {model_type}")

        # Create model with optional frozen layers
        if frozen_layers:
            from .pretrained_models import get_pretrained_service
            pretrained_service = get_pretrained_service()

            model = pretrained_service.create_model_with_frozen_layers(
                architecture=model_type,
                encoder=architecture.get("backbone", "resnet34"),
                num_channels=input_config["num_channels"],
                num_classes=len(classes),
                frozen_layers=frozen_layers
            )
            logger.info(f"Created model with {len(frozen_layers)} frozen layer groups")
        else:
            model = self._create_model(
                model_type=model_type,
                architecture=architecture,
                num_channels=input_config["num_channels"],
                num_classes=len(classes)
            )
        model = model.to(self.device)

        # Create datasets
        data_path = Path(data_path)
        augmentation_config = training_params.get("augmentation_config", {})

        train_dataset = SegmentationDataset(
            images_dir=str(data_path / "train" / "images"),
            masks_dir=str(data_path / "train" / "masks"),
            input_config=input_config,
            augment=training_params.get("augmentation", True),
            augmentation_config=augmentation_config
        )

        val_dataset = SegmentationDataset(
            images_dir=str(data_path / "validation" / "images"),
            masks_dir=str(data_path / "validation" / "masks"),
            input_config=input_config,
            augment=False  # Never augment validation
        )

        # Create data loaders
        batch_size = training_params.get("batch_size", 8)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # Setup optimizer - only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)

        logger.info(f"Model parameters: {total_params:,} total, {trainable_count:,} trainable "
                   f"({100*trainable_count/total_params:.1f}%)")

        learning_rate = training_params.get("learning_rate", 0.001)
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=training_params.get("weight_decay", 1e-4)
        )

        # Setup learning rate scheduler
        epochs = training_params.get("epochs", 50)
        scheduler = self._create_scheduler(
            optimizer=optimizer,
            scheduler_type=training_params.get("scheduler", "cosine"),
            scheduler_config=training_params.get("scheduler_config", {}),
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )

        # Setup early stopping
        early_stopping = None
        if training_params.get("early_stopping", True):
            early_stopping = EarlyStopping(
                patience=training_params.get("early_stopping_patience", 10),
                min_delta=training_params.get("early_stopping_min_delta", 0.001),
                restore_best_weights=True
            )
            logger.info(f"Early stopping enabled with patience={early_stopping.patience}")

        # Load class weights and unlabeled index from exported config
        unlabeled_index = 255
        class_weights = None
        config_path = data_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                export_config = json.load(f)
            unlabeled_index = export_config.get("unlabeled_index", 255)
            weights_list = export_config.get("class_weights", None)
            if weights_list:
                class_weights = torch.tensor(weights_list, dtype=torch.float32).to(self.device)
                logger.info(f"Using class weights: {weights_list}")

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=unlabeled_index
        )

        # Training loop
        best_loss = float("inf")
        best_model_state = None
        final_loss = 0.0
        final_accuracy = 0.0
        training_history = []

        for epoch in range(epochs):
            # Check for cancellation
            if cancel_flag and cancel_flag.is_set():
                logger.info("Training cancelled")
                break

            # Clear GPU cache at epoch start to prevent memory accumulation
            self.gpu_manager.clear_cache()

            # Log memory status at start of epoch
            self.gpu_manager.log_memory_status(prefix=f"Epoch {epoch+1}/{epochs} start: ")

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # Train epoch
            model.train()
            train_loss = 0.0

            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Step scheduler if using OneCycleLR (per-batch)
                if isinstance(scheduler, OneCycleLR):
                    scheduler.step()

            train_loss /= max(len(train_loader), 1)

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    _, predicted = outputs.max(1)
                    # Only count labeled pixels for accuracy
                    labeled_mask = masks != unlabeled_index
                    total += labeled_mask.sum().item()
                    correct += ((predicted == masks) & labeled_mask).sum().item()

            val_loss /= max(len(val_loader), 1)
            accuracy = correct / max(total, 1)

            final_loss = val_loss
            final_accuracy = accuracy

            # Record history
            training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "learning_rate": current_lr
            })

            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                       f"val_loss={val_loss:.4f}, acc={accuracy:.4f}, lr={current_lr:.6f}")

            if progress_callback:
                progress_callback(epoch + 1, train_loss, val_loss, accuracy)

            # Step scheduler (for epoch-based schedulers)
            if scheduler is not None and not isinstance(scheduler, OneCycleLR):
                if isinstance(scheduler, CosineAnnealingWarmRestarts):
                    scheduler.step(epoch + 1)
                else:
                    scheduler.step()

            # Save best model checkpoint (independent of early stopping)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.info(f"  New best model at epoch {epoch+1} (loss={val_loss:.4f})")

            # Check early stopping
            if early_stopping is not None:
                if early_stopping(epoch + 1, val_loss, model):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Log final memory status
        self.gpu_manager.log_memory_status(prefix="Training complete: ")

        # Clear cache before restoring weights
        self.gpu_manager.clear_cache()

        # Restore best model weights before saving
        if early_stopping is not None and early_stopping.best_state is not None:
            early_stopping.restore_best(model)
            model = model.to(self.device)
        elif best_model_state is not None:
            model.load_state_dict(best_model_state)
            model = model.to(self.device)
            logger.info(f"Restored best model weights (loss={best_loss:.4f})")

        # Save final model
        model_path = self._save_model(
            model=model,
            model_type=model_type,
            architecture=architecture,
            input_config=input_config,
            classes=classes,
            data_path=str(data_path),
            training_history=training_history
        )

        return {
            "model_path": model_path,
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
            "best_loss": best_loss,
            "epochs_trained": len(training_history),
            "early_stopped": early_stopping.should_stop if early_stopping else False
        }

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        scheduler_config: Dict[str, Any],
        epochs: int,
        steps_per_epoch: int
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create learning rate scheduler.

        Args:
            optimizer: The optimizer
            scheduler_type: Type of scheduler ("cosine", "step", "onecycle", "none")
            scheduler_config: Scheduler-specific configuration
            epochs: Total training epochs
            steps_per_epoch: Number of batches per epoch

        Returns:
            Learning rate scheduler or None
        """
        if scheduler_type == "none" or scheduler_type is None:
            logger.info("No learning rate scheduler")
            return None

        if scheduler_type == "cosine":
            # Cosine annealing with warm restarts
            T_0 = scheduler_config.get("T_0", max(epochs // 3, 1))  # Restart every T_0 epochs
            T_mult = scheduler_config.get("T_mult", 2)  # Double period after each restart
            eta_min = scheduler_config.get("eta_min", 1e-6)  # Minimum learning rate

            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )
            logger.info(f"Using CosineAnnealingWarmRestarts scheduler (T_0={T_0}, T_mult={T_mult})")
            return scheduler

        elif scheduler_type == "step":
            # Step decay
            step_size = scheduler_config.get("step_size", epochs // 3)
            gamma = scheduler_config.get("gamma", 0.1)

            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            logger.info(f"Using StepLR scheduler (step_size={step_size}, gamma={gamma})")
            return scheduler

        elif scheduler_type == "onecycle":
            # One-cycle policy (good for finding optimal LR)
            max_lr = scheduler_config.get("max_lr", optimizer.param_groups[0]["lr"] * 10)
            total_steps = epochs * steps_per_epoch

            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=scheduler_config.get("pct_start", 0.3),
                anneal_strategy=scheduler_config.get("anneal_strategy", "cos"),
                div_factor=scheduler_config.get("div_factor", 25.0),
                final_div_factor=scheduler_config.get("final_div_factor", 1e4)
            )
            logger.info(f"Using OneCycleLR scheduler (max_lr={max_lr})")
            return scheduler

        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, using none")
            return None

    def _create_model(
        self,
        model_type: str,
        architecture: Dict[str, Any],
        num_channels: int,
        num_classes: int
    ):
        """Create a segmentation model."""
        try:
            import segmentation_models_pytorch as smp
            from .pretrained_models import PretrainedModelsService, get_pretrained_service

            encoder_name = architecture.get("backbone", "resnet34")
            encoder_weights = "imagenet" if architecture.get("use_pretrained", True) else None

            # Map model types to smp classes
            model_map = {
                "unet": smp.Unet,
                "unetplusplus": smp.UnetPlusPlus,
                "deeplabv3": smp.DeepLabV3,
                "deeplabv3plus": smp.DeepLabV3Plus,
                "fpn": smp.FPN,
                "pspnet": smp.PSPNet,
                "manet": smp.MAnet,
                "linknet": smp.Linknet,
                "pan": smp.PAN,
            }

            if model_type not in model_map:
                raise ValueError(f"Unknown model type: {model_type}. "
                               f"Available: {list(model_map.keys())}")

            # Check if this is a histology-pretrained encoder
            if encoder_name in PretrainedModelsService.HISTOLOGY_ENCODERS:
                smp_encoder, hub_id = PretrainedModelsService.HISTOLOGY_ENCODERS[encoder_name]

                # Create model with imagenet weights first (correct architecture)
                model = model_map[model_type](
                    encoder_name=smp_encoder,
                    encoder_weights="imagenet",
                    in_channels=num_channels,
                    classes=num_classes
                )

                # Replace encoder weights with histology-pretrained weights
                pretrained_service = get_pretrained_service()
                pretrained_service._load_histology_weights(
                    model, hub_id, smp_encoder, num_channels)

                logger.info(f"Created {model_type} model with histology encoder "
                           f"{encoder_name} (weights: {hub_id})")
                return model

            model = model_map[model_type](
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=num_channels,
                classes=num_classes
            )

            logger.info(f"Created {model_type} model with {encoder_name} encoder "
                       f"(pretrained={encoder_weights is not None})")

            return model

        except ImportError:
            logger.error("segmentation_models_pytorch not installed")
            raise

    def _save_model(
        self,
        model,
        model_type: str,
        architecture: Dict[str, Any],
        input_config: Dict[str, Any],
        classes: List[str],
        data_path: str,
        training_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Save the trained model."""
        import time

        # Create output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_type}_{timestamp}"
        output_dir = Path(os.path.expanduser("~/.dlclassifier/models")) / model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        model_path = output_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Export to ONNX
        try:
            model.eval()
            input_size = architecture.get("input_size", [512, 512])
            num_channels = input_config["num_channels"]
            dummy_input = torch.randn(1, num_channels, input_size[0], input_size[1])
            dummy_input = dummy_input.to(self.device)

            onnx_path = output_dir / "model.onnx"
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                opset_version=14,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 2: "height", 3: "width"}
                }
            )
            logger.info(f"Exported ONNX model to {onnx_path}")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

        # Save metadata
        metadata = {
            "id": model_id,
            "name": f"{model_type.upper()} Classifier",
            "architecture": {
                "type": model_type,
                **architecture
            },
            "input_config": input_config,
            "classes": [{"index": i, "name": c} for i, c in enumerate(classes)]
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save training history if provided
        if training_history:
            history_path = output_dir / "training_history.json"
            with open(history_path, "w") as f:
                json.dump(training_history, f, indent=2)
            logger.info(f"Saved training history ({len(training_history)} epochs)")

        logger.info(f"Model saved to {output_dir}")
        return str(output_dir)
