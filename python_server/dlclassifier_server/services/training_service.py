"""Training service for deep learning models.

Includes:
- Data augmentation via albumentations
- Learning rate scheduling (cosine annealing, one-cycle, step decay)
- Early stopping with configurable metric (loss or mean IoU)
- Combined CE + Dice loss for improved segmentation quality
- Mixed precision training (AMP) for CUDA devices
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, OneCycleLR
from PIL import Image

from .gpu_manager import GPUManager, get_gpu_manager
from ..utils.normalization import normalize as normalize_image
from ..utils.normalization import compute_dataset_stats

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

        # Noise - std_range is fraction of image max value
        A.GaussNoise(std_range=(0.04, 0.2), p=p_noise),

    ], additional_targets={})


def get_validation_transform() -> Optional[A.Compose]:
    """Create validation transform (no augmentation, just normalization)."""
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    # No transforms for validation - just return as-is
    return None


class EarlyStopping:
    """Early stopping to stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        restore_best_weights: Whether to restore best model weights when stopping
        mode: "min" if lower metric is better (e.g. loss),
              "max" if higher metric is better (e.g. IoU)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode

        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0
        self.should_stop = False

        if mode == "min":
            self._is_better = lambda current, best: current < best - self.min_delta
        else:
            self._is_better = lambda current, best: current > best + self.min_delta

    def __call__(self, epoch: int, metric_value: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            epoch: Current epoch number
            metric_value: Current value of the monitored metric
            model: The model being trained

        Returns:
            True if training should stop, False otherwise
        """
        if self._is_better(metric_value, self.best_score):
            # Improvement found
            self.best_score = metric_value
            self.best_epoch = epoch
            self.counter = 0

            if self.restore_best_weights:
                # Save best model state
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            logger.debug(f"Early stopping: new best {metric_value:.4f} at epoch {epoch}")
            return False
        else:
            # No improvement
            self.counter += 1
            logger.debug(f"Early stopping: no improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}. "
                           f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True

        return False

    def restore_best(self, model: nn.Module) -> None:
        """Restore best model weights."""
        if self.best_state is not None and self.restore_best_weights:
            model.load_state_dict(self.best_state)
            logger.info(f"Restored best model weights from epoch {self.best_epoch}")


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation with ignore_index support.

    Computes per-class Dice loss and averages across classes.
    """

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            inputs: Model logits of shape (N, C, H, W)
            targets: Ground truth labels of shape (N, H, W)

        Returns:
            Scalar Dice loss (1 - mean Dice coefficient)
        """
        num_classes = inputs.shape[1]
        probs = F.softmax(inputs, dim=1)

        # Create valid pixel mask
        valid_mask = (targets != self.ignore_index)

        # One-hot encode targets (only valid pixels)
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0
        targets_one_hot = F.one_hot(targets_safe, num_classes).permute(0, 3, 1, 2).float()

        # Zero out invalid pixels
        valid_mask_expanded = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask_expanded
        targets_one_hot = targets_one_hot * valid_mask_expanded

        # Per-class Dice
        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_per_class.mean()


class CombinedCEDiceLoss(nn.Module):
    """Combined Cross-Entropy + Dice loss for segmentation.

    The CE component handles per-pixel classification while the Dice component
    optimizes region overlap directly, making this combination the modern
    standard for segmentation tasks.

    Args:
        class_weights: Optional per-class weights for the CE component
        ignore_index: Label index to ignore (default 255)
        ce_weight: Weight for Cross-Entropy component (default 0.5)
        dice_weight: Weight for Dice component (default 0.5)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class SegmentationDataset(Dataset):
    """Dataset for segmentation training with augmentation support."""

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        input_config: Dict[str, Any],
        augment: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None,
        context_dir: Optional[str] = None
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.input_config = input_config
        self.augment = augment and ALBUMENTATIONS_AVAILABLE
        self.context_dir = Path(context_dir) if context_dir else None

        # Find all images (including raw float32 files from N-channel export)
        self.image_files = sorted(list(self.images_dir.glob("*.tiff")) +
                                  list(self.images_dir.glob("*.tif")) +
                                  list(self.images_dir.glob("*.png")) +
                                  list(self.images_dir.glob("*.raw")))
        logger.info(f"Found {len(self.image_files)} images in {images_dir}")
        if self.context_dir:
            logger.info(f"Multi-scale context enabled from {context_dir}")

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
        # Load detail image (supports TIFF, PNG, and raw float32 files)
        img_path = self.image_files[idx]
        img_array = self._load_patch(img_path)

        # Handle channels
        if img_array.ndim == 2:
            img_array = img_array[..., np.newaxis]

        # Load and concatenate context tile when multi-scale is enabled
        if self.context_dir is not None:
            ctx_path = self.context_dir / img_path.name
            if ctx_path.exists():
                ctx_array = self._load_patch(ctx_path)
                if ctx_array.ndim == 2:
                    ctx_array = ctx_array[..., np.newaxis]
                # Concatenate detail + context along channel axis: (H,W,C) + (H,W,C) -> (H,W,2C)
                img_array = np.concatenate([img_array, ctx_array], axis=2)
            else:
                logger.warning(f"Context tile not found: {ctx_path}, duplicating detail tile")
                img_array = np.concatenate([img_array, img_array], axis=2)

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
        """Normalize image data.

        Delegates to shared normalization module which supports both
        per-tile and precomputed image-level statistics.
        """
        return normalize_image(img, self.input_config)

    @staticmethod
    def _load_patch(img_path: Path) -> np.ndarray:
        """Load a training patch from file.

        Supports:
        - .raw: N-channel float32 (12-byte header: H,W,C as int32 + float32 data)
        - .tif/.tiff: Multi-channel TIFF via tifffile (falls back to PIL)
        - .png and others: Standard formats via PIL
        """
        suffix = img_path.suffix.lower()
        if suffix == '.raw':
            with open(img_path, 'rb') as f:
                header = np.frombuffer(f.read(12), dtype=np.int32)
                h, w, c = int(header[0]), int(header[1]), int(header[2])
                data = np.frombuffer(f.read(), dtype=np.float32)
            return data.reshape(h, w, c).copy()
        if suffix in ('.tif', '.tiff'):
            try:
                import tifffile
                arr = tifffile.imread(str(img_path)).astype(np.float32)
                # tifffile may return (C,H,W) for multi-channel; convert to HWC
                if arr.ndim == 3 and arr.shape[0] < arr.shape[2]:
                    arr = arr.transpose(1, 2, 0)
                return arr
            except ImportError:
                pass  # fall through to PIL
        img = Image.open(img_path)
        return np.array(img, dtype=np.float32)



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
        progress_callback: Optional[Callable] = None,
        cancel_flag: Optional[threading.Event] = None,
        frozen_layers: Optional[List[str]] = None,
        pause_flag: Optional[threading.Event] = None,
        checkpoint_path: Optional[str] = None,
        start_epoch: int = 0
    ) -> Dict[str, Any]:
        """Train a model.

        Args:
            model_type: Type of model architecture (e.g., "unet", "deeplabv3plus")
            architecture: Architecture configuration dict
            input_config: Input configuration (channels, normalization)
            training_params: Training hyperparameters
            classes: List of class names
            data_path: Path to training data
            progress_callback: Optional callback for progress updates.
                Signature: (epoch, train_loss, val_loss, accuracy,
                            per_class_iou, per_class_loss, mean_iou)
            cancel_flag: Optional threading event for cancellation
            frozen_layers: Optional list of layer names to freeze for transfer learning
            pause_flag: Optional threading event for pause requests
            checkpoint_path: Optional path to checkpoint for resuming training
            start_epoch: Epoch to start from when resuming (0-based)

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

        try:
            return self._run_training(
                model_type=model_type,
                architecture=architecture,
                input_config=input_config,
                training_params=training_params,
                classes=classes,
                data_path=data_path,
                progress_callback=progress_callback,
                cancel_flag=cancel_flag,
                frozen_layers=frozen_layers,
                pause_flag=pause_flag,
                checkpoint_path=checkpoint_path,
                start_epoch=start_epoch
            )
        finally:
            # Always free GPU memory, even on crash/error/cancellation
            import gc
            gc.collect()
            self.gpu_manager.clear_cache()
            self.gpu_manager.log_memory_status(prefix="Training cleanup: ")

    def _run_training(
        self,
        model_type: str,
        architecture: Dict[str, Any],
        input_config: Dict[str, Any],
        training_params: Dict[str, Any],
        classes: List[str],
        data_path: str,
        progress_callback: Optional[Callable] = None,
        cancel_flag: Optional[threading.Event] = None,
        frozen_layers: Optional[List[str]] = None,
        pause_flag: Optional[threading.Event] = None,
        checkpoint_path: Optional[str] = None,
        start_epoch: int = 0
    ) -> Dict[str, Any]:
        """Internal training implementation. Called by train() with cleanup guarantee."""

        # Compute effective input channels: doubled when context_scale > 1
        # (detail + context tiles are concatenated along the channel axis)
        context_scale = architecture.get("context_scale", 1)
        base_channels = input_config["num_channels"]
        effective_channels = base_channels * 2 if context_scale > 1 else base_channels
        if effective_channels != base_channels:
            logger.info(f"Context scale {context_scale}: model input channels "
                        f"{base_channels} -> {effective_channels} (detail + context)")

        # Create model with optional frozen layers
        if frozen_layers:
            from .pretrained_models import get_pretrained_service
            pretrained_service = get_pretrained_service()

            model = pretrained_service.create_model_with_frozen_layers(
                architecture=model_type,
                encoder=architecture.get("backbone", "resnet34"),
                num_channels=effective_channels,
                num_classes=len(classes),
                frozen_layers=frozen_layers
            )
            logger.info(f"Created model with {len(frozen_layers)} frozen layer groups")
        else:
            model = self._create_model(
                model_type=model_type,
                architecture=architecture,
                num_channels=effective_channels,
                num_classes=len(classes)
            )
        model = model.to(self.device)

        # Create datasets
        data_path = Path(data_path)
        augmentation_config = training_params.get("augmentation_config", {})

        # Multi-scale context: when context_scale > 1, load context tiles from context/ dirs
        train_context_dir = None
        val_context_dir = None
        if context_scale > 1:
            train_ctx = data_path / "train" / "context"
            val_ctx = data_path / "validation" / "context"
            if train_ctx.exists():
                train_context_dir = str(train_ctx)
                logger.info(f"Multi-scale context enabled (scale={context_scale})")
            else:
                logger.warning(f"context_scale={context_scale} but no context/ directory found")
            if val_ctx.exists():
                val_context_dir = str(val_ctx)

        train_dataset = SegmentationDataset(
            images_dir=str(data_path / "train" / "images"),
            masks_dir=str(data_path / "train" / "masks"),
            input_config=input_config,
            augment=training_params.get("augmentation", True),
            augmentation_config=augmentation_config,
            context_dir=train_context_dir
        )

        val_dataset = SegmentationDataset(
            images_dir=str(data_path / "validation" / "images"),
            masks_dir=str(data_path / "validation" / "masks"),
            input_config=input_config,
            augment=False,  # Never augment validation
            context_dir=val_context_dir
        )

        # Compute dataset-level normalization statistics for consistent inference
        try:
            train_images = []
            for i in range(min(len(train_dataset), 200)):  # Sample up to 200 patches
                img_path = train_dataset.image_files[i]
                img_arr = SegmentationDataset._load_patch(img_path)
                if img_arr.ndim == 2:
                    img_arr = img_arr[..., np.newaxis]
                # Concatenate context tile if multi-scale is enabled
                if train_context_dir:
                    ctx_path = Path(train_context_dir) / img_path.name
                    if ctx_path.exists():
                        ctx_arr = SegmentationDataset._load_patch(ctx_path)
                        if ctx_arr.ndim == 2:
                            ctx_arr = ctx_arr[..., np.newaxis]
                        img_arr = np.concatenate([img_arr, ctx_arr], axis=2)
                train_images.append(img_arr)

            # Use actual channel count (includes context channels if present)
            stats_channels = train_images[0].shape[2] if train_images else input_config["num_channels"]
            dataset_norm_stats = compute_dataset_stats(
                train_images, num_channels=stats_channels
            )
            logger.info(f"Computed dataset normalization stats from "
                        f"{len(train_images)} training patches ({stats_channels} channels)")
        except Exception as e:
            logger.warning(f"Failed to compute dataset normalization stats: {e}")
            dataset_norm_stats = None

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
            scheduler_type=training_params.get("scheduler", "onecycle"),
            scheduler_config=training_params.get("scheduler_config", {}),
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )

        # Setup early stopping
        early_stopping = None
        early_stopping_metric = training_params.get("early_stopping_metric", "mean_iou")
        if training_params.get("early_stopping", True):
            es_mode = "max" if early_stopping_metric == "mean_iou" else "min"
            early_stopping = EarlyStopping(
                patience=training_params.get("early_stopping_patience", 15),
                min_delta=training_params.get("early_stopping_min_delta", 0.001),
                restore_best_weights=True,
                mode=es_mode
            )
            logger.info(f"Early stopping enabled: metric={early_stopping_metric}, "
                       f"mode={es_mode}, patience={early_stopping.patience}")

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

        loss_function = training_params.get("loss_function", "ce_dice")
        if loss_function == "ce_dice":
            criterion = CombinedCEDiceLoss(
                class_weights=class_weights,
                ignore_index=unlabeled_index
            )
            logger.info("Using Combined CE + Dice loss")
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=unlabeled_index
            )
            logger.info("Using CrossEntropy loss")

        # Setup mixed precision training
        use_mixed_precision = (
            training_params.get("mixed_precision", True)
            and self.device == "cuda"
        )
        scaler = torch.amp.GradScaler("cuda") if use_mixed_precision else None
        if use_mixed_precision:
            logger.info("Mixed precision training enabled")

        # Determine best-model tracking mode (same metric as early stopping)
        best_score_mode = "max" if early_stopping_metric == "mean_iou" else "min"
        best_score = float("-inf") if best_score_mode == "max" else float("inf")
        best_model_state = None
        training_history = []

        def _is_best(current, best):
            if best_score_mode == "max":
                return current > best
            return current < best

        # Restore from checkpoint if resuming
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Restore scheduler (recreate OneCycleLR with remaining steps)
            if scheduler is not None:
                if isinstance(scheduler, OneCycleLR):
                    remaining_epochs = epochs - start_epoch
                    remaining_steps = remaining_epochs * len(train_loader)
                    if remaining_steps > 0:
                        scheduler_config = training_params.get("scheduler_config", {})
                        max_lr = scheduler_config.get(
                            "max_lr", optimizer.param_groups[0]["lr"] * 10)
                        scheduler = OneCycleLR(
                            optimizer,
                            max_lr=max_lr,
                            total_steps=remaining_steps,
                            pct_start=scheduler_config.get("pct_start", 0.3),
                            anneal_strategy=scheduler_config.get("anneal_strategy", "cos"),
                            div_factor=scheduler_config.get("div_factor", 25.0),
                            final_div_factor=scheduler_config.get("final_div_factor", 1e4)
                        )
                elif "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Restore early stopping state (handle both old and new format)
            if early_stopping is not None and "early_stopping" in checkpoint:
                es_state = checkpoint["early_stopping"]
                if "best_score" in es_state:
                    early_stopping.best_score = es_state["best_score"]
                elif "best_loss" in es_state:
                    early_stopping.best_score = es_state["best_loss"]
                early_stopping.best_epoch = es_state["best_epoch"]
                early_stopping.counter = es_state["counter"]
                if "best_state" in es_state and es_state["best_state"] is not None:
                    early_stopping.best_state = es_state["best_state"]

            # Restore training history and best model (handle both formats)
            training_history = checkpoint.get("training_history", [])
            if "best_score" in checkpoint:
                best_score = checkpoint["best_score"]
            elif "best_loss" in checkpoint:
                best_score = checkpoint["best_loss"]
            if "best_model_state" in checkpoint:
                best_model_state = checkpoint["best_model_state"]

            logger.info(f"Resumed from checkpoint at epoch {start_epoch}, "
                       f"best_score={best_score:.4f}")

        # Training loop
        num_classes = len(classes)
        best_epoch = 0
        best_loss = 0.0
        best_accuracy = 0.0
        best_mean_iou = 0.0

        for epoch in range(start_epoch, epochs):
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

                if use_mixed_precision:
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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

            # Per-class accumulators
            class_tp = torch.zeros(num_classes, device=self.device)
            class_fp = torch.zeros(num_classes, device=self.device)
            class_fn = torch.zeros(num_classes, device=self.device)
            class_loss_sum = torch.zeros(num_classes, device=self.device)
            class_pixel_count = torch.zeros(num_classes, device=self.device)

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    if use_mixed_precision:
                        with torch.amp.autocast("cuda"):
                            outputs = model(images)
                            loss = criterion(outputs, masks)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    _, predicted = outputs.max(1)
                    # Only count labeled pixels for accuracy
                    labeled_mask = masks != unlabeled_index
                    total += labeled_mask.sum().item()
                    correct += ((predicted == masks) & labeled_mask).sum().item()

                    # Per-class TP/FP/FN
                    for c in range(num_classes):
                        pred_c = (predicted == c) & labeled_mask
                        true_c = (masks == c) & labeled_mask
                        class_tp[c] += (pred_c & true_c).sum()
                        class_fp[c] += (pred_c & ~true_c).sum()
                        class_fn[c] += (~pred_c & true_c).sum()

                    # Per-class loss (unreduced, must include ignore_index
                    # to avoid CUDA assertion on unlabeled pixels)
                    per_pixel_loss = F.cross_entropy(
                        outputs, masks, reduction='none',
                        ignore_index=unlabeled_index)
                    for c in range(num_classes):
                        c_mask = (masks == c) & labeled_mask
                        c_count = c_mask.sum()
                        if c_count > 0:
                            class_loss_sum[c] += per_pixel_loss[c_mask].sum()
                            class_pixel_count[c] += c_count

            val_loss /= max(len(val_loader), 1)
            accuracy = correct / max(total, 1)

            # Compute per-class IoU and loss
            per_class_iou = {}
            per_class_loss = {}
            for c in range(num_classes):
                denom = (class_tp[c] + class_fp[c] + class_fn[c]).item()
                iou = class_tp[c].item() / denom if denom > 0 else 0.0
                per_class_iou[classes[c]] = round(iou, 4)

                px_count = class_pixel_count[c].item()
                c_loss = class_loss_sum[c].item() / px_count if px_count > 0 else 0.0
                per_class_loss[classes[c]] = round(c_loss, 4)

            iou_values = list(per_class_iou.values())
            mean_iou = sum(iou_values) / len(iou_values) if iou_values else 0.0
            mean_iou = round(mean_iou, 4)

            # Record history
            training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "learning_rate": current_lr,
                "per_class_iou": per_class_iou,
                "per_class_loss": per_class_loss,
                "mean_iou": mean_iou
            })

            # Log with per-class breakdown
            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                       f"val_loss={val_loss:.4f}, acc={accuracy:.4f}, "
                       f"mIoU={mean_iou:.4f}, lr={current_lr:.6f}")
            iou_parts = " ".join(f"{k}={v:.3f}" for k, v in per_class_iou.items())
            loss_parts = " ".join(f"{k}={v:.4f}" for k, v in per_class_loss.items())
            logger.info(f"  IoU: {iou_parts}")
            logger.info(f"  Loss: {loss_parts}")

            if progress_callback:
                progress_callback(epoch + 1, train_loss, val_loss, accuracy,
                                  per_class_iou, per_class_loss, mean_iou)

            # Step scheduler (for epoch-based schedulers)
            if scheduler is not None and not isinstance(scheduler, OneCycleLR):
                if isinstance(scheduler, CosineAnnealingWarmRestarts):
                    scheduler.step(epoch + 1)
                else:
                    scheduler.step()

            # Save best model checkpoint (independent of early stopping)
            current_metric = mean_iou if early_stopping_metric == "mean_iou" else val_loss
            if _is_best(current_metric, best_score):
                best_score = current_metric
                best_epoch = epoch + 1
                best_loss = val_loss
                best_accuracy = accuracy
                best_mean_iou = mean_iou
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                metric_name = "mIoU" if early_stopping_metric == "mean_iou" else "loss"
                logger.info(f"  New best model at epoch {epoch+1} ({metric_name}={current_metric:.4f})")

            # Check early stopping
            if early_stopping is not None:
                es_value = mean_iou if early_stopping_metric == "mean_iou" else val_loss
                if early_stopping(epoch + 1, es_value, model):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # Check for pause request
            if pause_flag and pause_flag.is_set():
                logger.info(f"Training paused at epoch {epoch+1}")
                checkpoint_save_path = self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    early_stopping=early_stopping,
                    training_history=training_history,
                    best_score=best_score,
                    best_score_mode=best_score_mode,
                    best_model_state=best_model_state,
                    model_type=model_type,
                    training_config={
                        "model_type": model_type,
                        "architecture": architecture,
                        "input_config": input_config,
                        "training_params": training_params,
                        "classes": classes,
                    }
                )
                # Free GPU memory during pause
                model = model.cpu()
                self.gpu_manager.clear_cache()
                self.gpu_manager.log_memory_status(prefix="Paused (GPU freed): ")

                return {
                    "status": "paused",
                    "checkpoint_path": checkpoint_save_path,
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": accuracy,
                    "per_class_iou": per_class_iou,
                    "per_class_loss": per_class_loss,
                    "mean_iou": mean_iou,
                }

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
            logger.info(f"Restored best model weights (score={best_score:.4f})")

        # Save final model (include dataset normalization stats when available)
        model_path = self._save_model(
            model=model,
            model_type=model_type,
            architecture=architecture,
            input_config=input_config,
            classes=classes,
            data_path=str(data_path),
            training_history=training_history,
            normalization_stats=dataset_norm_stats
        )

        return {
            "model_path": model_path,
            "final_loss": best_loss,
            "final_accuracy": best_accuracy,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_mean_iou": best_mean_iou,
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

    def _save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        early_stopping,
        training_history: List[Dict[str, Any]],
        best_score: float,
        best_score_mode: str,
        best_model_state: Optional[Dict],
        model_type: str,
        training_config: Dict[str, Any]
    ) -> str:
        """Save a training checkpoint for pause/resume.

        Returns:
            Path to the saved checkpoint file.
        """
        import time

        checkpoint_dir = Path(os.path.expanduser("~/.dlclassifier/checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"checkpoint_{model_type}_{timestamp}.pt"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_history": training_history,
            "best_score": best_score,
            "best_score_mode": best_score_mode,
            "training_config": training_config,
        }

        if best_model_state is not None:
            checkpoint["best_model_state"] = best_model_state

        if scheduler is not None and not isinstance(scheduler, OneCycleLR):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if early_stopping is not None:
            checkpoint["early_stopping"] = {
                "best_score": early_stopping.best_score,
                "mode": early_stopping.mode,
                "best_epoch": early_stopping.best_epoch,
                "counter": early_stopping.counter,
                "best_state": early_stopping.best_state,
            }

        torch.save(checkpoint, str(checkpoint_path))
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)

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
        training_history: Optional[List[Dict[str, Any]]] = None,
        normalization_stats: Optional[List[Dict[str, float]]] = None
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
            # Detect actual in_channels from model weights (handles context_scale > 1
            # where model has 2*C channels but input_config.num_channels is C)
            try:
                actual_channels = model.encoder.conv1.weight.shape[1]
            except AttributeError:
                actual_channels = input_config["num_channels"]
            dummy_input = torch.randn(1, actual_channels, input_size[0], input_size[1])
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

        # Read class colors from training config.json if available
        class_colors = {}
        try:
            config_path = Path(data_path) / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    train_config = json.load(f)
                for cls_info in train_config.get("classes", []):
                    if "color" in cls_info and "name" in cls_info:
                        class_colors[cls_info["name"]] = cls_info["color"]
        except Exception as e:
            logger.warning(f"Could not read class colors from config.json: {e}")

        # Build class list with colors
        class_list = []
        for i, c in enumerate(classes):
            entry = {"index": i, "name": c}
            if c in class_colors:
                entry["color"] = class_colors[c]
            class_list.append(entry)

        # Save metadata
        metadata = {
            "id": model_id,
            "name": f"{model_type.upper()} Classifier",
            "architecture": {
                "type": model_type,
                **architecture
            },
            "input_config": input_config,
            "classes": class_list
        }

        # Include dataset normalization stats for consistent inference
        if normalization_stats:
            metadata["normalization_stats"] = normalization_stats
            logger.info(f"Saved normalization stats for {len(normalization_stats)} channels")

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
