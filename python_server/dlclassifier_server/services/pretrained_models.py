"""Pretrained models service for listing available architectures and layer structures."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EncoderInfo:
    """Information about an available encoder."""
    name: str
    display_name: str
    family: str
    params_millions: float
    pretrained_weights: List[str]
    license: str
    recommended_for: List[str] = field(default_factory=list)


@dataclass
class ArchitectureInfo:
    """Information about a segmentation architecture."""
    name: str
    display_name: str
    description: str
    decoder_channels: List[int]
    supports_aux_output: bool = False


@dataclass
class LayerInfo:
    """Information about a freezable layer."""
    name: str
    display_name: str
    param_count: int
    is_encoder: bool
    depth: int  # 0 = earliest/most general, higher = deeper/more specific
    recommended_freeze: bool  # Whether to freeze by default for fine-tuning


class PretrainedModelsService:
    """Service for managing pretrained model architectures and encoders."""

    def __init__(self):
        self._encoders = self._init_encoders()
        self._architectures = self._init_architectures()

    def _init_encoders(self) -> Dict[str, EncoderInfo]:
        """Initialize available encoders from segmentation-models-pytorch."""
        # These are the most commonly used and well-tested encoders for histopathology
        return {
            # ResNet family - good general purpose, well-tested
            "resnet18": EncoderInfo(
                name="resnet18", display_name="ResNet-18",
                family="resnet", params_millions=11.7,
                pretrained_weights=["imagenet"],
                license="MIT (torchvision)",
                recommended_for=["small_datasets", "fast_inference"]
            ),
            "resnet34": EncoderInfo(
                name="resnet34", display_name="ResNet-34",
                family="resnet", params_millions=21.8,
                pretrained_weights=["imagenet"],
                license="MIT (torchvision)",
                recommended_for=["general_purpose", "balanced"]
            ),
            "resnet50": EncoderInfo(
                name="resnet50", display_name="ResNet-50",
                family="resnet", params_millions=25.6,
                pretrained_weights=["imagenet"],
                license="MIT (torchvision)",
                recommended_for=["large_datasets", "high_accuracy"]
            ),
            "resnet101": EncoderInfo(
                name="resnet101", display_name="ResNet-101",
                family="resnet", params_millions=44.5,
                pretrained_weights=["imagenet"],
                license="MIT (torchvision)",
                recommended_for=["very_large_datasets"]
            ),

            # EfficientNet family - good accuracy/speed tradeoff
            "efficientnet-b0": EncoderInfo(
                name="efficientnet-b0", display_name="EfficientNet-B0",
                family="efficientnet", params_millions=5.3,
                pretrained_weights=["imagenet"],
                license="Apache 2.0",
                recommended_for=["efficient", "mobile"]
            ),
            "efficientnet-b3": EncoderInfo(
                name="efficientnet-b3", display_name="EfficientNet-B3",
                family="efficientnet", params_millions=12.0,
                pretrained_weights=["imagenet"],
                license="Apache 2.0",
                recommended_for=["balanced", "efficient"]
            ),
            "efficientnet-b4": EncoderInfo(
                name="efficientnet-b4", display_name="EfficientNet-B4",
                family="efficientnet", params_millions=19.0,
                pretrained_weights=["imagenet"],
                license="Apache 2.0",
                recommended_for=["high_accuracy", "efficient"]
            ),

            # SE-ResNet family - attention mechanism
            "se_resnet50": EncoderInfo(
                name="se_resnet50", display_name="SE-ResNet-50",
                family="se_resnet", params_millions=28.1,
                pretrained_weights=["imagenet"],
                license="MIT",
                recommended_for=["attention", "histopathology"]
            ),

            # DenseNet family - feature reuse
            "densenet121": EncoderInfo(
                name="densenet121", display_name="DenseNet-121",
                family="densenet", params_millions=8.0,
                pretrained_weights=["imagenet"],
                license="BSD",
                recommended_for=["feature_reuse", "small_objects"]
            ),
            "densenet169": EncoderInfo(
                name="densenet169", display_name="DenseNet-169",
                family="densenet", params_millions=14.1,
                pretrained_weights=["imagenet"],
                license="BSD",
                recommended_for=["feature_reuse"]
            ),

            # MobileNet - lightweight
            "mobilenet_v2": EncoderInfo(
                name="mobilenet_v2", display_name="MobileNet-V2",
                family="mobilenet", params_millions=3.5,
                pretrained_weights=["imagenet"],
                license="Apache 2.0",
                recommended_for=["mobile", "fast_inference", "low_memory"]
            ),

            # VGG - classic, good for texture
            "vgg16_bn": EncoderInfo(
                name="vgg16_bn", display_name="VGG-16 (BatchNorm)",
                family="vgg", params_millions=138.4,
                pretrained_weights=["imagenet"],
                license="MIT (torchvision)",
                recommended_for=["texture", "histopathology"]
            ),
        }

    def _init_architectures(self) -> Dict[str, ArchitectureInfo]:
        """Initialize available segmentation architectures."""
        return {
            "unet": ArchitectureInfo(
                name="unet", display_name="U-Net",
                description="Classic encoder-decoder with skip connections. Best general-purpose choice.",
                decoder_channels=[256, 128, 64, 32, 16]
            ),
            "unetplusplus": ArchitectureInfo(
                name="unetplusplus", display_name="U-Net++",
                description="Nested U-Net with dense skip connections. Better for small objects.",
                decoder_channels=[256, 128, 64, 32, 16],
                supports_aux_output=True
            ),
            "deeplabv3": ArchitectureInfo(
                name="deeplabv3", display_name="DeepLab V3",
                description="Atrous convolution for multi-scale context. Good for large structures.",
                decoder_channels=[256]
            ),
            "deeplabv3plus": ArchitectureInfo(
                name="deeplabv3plus", display_name="DeepLab V3+",
                description="DeepLab V3 with decoder. Better boundary delineation.",
                decoder_channels=[256, 48]
            ),
            "fpn": ArchitectureInfo(
                name="fpn", display_name="Feature Pyramid Network",
                description="Multi-scale feature pyramid. Good for varying object sizes.",
                decoder_channels=[256, 256, 256, 256]
            ),
            "pspnet": ArchitectureInfo(
                name="pspnet", display_name="PSPNet",
                description="Pyramid pooling module for global context.",
                decoder_channels=[512]
            ),
            "manet": ArchitectureInfo(
                name="manet", display_name="MA-Net",
                description="Multi-scale attention network. Good for complex boundaries.",
                decoder_channels=[256, 128, 64, 32, 16]
            ),
            "linknet": ArchitectureInfo(
                name="linknet", display_name="LinkNet",
                description="Lightweight encoder-decoder. Fast inference.",
                decoder_channels=[256, 128, 64, 32]
            ),
        }

    def list_encoders(self) -> List[Dict[str, Any]]:
        """List available encoders."""
        return [
            {
                "name": e.name,
                "display_name": e.display_name,
                "family": e.family,
                "params_millions": e.params_millions,
                "pretrained_weights": e.pretrained_weights,
                "license": e.license,
                "recommended_for": e.recommended_for
            }
            for e in self._encoders.values()
        ]

    def list_architectures(self) -> List[Dict[str, Any]]:
        """List available segmentation architectures."""
        return [
            {
                "name": a.name,
                "display_name": a.display_name,
                "description": a.description,
                "decoder_channels": a.decoder_channels,
                "supports_aux_output": a.supports_aux_output
            }
            for a in self._architectures.values()
        ]

    def get_model_layers(
        self,
        architecture: str,
        encoder: str,
        num_channels: int = 3,
        num_classes: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get the layer structure of a model for freeze/unfreeze configuration.

        Returns a list of layer groups that can be frozen/unfrozen.
        """
        try:
            import segmentation_models_pytorch as smp
            import torch

            # Create model to inspect structure
            model = self._create_model(architecture, encoder, num_channels, num_classes)

            layers = []

            # Get encoder layers
            encoder_layers = self._get_encoder_layers(model, encoder)
            layers.extend(encoder_layers)

            # Get decoder layers
            decoder_layers = self._get_decoder_layers(model, architecture)
            layers.extend(decoder_layers)

            return layers

        except ImportError:
            logger.error("segmentation_models_pytorch not installed")
            return []
        except Exception as e:
            logger.error(f"Error getting model layers: {e}")
            return []

    def _create_model(
        self,
        architecture: str,
        encoder: str,
        num_channels: int,
        num_classes: int
    ):
        """Create a segmentation model for inspection."""
        import segmentation_models_pytorch as smp

        arch_map = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
            "deeplabv3": smp.DeepLabV3,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "fpn": smp.FPN,
            "pspnet": smp.PSPNet,
            "manet": smp.MAnet,
            "linknet": smp.Linknet,
        }

        if architecture not in arch_map:
            raise ValueError(f"Unknown architecture: {architecture}")

        return arch_map[architecture](
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=num_channels,
            classes=num_classes
        )

    def _get_encoder_layers(self, model, encoder_name: str) -> List[Dict[str, Any]]:
        """Extract encoder layer information."""
        layers = []

        if hasattr(model, 'encoder'):
            encoder = model.encoder

            # Different encoder families have different structures
            encoder_family = self._encoders.get(encoder_name, EncoderInfo(
                name=encoder_name, display_name=encoder_name,
                family="unknown", params_millions=0,
                pretrained_weights=[], license=""
            )).family

            if encoder_family in ["resnet", "se_resnet"]:
                # ResNet-style: conv1 -> layer1 -> layer2 -> layer3 -> layer4
                layer_names = [
                    ("encoder.conv1", "Initial Conv", 0, True),  # Freeze by default
                    ("encoder.layer1", "Block 1 (64 filters)", 1, True),
                    ("encoder.layer2", "Block 2 (128 filters)", 2, True),
                    ("encoder.layer3", "Block 3 (256 filters)", 3, False),
                    ("encoder.layer4", "Block 4 (512 filters)", 4, False),
                ]
            elif encoder_family == "efficientnet":
                # EfficientNet-style
                layer_names = [
                    ("encoder._conv_stem", "Stem Conv", 0, True),
                    ("encoder._blocks[0:4]", "Blocks 0-3", 1, True),
                    ("encoder._blocks[4:10]", "Blocks 4-9", 2, True),
                    ("encoder._blocks[10:18]", "Blocks 10-17", 3, False),
                    ("encoder._blocks[18:]", "Blocks 18+", 4, False),
                ]
            elif encoder_family == "densenet":
                layer_names = [
                    ("encoder.features.conv0", "Initial Conv", 0, True),
                    ("encoder.features.denseblock1", "Dense Block 1", 1, True),
                    ("encoder.features.denseblock2", "Dense Block 2", 2, True),
                    ("encoder.features.denseblock3", "Dense Block 3", 3, False),
                    ("encoder.features.denseblock4", "Dense Block 4", 4, False),
                ]
            elif encoder_family == "vgg":
                layer_names = [
                    ("encoder.features[0:7]", "Layers 1-2 (64 filters)", 0, True),
                    ("encoder.features[7:14]", "Layers 3-4 (128 filters)", 1, True),
                    ("encoder.features[14:24]", "Layers 5-7 (256 filters)", 2, True),
                    ("encoder.features[24:34]", "Layers 8-10 (512 filters)", 3, False),
                    ("encoder.features[34:]", "Layers 11-13 (512 filters)", 4, False),
                ]
            elif encoder_family == "mobilenet":
                layer_names = [
                    ("encoder.features[0:2]", "Initial Conv", 0, True),
                    ("encoder.features[2:5]", "Blocks 1-3", 1, True),
                    ("encoder.features[5:9]", "Blocks 4-7", 2, True),
                    ("encoder.features[9:14]", "Blocks 8-12", 3, False),
                    ("encoder.features[14:]", "Blocks 13+", 4, False),
                ]
            else:
                # Generic fallback
                layer_names = [
                    ("encoder.layer_early", "Early Layers", 0, True),
                    ("encoder.layer_mid", "Middle Layers", 2, False),
                    ("encoder.layer_late", "Late Layers", 4, False),
                ]

            # Count parameters for each layer group
            for name, display_name, depth, freeze_default in layer_names:
                param_count = self._count_params_for_layer(encoder, name)
                layers.append({
                    "name": name,
                    "display_name": f"Encoder: {display_name}",
                    "param_count": param_count,
                    "is_encoder": True,
                    "depth": depth,
                    "recommended_freeze": freeze_default,
                    "description": self._get_layer_description(depth, True)
                })

        return layers

    def _get_decoder_layers(self, model, architecture: str) -> List[Dict[str, Any]]:
        """Extract decoder layer information."""
        layers = []

        if hasattr(model, 'decoder'):
            decoder = model.decoder
            param_count = sum(p.numel() for p in decoder.parameters())

            layers.append({
                "name": "decoder",
                "display_name": "Decoder (all layers)",
                "param_count": param_count,
                "is_encoder": False,
                "depth": 5,
                "recommended_freeze": False,  # Never freeze decoder for fine-tuning
                "description": "Task-specific layers - should always be trained"
            })

        if hasattr(model, 'segmentation_head'):
            head = model.segmentation_head
            param_count = sum(p.numel() for p in head.parameters())

            layers.append({
                "name": "segmentation_head",
                "display_name": "Segmentation Head",
                "param_count": param_count,
                "is_encoder": False,
                "depth": 6,
                "recommended_freeze": False,
                "description": "Final classification layer - must be trained"
            })

        return layers

    def _count_params_for_layer(self, module, layer_name: str) -> int:
        """Count parameters in a layer group."""
        try:
            # Try to get the actual module
            if "[" in layer_name:
                # Handle slice notation like "features[0:7]"
                return 0  # Can't easily count for slices

            parts = layer_name.replace("encoder.", "").split(".")
            current = module
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return 0

            return sum(p.numel() for p in current.parameters())
        except Exception:
            return 0

    def _get_layer_description(self, depth: int, is_encoder: bool) -> str:
        """Get description for a layer based on its depth and what features it captures."""
        if not is_encoder:
            return "Task-specific output layer - always train"

        # Descriptions based on what the layer learns and how well it transfers
        # from ImageNet to histopathology (significant domain shift)
        descriptions = {
            0: "Edges, gradients, basic textures - universal features, transfer well, freeze",
            1: "Low-level patterns (gabor-like filters) - transfer well across domains, freeze",
            2: "Texture combinations, local patterns - partial transfer, consider fine-tuning",
            3: "Mid-level shapes, larger patterns - limited transfer to histopathology, train",
            4: "High-level semantic features - ImageNet concepts don't apply, must retrain",
        }
        return descriptions.get(depth, "Deep features - likely need retraining for histopathology")

    def get_freeze_recommendations(self, dataset_size: str) -> Dict[str, bool]:
        """
        Get recommended freeze settings based on domain adaptation needs.

        The primary factor is what each layer learns and how well it transfers
        from ImageNet to histopathology (significant domain shift):

        - Depth 0-1: Universal low-level features (edges, textures) - always freeze
        - Depth 2: Mid-level patterns - partial transfer, depends on data availability
        - Depth 3-4: High-level semantic features - don't transfer, need retraining

        Dataset size affects HOW MUCH you can safely unfreeze without overfitting,
        but the need for retraining is primarily driven by feature type.

        Args:
            dataset_size: "small" (<500 tiles), "medium" (500-5000), "large" (>5000)
                         This affects risk of overfitting when training more layers

        Returns:
            Dict mapping layer depth to freeze recommendation
        """
        # Base recommendation: freeze universal features, train domain-specific
        # Early layers (0-1): Universal visual features - always freeze
        # Late layers (3-4): Semantic features that don't transfer - always train
        # Middle layers (2): Depends on data available to prevent overfitting

        if dataset_size == "small":
            # Small dataset: risk of overfitting if training too many params
            # Freeze through mid-level, only train highest semantic layers
            return {0: True, 1: True, 2: True, 3: True, 4: False}
        elif dataset_size == "medium":
            # Medium: can afford to train semantic layers
            # Still freeze universal features
            return {0: True, 1: True, 2: True, 3: False, 4: False}
        else:  # large
            # Large dataset: can fine-tune more, including mid-level
            # Still freeze truly universal features (edges, gradients)
            return {0: True, 1: True, 2: False, 3: False, 4: False}

    def create_model_with_frozen_layers(
        self,
        architecture: str,
        encoder: str,
        num_channels: int,
        num_classes: int,
        frozen_layers: List[str]
    ):
        """
        Create a model with specified layers frozen.

        Args:
            architecture: Model architecture name
            encoder: Encoder name
            num_channels: Input channels
            num_classes: Number of output classes
            frozen_layers: List of layer names to freeze

        Returns:
            PyTorch model with frozen layers
        """
        model = self._create_model(architecture, encoder, num_channels, num_classes)

        # Freeze specified layers
        for layer_name in frozen_layers:
            self._freeze_layer(model, layer_name)

        # Log freeze status
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created: {trainable:,}/{total:,} parameters trainable "
                   f"({100*trainable/total:.1f}%)")

        return model

    def _freeze_layer(self, model, layer_name: str):
        """Freeze a specific layer by name."""
        try:
            # Handle encoder prefix
            if layer_name.startswith("encoder."):
                parts = layer_name[8:].split(".")  # Remove "encoder."
                current = model.encoder
            elif layer_name == "decoder":
                for param in model.decoder.parameters():
                    param.requires_grad = False
                return
            elif layer_name == "segmentation_head":
                for param in model.segmentation_head.parameters():
                    param.requires_grad = False
                return
            else:
                parts = layer_name.split(".")
                current = model

            # Navigate to the layer
            for part in parts:
                if "[" in part:
                    # Handle indexed access like "features[0:7]"
                    # This is complex - skip for now
                    continue
                if hasattr(current, part):
                    current = getattr(current, part)

            # Freeze all parameters
            for param in current.parameters():
                param.requires_grad = False

            logger.debug(f"Frozen layer: {layer_name}")

        except Exception as e:
            logger.warning(f"Could not freeze layer {layer_name}: {e}")


    # =========================================================================
    # MULTI-CHANNEL SUPPORT - PLACEHOLDER
    # =========================================================================
    # The following methods are placeholders for future multi-channel
    # fluorescence model support. Currently, the extension focuses on
    # brightfield (RGB) images with ImageNet-pretrained encoders.
    #
    # Future additions may include:
    # - MicroNet pretrained encoders (better for microscopy)
    # - Support for models trained on TissueNet or similar datasets
    # - Channel-agnostic architectures
    # =========================================================================

    def get_multichannel_encoders(self) -> List[Dict[str, Any]]:
        """
        PLACEHOLDER: Get encoders suitable for multi-channel fluorescence images.

        Currently returns the same ImageNet encoders with notes about
        channel handling. Future versions may include microscopy-specific
        pretrained encoders.
        """
        # For now, use same encoders - smp handles channel adaptation
        encoders = self.list_encoders()

        # Add notes about multi-channel handling
        for enc in encoders:
            enc["multichannel_note"] = (
                "ImageNet weights are adapted for N channels by the model. "
                "For >3 channels, encoder weights are repeated. "
                "Fine-tuning is recommended for best results."
            )

        return encoders

    def get_multichannel_recommendations(self) -> Dict[str, Any]:
        """
        PLACEHOLDER: Get recommendations for multi-channel training.
        """
        return {
            "status": "placeholder",
            "message": "Multi-channel support uses ImageNet encoders with channel adaptation",
            "recommendations": [
                "Use smp with in_channels=N for your channel count",
                "Consider per-channel normalization (percentile_99)",
                "Fine-tune all encoder layers for domain adaptation",
                "Larger datasets needed for multi-channel training"
            ],
            "future_plans": [
                "MicroNet pretrained encoders for microscopy",
                "TissueNet-based pretrained models",
                "Specialized fluorescence architectures"
            ]
        }


# Global instance
_pretrained_service: Optional[PretrainedModelsService] = None


def get_pretrained_service() -> PretrainedModelsService:
    """Get or create the pretrained models service."""
    global _pretrained_service
    if _pretrained_service is None:
        _pretrained_service = PretrainedModelsService()
    return _pretrained_service
