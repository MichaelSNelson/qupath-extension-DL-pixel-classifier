"""Shared normalization utilities for inference and training.

This module provides consistent normalization logic used by both
InferenceService and TrainingService (SegmentationDataset).
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def normalize(img: np.ndarray, input_config: Dict[str, Any]) -> np.ndarray:
    """Normalize image data using config-specified strategy.

    Supports precomputed image-level statistics (when available) to ensure
    consistent normalization across tiles. When precomputed stats are absent,
    falls back to per-tile computation.

    Args:
        img: Input image array (HWC or HW)
        input_config: Configuration dict with "normalization" sub-dict

    Returns:
        Normalized image array
    """
    norm_config = input_config.get("normalization", {})
    strategy = norm_config.get("strategy", "percentile_99")
    per_channel = norm_config.get("per_channel", False)
    precomputed = norm_config.get("precomputed", False)
    channel_stats = norm_config.get("channel_stats", None)

    # Use precomputed image-level stats when available
    if precomputed and channel_stats:
        return _normalize_precomputed(img, channel_stats, strategy, per_channel)

    # Fall back to per-tile normalization
    if per_channel and img.ndim == 3 and img.shape[2] > 1:
        for c in range(img.shape[2]):
            img[..., c] = _normalize_single(img[..., c], norm_config, strategy)
    else:
        img = _normalize_single(img, norm_config, strategy)

    return img


def _normalize_precomputed(
    img: np.ndarray,
    channel_stats: List[Dict[str, float]],
    strategy: str,
    per_channel: bool
) -> np.ndarray:
    """Normalize using pre-computed image-level statistics.

    Args:
        img: Input image array (HWC or HW)
        channel_stats: List of per-channel stat dicts with keys:
            p1, p99, min, max, mean, std
        strategy: Normalization strategy name
        per_channel: Whether to normalize each channel independently

    Returns:
        Normalized image array
    """
    if per_channel and img.ndim == 3 and img.shape[2] > 1:
        for c in range(min(img.shape[2], len(channel_stats))):
            stats = channel_stats[c]
            img[..., c] = _normalize_with_stats(img[..., c], stats, strategy)
    else:
        # Use stats from first channel (or compute from tile if no stats)
        stats = channel_stats[0] if channel_stats else {}
        img = _normalize_with_stats(img, stats, strategy)

    return img


def _normalize_with_stats(
    img: np.ndarray,
    stats: Dict[str, float],
    strategy: str
) -> np.ndarray:
    """Normalize a single channel/image using pre-computed statistics.

    Args:
        img: Single-channel image or full image array
        stats: Dict with keys: p1, p99, min, max, mean, std
        strategy: Normalization strategy name

    Returns:
        Normalized array
    """
    if strategy == "percentile_99":
        p_min = stats.get("p1", float(img.min()))
        p_max = stats.get("p99", float(img.max()))
        img = np.clip(img, p_min, p_max)
        if p_max > p_min:
            img = (img - p_min) / (p_max - p_min)

    elif strategy == "min_max":
        i_min = stats.get("min", float(img.min()))
        i_max = stats.get("max", float(img.max()))
        if i_max > i_min:
            img = (img - i_min) / (i_max - i_min)

    elif strategy == "z_score":
        mean = stats.get("mean", float(img.mean()))
        std = stats.get("std", float(img.std()))
        if std > 0:
            img = (img - mean) / std
            img = np.clip(img, -5, 5)
            # Rescale to 0-1 for model compatibility
            img = (img + 5) / 10

    # fixed_range uses global fixed values, not precomputed stats
    elif strategy == "fixed_range":
        fixed_min = stats.get("min", 0)
        fixed_max = stats.get("max", 255)
        img = np.clip(img, fixed_min, fixed_max)
        if fixed_max > fixed_min:
            img = (img - fixed_min) / (fixed_max - fixed_min)

    return img


def _normalize_single(
    img: np.ndarray,
    norm_config: Dict[str, Any],
    strategy: str
) -> np.ndarray:
    """Normalize a single image or channel using per-tile statistics.

    This is the fallback path when precomputed stats are not available.

    Args:
        img: Image or channel array
        norm_config: Normalization configuration dict
        strategy: Normalization strategy name

    Returns:
        Normalized array
    """
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
            # Rescale to 0-1 for model compatibility
            img = (img + 5) / 10

    elif strategy == "fixed_range":
        fixed_min = norm_config.get("min", 0)
        fixed_max = norm_config.get("max", 255)
        img = np.clip(img, fixed_min, fixed_max)
        if fixed_max > fixed_min:
            img = (img - fixed_min) / (fixed_max - fixed_min)

    return img


def compute_dataset_stats(
    images: List[np.ndarray],
    num_channels: int,
    max_samples_per_channel: int = 500_000
) -> List[Dict[str, float]]:
    """Compute normalization statistics across a collection of images.

    Used during training to compute dataset-level stats that can be saved
    in model metadata for consistent inference normalization.

    Args:
        images: List of image arrays (HWC format, float32)
        num_channels: Number of channels to compute stats for
        max_samples_per_channel: Maximum pixel samples per channel
            (reservoir sampling for memory efficiency)

    Returns:
        List of per-channel stat dicts with keys:
            p1, p99, min, max, mean, std
    """
    if not images:
        return []

    # Collect samples using reservoir sampling
    channel_reservoirs = [[] for _ in range(num_channels)]
    total_pixels = 0

    for img in images:
        if img.ndim == 2:
            img = img[..., np.newaxis]
        h, w = img.shape[:2]
        c = min(img.shape[2], num_channels)
        n_pixels = h * w

        # Subsample rate to keep reservoir bounded
        subsample = max(1, (total_pixels + n_pixels) // max_samples_per_channel)

        flat_indices = np.arange(0, n_pixels, subsample)
        for ch in range(c):
            channel_data = img[..., ch].ravel()
            samples = channel_data[flat_indices]
            channel_reservoirs[ch].append(samples)

        total_pixels += n_pixels

    # Compute stats from collected samples
    channel_stats = []
    for ch in range(num_channels):
        if channel_reservoirs[ch]:
            all_samples = np.concatenate(channel_reservoirs[ch])
            # Trim to max_samples if overflow from multiple images
            if len(all_samples) > max_samples_per_channel:
                rng = np.random.default_rng(42)
                indices = rng.choice(
                    len(all_samples), max_samples_per_channel, replace=False
                )
                all_samples = all_samples[indices]

            stats = {
                "p1": float(np.percentile(all_samples, 1)),
                "p99": float(np.percentile(all_samples, 99)),
                "min": float(np.min(all_samples)),
                "max": float(np.max(all_samples)),
                "mean": float(np.mean(all_samples)),
                "std": float(np.std(all_samples)),
            }
        else:
            stats = {"p1": 0.0, "p99": 1.0, "min": 0.0, "max": 1.0,
                     "mean": 0.5, "std": 0.25}
        channel_stats.append(stats)

    return channel_stats
