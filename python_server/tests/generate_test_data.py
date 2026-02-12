#!/usr/bin/env python3
"""Generate synthetic test data for dlclassifier_server testing.

This script creates a minimal training dataset with synthetic images
containing random shapes. The dataset structure matches what the
training service expects.

Usage:
    python generate_test_data.py [output_dir] [--num-images N]

Output structure:
    output_dir/
        config.json                   # Classes, weights, unlabeled_index
        train/
            images/
                patch_0000.tiff
                ...
            masks/
                patch_0000.png
                ...
        validation/
            images/
            masks/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def generate_synthetic_dataset(
    output_dir: str,
    num_images: int = 6,
    image_size: int = 512,
    train_split: float = 0.67,
    seed: int = 42
) -> None:
    """Generate synthetic test images with shapes.

    Creates images with:
    - Random background texture
    - Random circles (foreground class)
    - Top strip annotated as background
    - Remaining area as unlabeled (255)

    Args:
        output_dir: Directory to save the dataset
        num_images: Total number of images to generate
        image_size: Size of square images (pixels)
        train_split: Fraction of images for training (rest is validation)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    output_path = Path(output_dir)

    # Create directory structure
    for split in ["train", "validation"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "masks").mkdir(parents=True, exist_ok=True)

    num_train = int(num_images * train_split)

    # Track pixel counts for weight calculation
    class_pixels = {0: 0, 1: 0}  # Background, Foreground

    for i in range(num_images):
        # Create image with random textured background
        img = np.random.randint(50, 150, (image_size, image_size, 3), dtype=np.uint8)

        # Add some Gaussian blur effect via averaging
        for c in range(3):
            kernel_size = 3
            from scipy.ndimage import uniform_filter
            try:
                img[:, :, c] = uniform_filter(img[:, :, c], size=kernel_size)
            except ImportError:
                pass  # Skip smoothing if scipy not available

        # Start with unlabeled mask
        mask = np.full((image_size, image_size), 255, dtype=np.uint8)

        # Add random circles (class 1 = foreground)
        num_circles = np.random.randint(3, 8)
        for _ in range(num_circles):
            # Circle center and radius
            margin = 50
            cx = np.random.randint(margin, image_size - margin)
            cy = np.random.randint(margin, image_size - margin)
            r = np.random.randint(20, 60)

            # Create circular mask
            y, x = np.ogrid[:image_size, :image_size]
            circle_mask = (x - cx)**2 + (y - cy)**2 <= r**2

            # Color the circle (reddish tones for foreground)
            color = [
                np.random.randint(180, 220),
                np.random.randint(80, 120),
                np.random.randint(80, 120)
            ]
            img[circle_mask] = color
            mask[circle_mask] = 1

        # Add background annotation region (class 0) - top strip
        bg_height = image_size // 5  # Top 20%
        mask[0:bg_height, :] = 0

        # Ensure background region looks different
        img[0:bg_height, :] = np.clip(
            img[0:bg_height, :].astype(np.int16) - 30,
            0, 255
        ).astype(np.uint8)

        # Count pixels for class weighting
        class_pixels[0] += (mask == 0).sum()
        class_pixels[1] += (mask == 1).sum()

        # Determine split
        split = "train" if i < num_train else "validation"

        # Save image as TIFF
        Image.fromarray(img).save(
            output_path / split / "images" / f"patch_{i:04d}.tiff"
        )

        # Save mask as PNG
        Image.fromarray(mask).save(
            output_path / split / "masks" / f"patch_{i:04d}.png"
        )

        print(f"Generated {split}/patch_{i:04d}")

    # Calculate class weights (inverse frequency)
    total_labeled = class_pixels[0] + class_pixels[1]
    if total_labeled > 0:
        weight_bg = total_labeled / (2.0 * max(class_pixels[0], 1))
        weight_fg = total_labeled / (2.0 * max(class_pixels[1], 1))
        # Normalize so minimum weight is 1.0
        min_weight = min(weight_bg, weight_fg)
        weight_bg /= min_weight
        weight_fg /= min_weight
    else:
        weight_bg, weight_fg = 1.0, 1.0

    # Create config.json
    config = {
        "classes": ["Background", "Foreground"],
        "class_weights": [round(weight_bg, 3), round(weight_fg, 3)],
        "unlabeled_index": 255,
        "pixel_counts": {
            "background": int(class_pixels[0]),
            "foreground": int(class_pixels[1]),
            "unlabeled": int(num_images * image_size * image_size - total_labeled)
        },
        "metadata": {
            "num_images": num_images,
            "image_size": image_size,
            "train_count": num_train,
            "validation_count": num_images - num_train,
            "seed": seed
        }
    }

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDataset generated in {output_path}")
    print(f"  Training images: {num_train}")
    print(f"  Validation images: {num_images - num_train}")
    print(f"  Class weights: {config['class_weights']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test data for DL classifier"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="tests/test_data/synthetic",
        help="Output directory for test data"
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=6,
        help="Number of images to generate (default: 6)"
    )
    parser.add_argument(
        "--image-size", "-s",
        type=int,
        default=256,
        help="Image size in pixels (default: 256 for faster tests)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    generate_synthetic_dataset(
        output_dir=args.output_dir,
        num_images=args.num_images,
        image_size=args.image_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
