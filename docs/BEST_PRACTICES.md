# Best Practices

Guidance for backbone selection, annotation strategy, hyperparameter tuning, and improving classifier results.

## Backbone Selection

### By dataset size

| Dataset size | Recommended backbone | Freeze strategy |
|-------------|---------------------|-----------------|
| <200 tiles | resnet34 or efficientnet-b0 | Freeze all encoder |
| 200-1000 tiles | resnet34 | Freeze early layers |
| 1000-5000 tiles | resnet34 or resnet50 | Freeze first 1-2 blocks |
| >5000 tiles | resnet50 or histology backbone | Unfreeze most/all |

### By image type

| Image type | Recommended backbone | Notes |
|-----------|---------------------|-------|
| H&E brightfield | Histology backbone (if available) | Tissue-pretrained features transfer directly |
| H&E brightfield | resnet34 (ImageNet) | Good alternative, widely tested |
| Fluorescence | resnet34 | ImageNet features transfer reasonably well |
| Multi-channel (>3) | resnet34 | First conv layer adapts to channel count |

### By computational resources

| VRAM | Recommended | Max tile size at batch=8 |
|------|-------------|--------------------------|
| 4 GB | efficientnet-b0 | 256px |
| 8 GB | resnet34 | 512px |
| 12+ GB | resnet50 | 512-1024px |
| 24+ GB | resnet50 | 1024px |

## Annotation Strategy

### Quality principles

1. **Annotate boundaries carefully**: The model learns most from class transition zones
2. **Cover variability**: Include examples from different staining intensities, tissue regions, and preparation qualities
3. **Be consistent**: Apply the same classification criteria throughout
4. **Include "hard" cases**: Annotate areas where classification is ambiguous -- these are most informative
5. **Use multiple images**: Multi-image training produces more robust classifiers

### Minimum annotation requirements

| Task complexity | Minimum annotations per class | Recommended |
|----------------|------------------------------|-------------|
| 2-class (foreground/background) | 10 brush strokes | 30+ brush strokes |
| 3-4 class | 15 annotations per class | 40+ per class |
| 5+ classes | 20 annotations per class | 50+ per class |

### Common mistakes

- **Only annotating "easy" regions**: The model needs hard examples to learn boundaries
- **Unbalanced annotations**: One class has 10x more area than another -- use weight multipliers to compensate
- **Inconsistent labeling**: Different annotators applying different criteria to the same tissue
- **Annotating only one image**: Single-image classifiers often fail on new images

## Hyperparameter Tuning

### First training run

Use these conservative settings for your first attempt:

```
Architecture: UNet
Backbone: resnet34
Epochs: 50 (early stopping will handle the rest)
Batch Size: 8
Learning Rate: 0.001
Tile Size: 512
Pretrained: Yes
LR Scheduler: One Cycle
Loss: CE + Dice
```

### If results are poor

#### Model underfits (low accuracy on both train and validation)

- Increase epochs
- Increase model capacity (resnet50 instead of resnet34)
- Reduce tile overlap if generating too many similar patches
- Unfreeze more encoder layers
- Check annotation quality

#### Model overfits (high train accuracy, low validation accuracy)

- Freeze more encoder layers
- Reduce learning rate
- Enable more augmentation (color jitter, elastic deformation)
- Add more training data (more annotations, more images)
- Reduce model capacity (resnet34 instead of resnet50)
- Increase validation split to detect overfitting earlier

#### Training is unstable (loss oscillates)

- Reduce learning rate (try 1e-4 or 1e-5)
- Switch to One Cycle scheduler
- Reduce batch size
- Check for annotation errors (mislabeled regions)

#### Specific classes perform poorly

- Increase weight multiplier for that class
- Add more annotations for that class, especially at boundaries
- Check if the class has consistent visual features
- Consider merging visually similar classes

### Advanced tuning

| Scenario | Adjustment |
|----------|-----------|
| Very small dataset (<200 tiles) | efficientnet-b0, freeze all encoder, augmentation on, epochs=200, batch=4 |
| Large dataset (>10000 tiles) | resnet50, unfreeze all, epochs=50, batch=16, lr=5e-4 |
| Multi-scale features needed | Try a larger backbone (resnet50) or import a custom ONNX model |
| Staining variation between slides | Enable color jitter augmentation |
| Tissue distortion artifacts | Enable elastic deformation augmentation |

## Monitoring Training

### What to watch in the loss chart

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both losses decrease smoothly | Good training | Continue |
| Train loss decreases, val loss increases | Overfitting | More data, more augmentation, more freezing |
| Both losses plateau early | Underfitting | More capacity, unfreeze layers, check data |
| Loss oscillates wildly | LR too high | Reduce learning rate |
| Loss barely decreases | LR too low or data issue | Increase LR, check annotations |

### When to stop manually

Early stopping handles this automatically, but you can also:
- Cancel training if validation loss has been increasing for 10+ epochs
- Cancel if the loss chart shows clear divergence between train and val

## Normalization Strategy

The extension supports four normalization strategies. The choice affects how pixel intensities are scaled before the model sees them.

| Strategy | Best for | Notes |
|----------|----------|-------|
| **PERCENTILE_99** (default) | Most images | Clips to 1st/99th percentile, robust to outliers |
| **MIN_MAX** | Uniform-intensity images | Uses full dynamic range, sensitive to outliers |
| **Z_SCORE** | Images with consistent intensity distributions | Mean/std normalization, good for fluorescence |
| **FIXED_RANGE** | When you know the exact intensity range | Specify min/max values explicitly (e.g., 0-4095 for 12-bit) |

**Image-level normalization** (enabled by default) computes statistics once across the entire image, then applies them consistently to every tile. This eliminates visible tile boundary artifacts that occur when each tile independently computes its own statistics. Newly trained models also save training dataset statistics in their metadata for even better consistency across different images.

## Improving Results

### Quick wins

1. **Add more annotations** at class boundaries
2. **Use multi-image training** if available
3. **Enable augmentation** (especially flips and rotation)
4. **Use a histology backbone** for H&E images
5. **Increase epochs** with early stopping (it is safe to overshoot)
6. **Re-train models** to save normalization statistics -- new models automatically store training dataset stats for improved inference consistency

### Medium effort

1. **Adjust class weights** for imbalanced datasets
2. **Try transfer learning presets** (small/medium/large)
3. **Experiment with tile size** (256 vs 512)
4. **Try different downsample levels** for tissue-level features

### High effort

1. **Annotate more images** for diversity
2. **Try a custom ONNX model** trained externally with a different architecture
3. **Manual layer freeze tuning** for your specific dataset
4. **Cross-validation** using multiple train/test splits
