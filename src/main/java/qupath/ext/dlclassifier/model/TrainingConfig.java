package qupath.ext.dlclassifier.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Configuration parameters for training a deep learning pixel classifier.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class TrainingConfig {

    // Model architecture
    private final String modelType;
    private final String backbone;

    // Training hyperparameters
    private final int epochs;
    private final int batchSize;
    private final double learningRate;
    private final double weightDecay;

    // Tile parameters
    private final int tileSize;
    private final int overlap;
    private final double downsample;

    // Data configuration
    private final double validationSplit;
    private final Map<String, Boolean> augmentationConfig;

    // Transfer learning
    private final boolean usePretrainedWeights;
    private final int freezeEncoderLayers;
    private final List<String> frozenLayers;

    // Annotation rendering
    private final int lineStrokeWidth;

    // Class weight multipliers (user-supplied multipliers on auto-computed inverse-frequency weights)
    private final Map<String, Double> classWeightMultipliers;

    private TrainingConfig(Builder builder) {
        this.modelType = builder.modelType;
        this.backbone = builder.backbone;
        this.epochs = builder.epochs;
        this.batchSize = builder.batchSize;
        this.learningRate = builder.learningRate;
        this.weightDecay = builder.weightDecay;
        this.tileSize = builder.tileSize;
        this.overlap = builder.overlap;
        this.downsample = builder.downsample;
        this.validationSplit = builder.validationSplit;
        this.augmentationConfig = Collections.unmodifiableMap(new LinkedHashMap<>(builder.augmentationConfig));
        this.usePretrainedWeights = builder.usePretrainedWeights;
        this.freezeEncoderLayers = builder.freezeEncoderLayers;
        this.frozenLayers = Collections.unmodifiableList(new ArrayList<>(builder.frozenLayers));
        this.lineStrokeWidth = builder.lineStrokeWidth;
        this.classWeightMultipliers = Collections.unmodifiableMap(new LinkedHashMap<>(builder.classWeightMultipliers));
    }

    // Getters

    public String getModelType() {
        return modelType;
    }

    public String getBackbone() {
        return backbone;
    }

    public int getEpochs() {
        return epochs;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getWeightDecay() {
        return weightDecay;
    }

    public int getTileSize() {
        return tileSize;
    }

    public int getOverlap() {
        return overlap;
    }

    /**
     * Gets the downsample factor for tile extraction.
     * <p>
     * At downsample 1.0, tiles are extracted at full resolution.
     * At downsample 4.0, each tile covers 4x the spatial area,
     * providing more context for tissue-level classification.
     *
     * @return downsample factor (1.0 = full resolution)
     */
    public double getDownsample() {
        return downsample;
    }

    public double getValidationSplit() {
        return validationSplit;
    }

    /**
     * Gets the augmentation configuration map.
     *
     * @return map of augmentation type to enabled status
     */
    public Map<String, Boolean> getAugmentationConfig() {
        return augmentationConfig;
    }

    /**
     * Checks if any augmentation is enabled.
     *
     * @return true if at least one augmentation is enabled
     */
    public boolean isAugmentation() {
        return augmentationConfig.values().stream().anyMatch(v -> v);
    }

    public boolean isUsePretrainedWeights() {
        return usePretrainedWeights;
    }

    public int getFreezeEncoderLayers() {
        return freezeEncoderLayers;
    }

    /**
     * Gets the list of layer names to freeze during training.
     *
     * @return list of layer names (e.g., "encoder.layer1", "encoder.layer2")
     */
    public List<String> getFrozenLayers() {
        return frozenLayers;
    }

    /**
     * Gets the stroke width for rendering line/polyline annotations as training masks.
     *
     * @return stroke width in pixels
     */
    public int getLineStrokeWidth() {
        return lineStrokeWidth;
    }

    /**
     * Gets the user-supplied class weight multipliers.
     * <p>
     * These multipliers are applied on top of auto-computed inverse-frequency weights.
     * A multiplier of 1.0 means no change; values &gt; 1.0 emphasize a class.
     *
     * @return map of class name to weight multiplier (empty map means no modification)
     */
    public Map<String, Double> getClassWeightMultipliers() {
        return classWeightMultipliers;
    }

    /**
     * Returns the effective tile step size (tileSize - overlap).
     */
    public int getStepSize() {
        return tileSize - overlap;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TrainingConfig that = (TrainingConfig) o;
        return epochs == that.epochs &&
                batchSize == that.batchSize &&
                Double.compare(that.learningRate, learningRate) == 0 &&
                Double.compare(that.weightDecay, weightDecay) == 0 &&
                tileSize == that.tileSize &&
                overlap == that.overlap &&
                Double.compare(that.downsample, downsample) == 0 &&
                Double.compare(that.validationSplit, validationSplit) == 0 &&
                usePretrainedWeights == that.usePretrainedWeights &&
                freezeEncoderLayers == that.freezeEncoderLayers &&
                lineStrokeWidth == that.lineStrokeWidth &&
                Objects.equals(modelType, that.modelType) &&
                Objects.equals(backbone, that.backbone) &&
                Objects.equals(augmentationConfig, that.augmentationConfig) &&
                Objects.equals(frozenLayers, that.frozenLayers) &&
                Objects.equals(classWeightMultipliers, that.classWeightMultipliers);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelType, backbone, epochs, batchSize, learningRate,
                weightDecay, tileSize, overlap, downsample, validationSplit, augmentationConfig,
                usePretrainedWeights, freezeEncoderLayers, frozenLayers, lineStrokeWidth,
                classWeightMultipliers);
    }

    @Override
    public String toString() {
        return String.format("TrainingConfig{model=%s, backbone=%s, epochs=%d, lr=%.6f, tile=%d, downsample=%.1f, lineStroke=%d}",
                modelType, backbone, epochs, learningRate, tileSize, downsample, lineStrokeWidth);
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for TrainingConfig.
     */
    public static class Builder {
        private String modelType = "unet";
        private String backbone = "resnet34";
        private int epochs = 50;
        private int batchSize = 8;
        private double learningRate = 0.001;
        private double weightDecay = 1e-4;
        private int tileSize = 512;
        private int overlap = 64;
        private double downsample = 1.0;
        private double validationSplit = 0.2;
        private Map<String, Boolean> augmentationConfig = new LinkedHashMap<>();
        private boolean usePretrainedWeights = true;
        private int freezeEncoderLayers = 0;
        private List<String> frozenLayers = new ArrayList<>();
        private int lineStrokeWidth = 5;
        private Map<String, Double> classWeightMultipliers = new LinkedHashMap<>();

        public Builder() {
            // Default augmentation configuration
            augmentationConfig.put("flip_horizontal", true);
            augmentationConfig.put("flip_vertical", true);
            augmentationConfig.put("rotation_90", true);
            augmentationConfig.put("color_jitter", false);
            augmentationConfig.put("elastic_deformation", false);
        }

        public Builder modelType(String modelType) {
            this.modelType = modelType;
            return this;
        }

        /**
         * Alias for modelType() for more readable code.
         */
        public Builder classifierType(String classifierType) {
            return modelType(classifierType);
        }

        public Builder backbone(String backbone) {
            this.backbone = backbone;
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder weightDecay(double weightDecay) {
            this.weightDecay = weightDecay;
            return this;
        }

        public Builder tileSize(int tileSize) {
            this.tileSize = tileSize;
            return this;
        }

        public Builder overlap(int overlap) {
            this.overlap = overlap;
            return this;
        }

        /**
         * Sets the downsample factor for tile extraction.
         * <p>
         * Higher downsample = more spatial context per tile but less detail.
         * Recommended: 2-4x for tissue-level classification, 1x for cell-level.
         *
         * @param downsample downsample factor (1.0-32.0)
         */
        public Builder downsample(double downsample) {
            this.downsample = downsample;
            return this;
        }

        public Builder validationSplit(double validationSplit) {
            this.validationSplit = validationSplit;
            return this;
        }

        /**
         * Sets detailed augmentation configuration.
         *
         * @param augmentationConfig map of augmentation type to enabled status
         */
        public Builder augmentation(Map<String, Boolean> augmentationConfig) {
            this.augmentationConfig = new LinkedHashMap<>(augmentationConfig);
            return this;
        }

        /**
         * Enables or disables all augmentation.
         *
         * @param enabled true to enable default augmentation, false to disable all
         */
        public Builder augmentation(boolean enabled) {
            if (enabled) {
                augmentationConfig.put("flip_horizontal", true);
                augmentationConfig.put("flip_vertical", true);
                augmentationConfig.put("rotation_90", true);
            } else {
                augmentationConfig.clear();
            }
            return this;
        }

        public Builder usePretrainedWeights(boolean usePretrainedWeights) {
            this.usePretrainedWeights = usePretrainedWeights;
            return this;
        }

        public Builder freezeEncoderLayers(int freezeEncoderLayers) {
            this.freezeEncoderLayers = freezeEncoderLayers;
            return this;
        }

        /**
         * Sets the list of layer names to freeze during training.
         * This provides fine-grained control over transfer learning.
         *
         * @param frozenLayers list of layer names to freeze
         */
        public Builder frozenLayers(List<String> frozenLayers) {
            this.frozenLayers = new ArrayList<>(frozenLayers);
            return this;
        }

        /**
         * Adds a single layer to freeze during training.
         *
         * @param layerName name of the layer to freeze
         */
        public Builder freezeLayer(String layerName) {
            this.frozenLayers.add(layerName);
            return this;
        }

        /**
         * Sets the stroke width for rendering line/polyline annotations as training masks.
         *
         * @param lineStrokeWidth stroke width in pixels (1-50)
         */
        public Builder lineStrokeWidth(int lineStrokeWidth) {
            this.lineStrokeWidth = lineStrokeWidth;
            return this;
        }

        /**
         * Sets class weight multipliers applied on top of auto-computed inverse-frequency weights.
         *
         * @param classWeightMultipliers map of class name to multiplier (default 1.0)
         */
        public Builder classWeightMultipliers(Map<String, Double> classWeightMultipliers) {
            this.classWeightMultipliers = new LinkedHashMap<>(classWeightMultipliers);
            return this;
        }

        public TrainingConfig build() {
            if (modelType == null || modelType.isEmpty()) {
                throw new IllegalStateException("Model type must be specified");
            }
            if (tileSize < 64 || tileSize > 2048) {
                throw new IllegalStateException("Tile size must be between 64 and 2048");
            }
            if (overlap < 0 || overlap >= tileSize / 2) {
                throw new IllegalStateException("Overlap must be between 0 and half of tile size");
            }
            if (downsample < 1.0 || downsample > 32.0) {
                throw new IllegalStateException("Downsample must be between 1.0 and 32.0");
            }
            if (epochs < 1) {
                throw new IllegalStateException("Epochs must be at least 1");
            }
            if (batchSize < 1) {
                throw new IllegalStateException("Batch size must be at least 1");
            }
            return new TrainingConfig(this);
        }
    }
}
