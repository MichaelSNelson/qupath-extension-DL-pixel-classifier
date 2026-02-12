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

    // Data configuration
    private final double validationSplit;
    private final Map<String, Boolean> augmentationConfig;

    // Transfer learning
    private final boolean usePretrainedWeights;
    private final int freezeEncoderLayers;
    private final List<String> frozenLayers;

    private TrainingConfig(Builder builder) {
        this.modelType = builder.modelType;
        this.backbone = builder.backbone;
        this.epochs = builder.epochs;
        this.batchSize = builder.batchSize;
        this.learningRate = builder.learningRate;
        this.weightDecay = builder.weightDecay;
        this.tileSize = builder.tileSize;
        this.overlap = builder.overlap;
        this.validationSplit = builder.validationSplit;
        this.augmentationConfig = Collections.unmodifiableMap(new LinkedHashMap<>(builder.augmentationConfig));
        this.usePretrainedWeights = builder.usePretrainedWeights;
        this.freezeEncoderLayers = builder.freezeEncoderLayers;
        this.frozenLayers = Collections.unmodifiableList(new ArrayList<>(builder.frozenLayers));
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
                Double.compare(that.validationSplit, validationSplit) == 0 &&
                usePretrainedWeights == that.usePretrainedWeights &&
                freezeEncoderLayers == that.freezeEncoderLayers &&
                Objects.equals(modelType, that.modelType) &&
                Objects.equals(backbone, that.backbone) &&
                Objects.equals(augmentationConfig, that.augmentationConfig) &&
                Objects.equals(frozenLayers, that.frozenLayers);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelType, backbone, epochs, batchSize, learningRate,
                weightDecay, tileSize, overlap, validationSplit, augmentationConfig,
                usePretrainedWeights, freezeEncoderLayers, frozenLayers);
    }

    @Override
    public String toString() {
        return String.format("TrainingConfig{model=%s, backbone=%s, epochs=%d, lr=%.6f, tile=%d}",
                modelType, backbone, epochs, learningRate, tileSize);
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
        private double validationSplit = 0.2;
        private Map<String, Boolean> augmentationConfig = new LinkedHashMap<>();
        private boolean usePretrainedWeights = true;
        private int freezeEncoderLayers = 0;
        private List<String> frozenLayers = new ArrayList<>();

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
