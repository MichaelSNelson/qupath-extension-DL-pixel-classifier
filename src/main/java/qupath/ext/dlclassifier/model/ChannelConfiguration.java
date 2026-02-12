package qupath.ext.dlclassifier.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Configuration for input channels used in classification.
 * <p>
 * This class specifies which image channels to use, how to normalize them,
 * and other channel-related parameters for training and inference.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ChannelConfiguration {

    /**
     * Normalization strategies for channel data.
     */
    public enum NormalizationStrategy {
        /** Normalize to [0, 1] based on min/max values */
        MIN_MAX,
        /** Clip at 99th percentile, then normalize to [0, 1] */
        PERCENTILE_99,
        /** Z-score normalization: (x - mean) / std */
        Z_SCORE,
        /** User-specified fixed min/max range */
        FIXED_RANGE
    }

    private final List<Integer> selectedChannels;
    private final List<String> channelNames;
    private final int bitDepth;
    private final NormalizationStrategy normalizationStrategy;
    private final boolean perChannelNormalization;
    private final double clipPercentile;
    private final double fixedMin;
    private final double fixedMax;

    private ChannelConfiguration(Builder builder) {
        this.selectedChannels = Collections.unmodifiableList(new ArrayList<>(builder.selectedChannels));
        this.channelNames = Collections.unmodifiableList(new ArrayList<>(builder.channelNames));
        this.bitDepth = builder.bitDepth;
        this.normalizationStrategy = builder.normalizationStrategy;
        this.perChannelNormalization = builder.perChannelNormalization;
        this.clipPercentile = builder.clipPercentile;
        this.fixedMin = builder.fixedMin;
        this.fixedMax = builder.fixedMax;
    }

    // Getters

    public List<Integer> getSelectedChannels() {
        return selectedChannels;
    }

    public List<String> getChannelNames() {
        return channelNames;
    }

    public int getNumChannels() {
        return selectedChannels.size();
    }

    public int getBitDepth() {
        return bitDepth;
    }

    public NormalizationStrategy getNormalizationStrategy() {
        return normalizationStrategy;
    }

    public boolean isPerChannelNormalization() {
        return perChannelNormalization;
    }

    public double getClipPercentile() {
        return clipPercentile;
    }

    public double getFixedMin() {
        return fixedMin;
    }

    public double getFixedMax() {
        return fixedMax;
    }

    /**
     * Returns the maximum possible value for the configured bit depth.
     */
    public int getMaxValue() {
        return (1 << bitDepth) - 1;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ChannelConfiguration that = (ChannelConfiguration) o;
        return bitDepth == that.bitDepth &&
                perChannelNormalization == that.perChannelNormalization &&
                Double.compare(that.clipPercentile, clipPercentile) == 0 &&
                Double.compare(that.fixedMin, fixedMin) == 0 &&
                Double.compare(that.fixedMax, fixedMax) == 0 &&
                Objects.equals(selectedChannels, that.selectedChannels) &&
                Objects.equals(channelNames, that.channelNames) &&
                normalizationStrategy == that.normalizationStrategy;
    }

    @Override
    public int hashCode() {
        return Objects.hash(selectedChannels, channelNames, bitDepth,
                normalizationStrategy, perChannelNormalization,
                clipPercentile, fixedMin, fixedMax);
    }

    @Override
    public String toString() {
        return String.format("ChannelConfiguration{channels=%s, bitDepth=%d, normalization=%s}",
                channelNames, bitDepth, normalizationStrategy);
    }

    /**
     * Creates a default configuration for RGB images.
     */
    public static ChannelConfiguration forRGB() {
        return new Builder()
                .selectedChannels(List.of(0, 1, 2))
                .channelNames(List.of("Red", "Green", "Blue"))
                .bitDepth(8)
                .normalizationStrategy(NormalizationStrategy.MIN_MAX)
                .build();
    }

    /**
     * Creates a builder for constructing ChannelConfiguration instances.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for ChannelConfiguration.
     */
    public static class Builder {
        private List<Integer> selectedChannels = new ArrayList<>();
        private List<String> channelNames = new ArrayList<>();
        private int bitDepth = 8;
        private NormalizationStrategy normalizationStrategy = NormalizationStrategy.PERCENTILE_99;
        private boolean perChannelNormalization = true;
        private double clipPercentile = 99.0;
        private double fixedMin = 0.0;
        private double fixedMax = 255.0;

        public Builder selectedChannels(List<Integer> channels) {
            this.selectedChannels = new ArrayList<>(channels);
            return this;
        }

        public Builder channelNames(List<String> names) {
            this.channelNames = new ArrayList<>(names);
            return this;
        }

        public Builder bitDepth(int bitDepth) {
            this.bitDepth = bitDepth;
            return this;
        }

        public Builder normalizationStrategy(NormalizationStrategy strategy) {
            this.normalizationStrategy = strategy;
            return this;
        }

        public Builder perChannelNormalization(boolean perChannel) {
            this.perChannelNormalization = perChannel;
            return this;
        }

        public Builder clipPercentile(double percentile) {
            this.clipPercentile = percentile;
            return this;
        }

        public Builder fixedRange(double min, double max) {
            this.fixedMin = min;
            this.fixedMax = max;
            return this;
        }

        public ChannelConfiguration build() {
            if (selectedChannels.isEmpty()) {
                throw new IllegalStateException("At least one channel must be selected");
            }
            if (channelNames.isEmpty()) {
                // Generate default names
                for (int i = 0; i < selectedChannels.size(); i++) {
                    channelNames.add("Channel " + selectedChannels.get(i));
                }
            }
            if (channelNames.size() != selectedChannels.size()) {
                throw new IllegalStateException(
                        "Channel names count must match selected channels count");
            }
            return new ChannelConfiguration(this);
        }
    }
}
