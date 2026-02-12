package qupath.ext.dlclassifier.utilities;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ChannelConfiguration.NormalizationStrategy;

import java.util.Arrays;

/**
 * Normalizes image channel data for deep learning input.
 * <p>
 * This class provides per-channel normalization using various strategies
 * suitable for different image types and bit depths.
 *
 * <h3>Normalization Strategies</h3>
 * <ul>
 *   <li><strong>MIN_MAX:</strong> Normalizes to [0, 1] based on min/max values</li>
 *   <li><strong>PERCENTILE_99:</strong> Clips at 99th percentile, then normalizes</li>
 *   <li><strong>Z_SCORE:</strong> Standardizes to zero mean and unit variance</li>
 *   <li><strong>FIXED_RANGE:</strong> Uses user-specified min/max values</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ChannelNormalizer {

    private static final Logger logger = LoggerFactory.getLogger(ChannelNormalizer.class);

    private final NormalizationStrategy strategy;
    private final boolean perChannel;
    private final double clipPercentile;
    private final double fixedMin;
    private final double fixedMax;

    // Computed statistics (populated during normalization)
    private double[] channelMins;
    private double[] channelMaxs;
    private double[] channelMeans;
    private double[] channelStds;

    /**
     * Creates a normalizer from channel configuration.
     *
     * @param config channel configuration
     */
    public ChannelNormalizer(ChannelConfiguration config) {
        this.strategy = config.getNormalizationStrategy();
        this.perChannel = config.isPerChannelNormalization();
        this.clipPercentile = config.getClipPercentile();
        this.fixedMin = config.getFixedMin();
        this.fixedMax = config.getFixedMax();
    }

    /**
     * Creates a normalizer with explicit parameters.
     *
     * @param strategy        normalization strategy
     * @param perChannel      whether to normalize each channel separately
     * @param clipPercentile  percentile for clipping (for PERCENTILE strategy)
     * @param fixedMin        minimum value (for FIXED_RANGE strategy)
     * @param fixedMax        maximum value (for FIXED_RANGE strategy)
     */
    public ChannelNormalizer(NormalizationStrategy strategy, boolean perChannel,
                             double clipPercentile, double fixedMin, double fixedMax) {
        this.strategy = strategy;
        this.perChannel = perChannel;
        this.clipPercentile = clipPercentile;
        this.fixedMin = fixedMin;
        this.fixedMax = fixedMax;
    }

    /**
     * Normalizes multi-channel image data.
     * <p>
     * Input format: [height][width][channels] with values in original bit depth.
     * Output format: [height][width][channels] with values in [0, 1].
     *
     * @param data     input image data
     * @param bitDepth original bit depth (8, 12, 16, etc.)
     * @return normalized data in [0, 1] range
     */
    public float[][][] normalize(float[][][] data, int bitDepth) {
        int height = data.length;
        int width = data[0].length;
        int numChannels = data[0][0].length;

        float[][][] output = new float[height][width][numChannels];

        // Compute statistics if needed
        if (strategy != NormalizationStrategy.FIXED_RANGE) {
            computeStatistics(data, numChannels);
        }

        // Apply normalization
        switch (strategy) {
            case MIN_MAX -> normalizeMinMax(data, output, numChannels);
            case PERCENTILE_99 -> normalizePercentile(data, output, numChannels);
            case Z_SCORE -> normalizeZScore(data, output, numChannels);
            case FIXED_RANGE -> normalizeFixedRange(data, output, numChannels, bitDepth);
        }

        return output;
    }

    /**
     * Normalizes a single channel.
     *
     * @param channelData single channel data
     * @param bitDepth    original bit depth
     * @return normalized channel data in [0, 1] range
     */
    public float[][] normalizeChannel(float[][] channelData, int bitDepth) {
        int height = channelData.length;
        int width = channelData[0].length;

        // Convert to 3D format
        float[][][] data3d = new float[height][width][1];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                data3d[y][x][0] = channelData[y][x];
            }
        }

        // Normalize
        float[][][] normalized = normalize(data3d, bitDepth);

        // Extract back to 2D
        float[][] output = new float[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                output[y][x] = normalized[y][x][0];
            }
        }

        return output;
    }

    /**
     * Computes statistics for each channel.
     */
    private void computeStatistics(float[][][] data, int numChannels) {
        int height = data.length;
        int width = data[0].length;
        int numPixels = height * width;

        channelMins = new double[numChannels];
        channelMaxs = new double[numChannels];
        channelMeans = new double[numChannels];
        channelStds = new double[numChannels];

        Arrays.fill(channelMins, Double.MAX_VALUE);
        Arrays.fill(channelMaxs, Double.MIN_VALUE);

        // First pass: compute min, max, mean
        for (int c = 0; c < numChannels; c++) {
            double sum = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    float val = data[y][x][c];
                    if (val < channelMins[c]) channelMins[c] = val;
                    if (val > channelMaxs[c]) channelMaxs[c] = val;
                    sum += val;
                }
            }
            channelMeans[c] = sum / numPixels;
        }

        // For percentile normalization, compute the percentile value
        if (strategy == NormalizationStrategy.PERCENTILE_99) {
            computePercentileMax(data, numChannels);
        }

        // Second pass: compute std (for z-score)
        if (strategy == NormalizationStrategy.Z_SCORE) {
            for (int c = 0; c < numChannels; c++) {
                double sumSq = 0;
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        double diff = data[y][x][c] - channelMeans[c];
                        sumSq += diff * diff;
                    }
                }
                channelStds[c] = Math.sqrt(sumSq / numPixels);
                // Prevent division by zero
                if (channelStds[c] < 1e-8) {
                    channelStds[c] = 1.0;
                }
            }
        }

        logger.debug("Computed statistics for {} channels", numChannels);
    }

    /**
     * Computes the percentile max value for each channel.
     */
    private void computePercentileMax(float[][][] data, int numChannels) {
        int height = data.length;
        int width = data[0].length;
        int numPixels = height * width;

        // Sample for percentile calculation (to avoid memory issues with large images)
        int sampleSize = Math.min(numPixels, 100000);
        float sampleStep = (float) numPixels / sampleSize;

        for (int c = 0; c < numChannels; c++) {
            float[] samples = new float[sampleSize];
            int sampleIdx = 0;
            float pos = 0;

            for (int y = 0; y < height && sampleIdx < sampleSize; y++) {
                for (int x = 0; x < width && sampleIdx < sampleSize; x++) {
                    if ((int) pos == y * width + x) {
                        samples[sampleIdx++] = data[y][x][c];
                        pos += sampleStep;
                    }
                }
            }

            // Sort and get percentile
            Arrays.sort(samples, 0, sampleIdx);
            int percentileIdx = (int) (sampleIdx * clipPercentile / 100.0);
            percentileIdx = Math.max(0, Math.min(percentileIdx, sampleIdx - 1));
            channelMaxs[c] = samples[percentileIdx];

            // Ensure max > min
            if (channelMaxs[c] <= channelMins[c]) {
                channelMaxs[c] = channelMins[c] + 1;
            }
        }
    }

    /**
     * Min-max normalization to [0, 1].
     */
    private void normalizeMinMax(float[][][] data, float[][][] output, int numChannels) {
        int height = data.length;
        int width = data[0].length;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < numChannels; c++) {
                    int channelIdx = perChannel ? c : 0;
                    double range = channelMaxs[channelIdx] - channelMins[channelIdx];
                    if (range < 1e-8) range = 1;

                    float normalized = (float) ((data[y][x][c] - channelMins[channelIdx]) / range);
                    output[y][x][c] = Math.max(0, Math.min(1, normalized));
                }
            }
        }
    }

    /**
     * Percentile normalization with clipping.
     */
    private void normalizePercentile(float[][][] data, float[][][] output, int numChannels) {
        int height = data.length;
        int width = data[0].length;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < numChannels; c++) {
                    int channelIdx = perChannel ? c : 0;
                    double range = channelMaxs[channelIdx] - channelMins[channelIdx];
                    if (range < 1e-8) range = 1;

                    // Clip to percentile range
                    float clipped = Math.max((float) channelMins[channelIdx],
                            Math.min((float) channelMaxs[channelIdx], data[y][x][c]));
                    float normalized = (float) ((clipped - channelMins[channelIdx]) / range);
                    output[y][x][c] = Math.max(0, Math.min(1, normalized));
                }
            }
        }
    }

    /**
     * Z-score normalization (standardization).
     */
    private void normalizeZScore(float[][][] data, float[][][] output, int numChannels) {
        int height = data.length;
        int width = data[0].length;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < numChannels; c++) {
                    int channelIdx = perChannel ? c : 0;
                    float normalized = (float) ((data[y][x][c] - channelMeans[channelIdx])
                            / channelStds[channelIdx]);
                    // Z-score output is unbounded, clamp to reasonable range
                    output[y][x][c] = Math.max(-5, Math.min(5, normalized));
                }
            }
        }
    }

    /**
     * Fixed range normalization based on bit depth.
     */
    private void normalizeFixedRange(float[][][] data, float[][][] output,
                                     int numChannels, int bitDepth) {
        int height = data.length;
        int width = data[0].length;

        // Use fixed range or bit depth max
        double minVal = fixedMin;
        double maxVal = fixedMax;
        if (maxVal <= minVal) {
            maxVal = (1 << bitDepth) - 1;
        }
        double range = maxVal - minVal;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < numChannels; c++) {
                    float normalized = (float) ((data[y][x][c] - minVal) / range);
                    output[y][x][c] = Math.max(0, Math.min(1, normalized));
                }
            }
        }
    }

    // ==================== Getters ====================

    public NormalizationStrategy getStrategy() {
        return strategy;
    }

    public boolean isPerChannel() {
        return perChannel;
    }

    public double getClipPercentile() {
        return clipPercentile;
    }

    public double[] getChannelMins() {
        return channelMins;
    }

    public double[] getChannelMaxs() {
        return channelMaxs;
    }

    public double[] getChannelMeans() {
        return channelMeans;
    }

    public double[] getChannelStds() {
        return channelStds;
    }
}
