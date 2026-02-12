package qupath.ext.dlclassifier.utilities;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.Raster;

/**
 * Converts images between different bit depths and data types.
 * <p>
 * This class handles conversion from various image formats (8-bit, 12-bit, 16-bit)
 * to floating-point format suitable for deep learning input.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class BitDepthConverter {

    private static final Logger logger = LoggerFactory.getLogger(BitDepthConverter.class);

    private BitDepthConverter() {
        // Utility class - no instantiation
    }

    /**
     * Converts a BufferedImage to float array format.
     * <p>
     * Output format: [height][width][channels] with raw pixel values.
     *
     * @param image the input image
     * @return float array with shape [height][width][channels]
     */
    public static float[][][] toFloatArray(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        Raster raster = image.getRaster();
        int numBands = raster.getNumBands();

        float[][][] result = new float[height][width][numBands];

        DataBuffer dataBuffer = raster.getDataBuffer();
        int dataType = dataBuffer.getDataType();

        logger.debug("Converting image {}x{} with {} bands, data type={}",
                width, height, numBands, dataType);

        switch (dataType) {
            case DataBuffer.TYPE_BYTE -> extractByte(raster, result, width, height, numBands);
            case DataBuffer.TYPE_USHORT -> extractUShort(raster, result, width, height, numBands);
            case DataBuffer.TYPE_SHORT -> extractShort(raster, result, width, height, numBands);
            case DataBuffer.TYPE_INT -> extractInt(raster, result, width, height, numBands);
            case DataBuffer.TYPE_FLOAT -> extractFloat(raster, result, width, height, numBands);
            case DataBuffer.TYPE_DOUBLE -> extractDouble(raster, result, width, height, numBands);
            default -> throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }

        return result;
    }

    /**
     * Extracts data from byte (8-bit) raster.
     */
    private static void extractByte(Raster raster, float[][][] result,
                                    int width, int height, int numBands) {
        int[] pixel = new int[numBands];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                raster.getPixel(x, y, pixel);
                for (int c = 0; c < numBands; c++) {
                    result[y][x][c] = pixel[c] & 0xFF;
                }
            }
        }
    }

    /**
     * Extracts data from unsigned short (12/16-bit) raster.
     */
    private static void extractUShort(Raster raster, float[][][] result,
                                      int width, int height, int numBands) {
        int[] pixel = new int[numBands];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                raster.getPixel(x, y, pixel);
                for (int c = 0; c < numBands; c++) {
                    result[y][x][c] = pixel[c] & 0xFFFF;
                }
            }
        }
    }

    /**
     * Extracts data from signed short raster.
     */
    private static void extractShort(Raster raster, float[][][] result,
                                     int width, int height, int numBands) {
        int[] pixel = new int[numBands];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                raster.getPixel(x, y, pixel);
                for (int c = 0; c < numBands; c++) {
                    result[y][x][c] = pixel[c];
                }
            }
        }
    }

    /**
     * Extracts data from int raster.
     */
    private static void extractInt(Raster raster, float[][][] result,
                                   int width, int height, int numBands) {
        int[] pixel = new int[numBands];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                raster.getPixel(x, y, pixel);
                for (int c = 0; c < numBands; c++) {
                    result[y][x][c] = pixel[c];
                }
            }
        }
    }

    /**
     * Extracts data from float raster.
     */
    private static void extractFloat(Raster raster, float[][][] result,
                                     int width, int height, int numBands) {
        float[] pixel = new float[numBands];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                raster.getPixel(x, y, pixel);
                System.arraycopy(pixel, 0, result[y][x], 0, numBands);
            }
        }
    }

    /**
     * Extracts data from double raster.
     */
    private static void extractDouble(Raster raster, float[][][] result,
                                      int width, int height, int numBands) {
        double[] pixel = new double[numBands];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                raster.getPixel(x, y, pixel);
                for (int c = 0; c < numBands; c++) {
                    result[y][x][c] = (float) pixel[c];
                }
            }
        }
    }

    /**
     * Detects the effective bit depth from an image.
     *
     * @param image the image to analyze
     * @return estimated bit depth (8, 12, 16, etc.)
     */
    public static int detectBitDepth(BufferedImage image) {
        int dataType = image.getRaster().getDataBuffer().getDataType();

        switch (dataType) {
            case DataBuffer.TYPE_BYTE:
                return 8;
            case DataBuffer.TYPE_USHORT:
                // Could be 12-bit or 16-bit, analyze actual values
                return analyzeUShortBitDepth(image);
            case DataBuffer.TYPE_SHORT:
                return 16;
            case DataBuffer.TYPE_INT:
                return 32;
            case DataBuffer.TYPE_FLOAT:
            case DataBuffer.TYPE_DOUBLE:
                return 32; // Floating point, treated as 32-bit
            default:
                logger.warn("Unknown data type {}, assuming 16-bit", dataType);
                return 16;
        }
    }

    /**
     * Analyzes unsigned short image to determine if it's 12-bit or 16-bit.
     */
    private static int analyzeUShortBitDepth(BufferedImage image) {
        Raster raster = image.getRaster();
        int width = image.getWidth();
        int height = image.getHeight();
        int numBands = raster.getNumBands();

        // Sample pixels to find max value
        int maxValue = 0;
        int sampleStep = Math.max(1, (width * height) / 10000);
        int[] pixel = new int[numBands];

        for (int i = 0; i < width * height; i += sampleStep) {
            int x = i % width;
            int y = i / width;
            raster.getPixel(x, y, pixel);
            for (int c = 0; c < numBands; c++) {
                int val = pixel[c] & 0xFFFF;
                if (val > maxValue) {
                    maxValue = val;
                }
            }
        }

        // Determine bit depth based on max value
        if (maxValue <= 255) return 8;
        if (maxValue <= 4095) return 12;
        return 16;
    }

    /**
     * Extracts selected channels from a multi-channel image.
     *
     * @param data             source data [height][width][channels]
     * @param selectedChannels indices of channels to extract
     * @return extracted data [height][width][selectedChannels.length]
     */
    public static float[][][] extractChannels(float[][][] data, int[] selectedChannels) {
        int height = data.length;
        int width = data[0].length;
        int numSelected = selectedChannels.length;

        float[][][] result = new float[height][width][numSelected];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int i = 0; i < numSelected; i++) {
                    result[y][x][i] = data[y][x][selectedChannels[i]];
                }
            }
        }

        return result;
    }

    /**
     * Converts RGB image to grayscale using luminance formula.
     *
     * @param data RGB data [height][width][3]
     * @return grayscale data [height][width][1]
     */
    public static float[][][] rgbToGrayscale(float[][][] data) {
        int height = data.length;
        int width = data[0].length;

        if (data[0][0].length != 3) {
            throw new IllegalArgumentException("Input must have 3 channels for RGB to grayscale");
        }

        float[][][] result = new float[height][width][1];

        // ITU-R BT.601 luminance formula
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                result[y][x][0] = 0.299f * data[y][x][0] +
                        0.587f * data[y][x][1] +
                        0.114f * data[y][x][2];
            }
        }

        return result;
    }
}
