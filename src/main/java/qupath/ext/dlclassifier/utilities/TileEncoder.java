package qupath.ext.dlclassifier.utilities;

import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Base64;
import java.util.List;

/**
 * Static utility methods for encoding tile images into byte representations
 * suitable for transfer to the Python inference server.
 * <p>
 * Supports three encoding paths:
 * <ul>
 *   <li>Raw RGB uint8 bytes (fast path for simple 8-bit images)</li>
 *   <li>Float32 N-channel bytes (general path for any bit depth)</li>
 *   <li>Base64 PNG (legacy JSON endpoint format)</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public final class TileEncoder {

    private TileEncoder() {
        // Static utility class
    }

    /**
     * Returns true if the image is a simple 8-bit image with 3 or fewer bands,
     * suitable for the fast uint8 encoding path.
     */
    public static boolean isSimpleRgb(BufferedImage image) {
        int numBands = image.getRaster().getNumBands();
        int dataType = image.getRaster().getDataBuffer().getDataType();
        return numBands <= 3 && dataType == DataBuffer.TYPE_BYTE;
    }

    /**
     * Encodes a tile image as raw RGB bytes for binary transfer.
     * <p>
     * Extracts pixel data as a flat byte array in RGB/HWC order.
     * This avoids PNG compression overhead (~0.1ms vs ~5ms per tile).
     *
     * @param image the tile image
     * @return raw pixel bytes in RGB order (H * W * 3)
     */
    public static byte[] encodeTileRaw(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        int numChannels = 3; // Always extract as RGB
        byte[] result = new byte[h * w * numChannels];

        int type = image.getType();
        if (type == BufferedImage.TYPE_3BYTE_BGR) {
            // Fast path: extract BGR bytes and swap to RGB
            byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            for (int i = 0; i < h * w; i++) {
                result[i * 3]     = pixels[i * 3 + 2]; // R
                result[i * 3 + 1] = pixels[i * 3 + 1]; // G
                result[i * 3 + 2] = pixels[i * 3];     // B
            }
        } else if (type == BufferedImage.TYPE_INT_RGB || type == BufferedImage.TYPE_INT_ARGB) {
            // Extract from int[] pixel data
            int[] pixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
            for (int i = 0; i < h * w; i++) {
                int px = pixels[i];
                result[i * 3]     = (byte) ((px >> 16) & 0xFF); // R
                result[i * 3 + 1] = (byte) ((px >> 8) & 0xFF);  // G
                result[i * 3 + 2] = (byte) (px & 0xFF);         // B
            }
        } else {
            // Generic fallback using getRGB
            int[] rgbPixels = image.getRGB(0, 0, w, h, null, 0, w);
            for (int i = 0; i < h * w; i++) {
                int px = rgbPixels[i];
                result[i * 3]     = (byte) ((px >> 16) & 0xFF);
                result[i * 3 + 1] = (byte) ((px >> 8) & 0xFF);
                result[i * 3 + 2] = (byte) (px & 0xFF);
            }
        }

        return result;
    }

    /**
     * Encodes a tile image as float32 bytes for binary transfer of N-channel data.
     * <p>
     * Uses {@link BitDepthConverter#toFloatArray} to handle any bit depth and band count,
     * then writes as little-endian float32 in HWC order. Optionally extracts a subset
     * of channels via {@link BitDepthConverter#extractChannels}.
     *
     * @param image            the tile image (any bit depth, any number of bands)
     * @param selectedChannels channel indices to extract, or null for all channels
     * @return raw pixel bytes as little-endian float32 (H * W * C * 4 bytes)
     */
    public static byte[] encodeTileRawFloat(BufferedImage image, List<Integer> selectedChannels) {
        float[][][] data = BitDepthConverter.toFloatArray(image);
        if (selectedChannels != null && !selectedChannels.isEmpty()) {
            int[] indices = selectedChannels.stream().mapToInt(Integer::intValue).toArray();
            data = BitDepthConverter.extractChannels(data, indices);
        }
        int h = data.length;
        int w = data[0].length;
        int c = data[0][0].length;
        ByteBuffer buf = ByteBuffer.allocate(h * w * c * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int ch = 0; ch < c; ch++) {
                    buf.putFloat(data[y][x][ch]);
                }
            }
        }
        return buf.array();
    }

    /**
     * Interleaves detail and context channel bytes per pixel.
     * <p>
     * Input layout: detail bytes are flat HWC, context bytes are flat HWC.
     * Output layout: for each pixel, [detail_ch0..N, context_ch0..N] (HW-2C order).
     * <p>
     * This is required because numpy reshape interprets a flat array as having
     * contiguous channels per pixel. Sequential concatenation [all_detail | all_context]
     * reshapes incorrectly; per-pixel interleaving produces the correct (H, W, 2C) layout.
     *
     * @param detailBytes      detail tile bytes in HWC order
     * @param contextBytes     context tile bytes in HWC order
     * @param numPixels        total pixels (H * W)
     * @param channelsPerArray number of channels per tile (e.g. 3 for RGB)
     * @param bytesPerChannel  bytes per channel value (1 for uint8, 4 for float32)
     * @return interleaved byte array with 2 * channelsPerArray channels per pixel
     */
    public static byte[] interleaveContextChannels(byte[] detailBytes, byte[] contextBytes,
                                                    int numPixels, int channelsPerArray,
                                                    int bytesPerChannel) {
        int channelStride = channelsPerArray * bytesPerChannel;
        int totalStride = channelStride * 2;
        byte[] result = new byte[numPixels * totalStride];
        for (int px = 0; px < numPixels; px++) {
            System.arraycopy(detailBytes, px * channelStride,
                    result, px * totalStride, channelStride);
            System.arraycopy(contextBytes, px * channelStride,
                    result, px * totalStride + channelStride, channelStride);
        }
        return result;
    }

    /**
     * Encodes a tile image to base64 PNG (legacy format for JSON endpoints).
     *
     * @param image the tile image
     * @return data URI string with base64-encoded PNG
     * @throws IOException if encoding fails
     */
    public static String encodeTileBase64Png(BufferedImage image) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        javax.imageio.ImageIO.write(image, "png", baos);
        return "data:image/png;base64," +
                Base64.getEncoder().encodeToString(baos.toByteArray());
    }
}
