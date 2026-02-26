package qupath.ext.dlclassifier.service;

import qupath.lib.classifiers.pixel.PixelClassifier;
import qupath.lib.classifiers.pixel.PixelClassifierMetadata;
import qupath.lib.images.ImageData;
import qupath.lib.regions.RegionRequest;

import java.awt.image.BufferedImage;
import java.awt.image.IndexColorModel;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.util.List;

/**
 * A {@link PixelClassifier} that serves pre-computed, blended classification
 * data from memory.
 * <p>
 * Unlike {@link DLPixelClassifier} which classifies tiles on demand,
 * this classifier holds a complete classification map that was produced
 * by the batch inference pipeline with probability blending across tile
 * boundaries. This eliminates the tile boundary artifacts visible in
 * the on-demand overlay.
 * <p>
 * The classifier is immutable and thread-safe after construction.
 * It holds no external resources, so no cleanup is needed.
 *
 * @author UW-LOCI
 * @since 0.2.4
 */
public class PrecomputedPixelClassifier implements PixelClassifier {

    /**
     * A rectangular region of pre-computed classification data.
     *
     * @param classMap  byte[height][width] where each byte is the class index (argmax of blended probabilities)
     * @param offsetX   X offset in full-resolution image coordinates
     * @param offsetY   Y offset in full-resolution image coordinates
     * @param width     width of the classMap (pixels at inference resolution)
     * @param height    height of the classMap (pixels at inference resolution)
     */
    public record ClassifiedRegion(byte[][] classMap,
                                    int offsetX, int offsetY,
                                    int width, int height) {}

    private final List<ClassifiedRegion> regions;
    private final PixelClassifierMetadata metadata;
    private final IndexColorModel colorModel;

    /**
     * Creates a new precomputed pixel classifier.
     *
     * @param regions    classified regions (one per annotation/ROI processed)
     * @param metadata   pixel classifier metadata for QuPath's overlay system
     * @param colorModel color model mapping class indices to display colors
     */
    public PrecomputedPixelClassifier(List<ClassifiedRegion> regions,
                                       PixelClassifierMetadata metadata,
                                       IndexColorModel colorModel) {
        this.regions = List.copyOf(regions);
        this.metadata = metadata;
        this.colorModel = colorModel;
    }

    @Override
    public boolean supportsImage(ImageData<BufferedImage> imageData) {
        return imageData != null && imageData.getServer() != null;
    }

    @Override
    public BufferedImage applyClassification(ImageData<BufferedImage> imageData,
                                              RegionRequest request) throws IOException {
        // Output tile dimensions in downsampled pixel space
        int outW = (int) Math.max(1, Math.round(request.getWidth() / request.getDownsample()));
        int outH = (int) Math.max(1, Math.round(request.getHeight() / request.getDownsample()));

        BufferedImage result = new BufferedImage(outW, outH,
                BufferedImage.TYPE_BYTE_INDEXED, colorModel);
        WritableRaster raster = result.getRaster();

        // For each pre-computed region, check overlap with this tile request
        for (ClassifiedRegion region : regions) {
            // Intersection in full-res coordinates
            int interLeft = Math.max(request.getX(), region.offsetX);
            int interTop = Math.max(request.getY(), region.offsetY);
            int interRight = Math.min(request.getX() + request.getWidth(),
                    region.offsetX + region.width);
            int interBottom = Math.min(request.getY() + request.getHeight(),
                    region.offsetY + region.height);

            if (interLeft >= interRight || interTop >= interBottom) continue;

            // Map intersection to output pixel coordinates and write class indices
            for (int fy = interTop; fy < interBottom; fy++) {
                int outY = (int) Math.round((fy - request.getY()) / request.getDownsample());
                int mapY = fy - region.offsetY;
                if (outY < 0 || outY >= outH || mapY < 0 || mapY >= region.classMap.length) continue;

                for (int fx = interLeft; fx < interRight; fx++) {
                    int outX = (int) Math.round((fx - request.getX()) / request.getDownsample());
                    int mapX = fx - region.offsetX;
                    if (outX < 0 || outX >= outW || mapX < 0 || mapX >= region.classMap[0].length) continue;

                    raster.setSample(outX, outY, 0, region.classMap[mapY][mapX] & 0xFF);
                }
            }
        }

        return result;
    }

    @Override
    public PixelClassifierMetadata getMetadata() {
        return metadata;
    }
}
