package qupath.ext.dlclassifier.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.lib.classifiers.pixel.PixelClassifier;
import qupath.lib.classifiers.pixel.PixelClassifierMetadata;
import qupath.lib.common.ColorTools;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageChannel;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerMetadata;
import qupath.lib.images.servers.PixelCalibration;
import qupath.lib.images.servers.PixelType;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.RegionRequest;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.IndexColorModel;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Implements QuPath's {@link PixelClassifier} interface to integrate with
 * the native overlay system.
 * <p>
 * This classifier delegates tile inference to the Python DL server via
 * {@link ClassifierClient}. When used with QuPath's
 * {@code PixelClassificationOverlay}, tiles are classified on demand as
 * the user pans and zooms.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class DLPixelClassifier implements PixelClassifier {

    private static final Logger logger = LoggerFactory.getLogger(DLPixelClassifier.class);

    private final ClassifierMetadata metadata;
    private final ChannelConfiguration channelConfig;
    private final InferenceConfig inferenceConfig;
    private final PixelClassifierMetadata pixelMetadata;
    private final IndexColorModel colorModel;

    /**
     * Creates a new DL pixel classifier.
     *
     * @param metadata        classifier metadata from the server
     * @param channelConfig   channel configuration
     * @param inferenceConfig inference configuration
     * @param imageData       image data (used for pixel calibration)
     */
    public DLPixelClassifier(ClassifierMetadata metadata,
                             ChannelConfiguration channelConfig,
                             InferenceConfig inferenceConfig,
                             ImageData<BufferedImage> imageData) {
        this.metadata = metadata;
        this.channelConfig = channelConfig;
        this.inferenceConfig = inferenceConfig;
        this.pixelMetadata = buildPixelMetadata(imageData);
        this.colorModel = buildColorModel();
    }

    @Override
    public boolean supportsImage(ImageData<BufferedImage> imageData) {
        if (imageData == null || imageData.getServer() == null) return false;
        int imageChannels = imageData.getServer().nChannels();
        return imageChannels >= channelConfig.getNumChannels();
    }

    @Override
    public BufferedImage applyClassification(ImageData<BufferedImage> imageData,
                                              RegionRequest request) throws IOException {
        ImageServer<BufferedImage> server = imageData.getServer();

        // Read the tile from the image server
        BufferedImage tileImage = server.readRegion(request);
        if (tileImage == null) {
            throw new IOException("Failed to read tile at " + request);
        }

        // Encode tile as base64 PNG
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(tileImage, "png", baos);
        String encoded = Base64.getEncoder().encodeToString(baos.toByteArray());

        // Create tile data for server
        String tileId = String.format("%d_%d_%d_%d",
                request.getX(), request.getY(), request.getWidth(), request.getHeight());
        List<ClassifierClient.TileData> tiles = List.of(
                new ClassifierClient.TileData(tileId, encoded, request.getX(), request.getY())
        );

        // Send to Python server for classification
        ClassifierClient client = new ClassifierClient(
                DLClassifierPreferences.getServerHost(),
                DLClassifierPreferences.getServerPort()
        );

        // Use pixel inference to get per-pixel probability maps
        Path tempDir = Files.createTempDirectory("dl-overlay-tile-");
        try {
            ClassifierClient.PixelInferenceResult result = client.runPixelInference(
                    metadata.getId(), tiles, channelConfig, inferenceConfig, tempDir);

            if (result == null || result.outputPaths() == null || result.outputPaths().isEmpty()) {
                throw new IOException("No inference result returned for tile");
            }

            String outputPath = result.outputPaths().get(tileId);
            if (outputPath == null) {
                throw new IOException("No output path for tile " + tileId);
            }

            // Read probability map and convert to class index image
            int tileWidth = tileImage.getWidth();
            int tileHeight = tileImage.getHeight();
            float[][][] probMap = ClassifierClient.readProbabilityMap(
                    Path.of(outputPath), result.numClasses(), tileHeight, tileWidth);

            return createClassIndexImage(probMap, tileWidth, tileHeight);

        } finally {
            // Clean up temp directory
            try {
                Files.walk(tempDir)
                        .sorted(java.util.Comparator.reverseOrder())
                        .forEach(path -> {
                            try { Files.deleteIfExists(path); }
                            catch (IOException ignored) {}
                        });
            } catch (IOException e) {
                logger.warn("Failed to clean up temp dir: {}", tempDir, e);
            }
        }
    }

    @Override
    public PixelClassifierMetadata getMetadata() {
        return pixelMetadata;
    }

    /**
     * Creates a TYPE_BYTE_INDEXED image where each pixel value is the argmax
     * class index from the probability map.
     */
    private BufferedImage createClassIndexImage(float[][][] probMap, int width, int height) {
        BufferedImage indexed = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_INDEXED, colorModel);
        var raster = indexed.getRaster();

        int numClasses = probMap[0][0].length;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int maxClass = 0;
                float maxProb = probMap[y][x][0];
                for (int c = 1; c < numClasses; c++) {
                    if (probMap[y][x][c] > maxProb) {
                        maxProb = probMap[y][x][c];
                        maxClass = c;
                    }
                }
                raster.setSample(x, y, 0, maxClass);
            }
        }

        return indexed;
    }

    /**
     * Builds the QuPath PixelClassifierMetadata from our classifier metadata.
     */
    private PixelClassifierMetadata buildPixelMetadata(ImageData<BufferedImage> imageData) {
        PixelCalibration cal = imageData.getServer().getPixelCalibration();

        // Build classification labels map
        Map<Integer, PathClass> labels = new LinkedHashMap<>();
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();
        List<ImageChannel> channels = new ArrayList<>();

        for (ClassifierMetadata.ClassInfo classInfo : classes) {
            int color = parseClassColor(classInfo.color());
            PathClass pathClass = PathClass.fromString(classInfo.name(), color);
            labels.put(classInfo.index(), pathClass);
            channels.add(ImageChannel.getInstance(classInfo.name(), color));
        }

        int tileSize = inferenceConfig.getTileSize();
        int padding = inferenceConfig.getOverlap();

        return new PixelClassifierMetadata.Builder()
                .inputResolution(cal)
                .inputShape(tileSize, tileSize)
                .inputPadding(padding)
                .setChannelType(ImageServerMetadata.ChannelType.CLASSIFICATION)
                .outputPixelType(PixelType.UINT8)
                .classificationLabels(labels)
                .outputChannels(channels)
                .build();
    }

    /**
     * Builds an IndexColorModel for the class indices, used for TYPE_BYTE_INDEXED images.
     */
    private IndexColorModel buildColorModel() {
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();
        int numClasses = Math.max(classes.size(), 2);
        byte[] r = new byte[256];
        byte[] g = new byte[256];
        byte[] b = new byte[256];
        byte[] a = new byte[256];

        for (ClassifierMetadata.ClassInfo classInfo : classes) {
            int idx = classInfo.index();
            if (idx < 0 || idx >= 256) continue;
            int color = parseClassColor(classInfo.color());
            r[idx] = (byte) ColorTools.red(color);
            g[idx] = (byte) ColorTools.green(color);
            b[idx] = (byte) ColorTools.blue(color);
            a[idx] = (byte) 255;
        }

        return new IndexColorModel(8, 256, r, g, b, a);
    }

    /**
     * Parses a hex color string to a packed RGB integer (QuPath format).
     */
    private static int parseClassColor(String colorStr) {
        if (colorStr == null || colorStr.isEmpty()) {
            return ColorTools.packRGB(128, 128, 128);
        }
        try {
            String hex = colorStr.startsWith("#") ? colorStr.substring(1) : colorStr;
            int rgb = Integer.parseInt(hex, 16);
            return ColorTools.packRGB(
                    (rgb >> 16) & 0xFF,
                    (rgb >> 8) & 0xFF,
                    rgb & 0xFF);
        } catch (NumberFormatException e) {
            return ColorTools.packRGB(128, 128, 128);
        }
    }
}
