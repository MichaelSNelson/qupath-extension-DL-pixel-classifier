package qupath.ext.dlclassifier.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.controller.InferenceWorkflow;
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

import javafx.application.Platform;

import java.awt.image.BufferedImage;
import java.awt.image.IndexColorModel;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

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
    private final double downsample;
    private final PixelClassifierMetadata pixelMetadata;
    private final IndexColorModel colorModel;
    private final ClassifierClient client;
    private final Path sharedTempDir;
    private final String modelDirPath;

    /** Colors resolved from PathClass cache, keyed by class index. Used by buildColorModel(). */
    private final Map<Integer, Integer> resolvedClassColors = new LinkedHashMap<>();

    /** Circuit breaker: stops retrying server requests after persistent failures. */
    private static final int MAX_CONSECUTIVE_ERRORS = 3;
    private final AtomicInteger consecutiveErrors = new AtomicInteger(0);
    private final AtomicBoolean errorNotified = new AtomicBoolean(false);
    private volatile String lastErrorMessage;

    /** Set to true when the overlay is being removed, to suppress error counting on interrupted threads. */
    private volatile boolean shuttingDown = false;

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
        this.downsample = metadata.getDownsample();
        this.pixelMetadata = buildPixelMetadata(imageData);
        this.colorModel = buildColorModel();
        this.client = new ClassifierClient(
                DLClassifierPreferences.getServerHost(),
                DLClassifierPreferences.getServerPort());

        // Resolve classifier ID to filesystem path for the Python server
        ModelManager modelManager = new ModelManager();
        this.modelDirPath = modelManager.getModelPath(metadata.getId())
                .map(p -> p.getParent().toString())
                .orElse(metadata.getId());

        try {
            this.sharedTempDir = Files.createTempDirectory("dl-overlay-");
        } catch (IOException e) {
            throw new RuntimeException("Failed to create temp directory for overlay", e);
        }
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
        // If shutting down (overlay being removed), bail out immediately
        if (shuttingDown || Thread.currentThread().isInterrupted()) {
            throw new IOException("Classifier is shutting down");
        }

        // Circuit breaker: stop retrying after persistent server errors
        if (consecutiveErrors.get() >= MAX_CONSECUTIVE_ERRORS) {
            throw new IOException("Classification disabled after " + MAX_CONSECUTIVE_ERRORS +
                    " consecutive server errors: " + lastErrorMessage);
        }

        ImageServer<BufferedImage> server = imageData.getServer();

        // Read the tile from the image server
        BufferedImage tileImage = server.readRegion(request);
        if (tileImage == null) {
            throw new IOException("Failed to read tile at " + request);
        }

        // Encode tile as raw binary (uint8 fast path for simple RGB, float32 for N-channel)
        String dtype;
        byte[] rawBytes;
        int numChannels;
        if (InferenceWorkflow.isSimpleRgb(tileImage)) {
            dtype = "uint8";
            rawBytes = InferenceWorkflow.encodeTileRaw(tileImage);
            numChannels = 3;
        } else {
            dtype = "float32";
            rawBytes = InferenceWorkflow.encodeTileRawFloat(tileImage,
                    channelConfig.getSelectedChannels());
            numChannels = channelConfig.getSelectedChannels().isEmpty()
                    ? tileImage.getRaster().getNumBands()
                    : channelConfig.getSelectedChannels().size();
        }

        String tileId = String.format("%d_%d_%d_%d",
                request.getX(), request.getY(), request.getWidth(), request.getHeight());

        try {
            // Use binary pixel inference (single-tile batch)
            int reflectionPadding = DLClassifierPreferences.getOverlayReflectionPadding();
            ClassifierClient.PixelInferenceResult result = client.runPixelInferenceBinary(
                    modelDirPath, rawBytes, List.of(tileId),
                    tileImage.getHeight(), tileImage.getWidth(), numChannels,
                    dtype, channelConfig, inferenceConfig, sharedTempDir,
                    reflectionPadding);

            // Fall back to JSON/PNG path if binary endpoint unavailable
            if (result == null) {
                java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
                javax.imageio.ImageIO.write(tileImage, "png", baos);
                String encoded = "data:image/png;base64," +
                        java.util.Base64.getEncoder().encodeToString(baos.toByteArray());
                List<ClassifierClient.TileData> tiles = List.of(
                        new ClassifierClient.TileData(tileId, encoded,
                                request.getX(), request.getY()));
                result = client.runPixelInference(
                        modelDirPath, tiles, channelConfig, inferenceConfig,
                        sharedTempDir, reflectionPadding);
            }

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

            // Clean up this tile's prob map file (shared dir persists)
            try {
                Files.deleteIfExists(Path.of(outputPath));
            } catch (IOException e) {
                logger.debug("Failed to delete tile output: {}", outputPath);
            }

            // Success -- reset error counter
            consecutiveErrors.set(0);
            return createClassIndexImage(probMap, tileWidth, tileHeight);

        } catch (IOException e) {
            // During shutdown, interrupted threads and missing temp files are expected - don't count as errors
            if (shuttingDown || Thread.currentThread().isInterrupted()
                    || e instanceof java.io.InterruptedIOException) {
                logger.debug("Classification interrupted during shutdown");
                throw new IOException("Classification interrupted during shutdown", e);
            }

            int errorCount = consecutiveErrors.incrementAndGet();
            lastErrorMessage = e.getMessage();

            if (errorCount >= MAX_CONSECUTIVE_ERRORS && errorNotified.compareAndSet(false, true)) {
                logger.error("Classification overlay disabled after {} consecutive errors: {}",
                        errorCount, e.getMessage());
                Platform.runLater(() -> {
                    var alert = new javafx.scene.control.Alert(
                            javafx.scene.control.Alert.AlertType.ERROR);
                    alert.setTitle("Classification Error");
                    alert.setHeaderText("Classification overlay has been disabled");
                    alert.setContentText("The server returned repeated errors:\n" +
                            lastErrorMessage + "\n\n" +
                            "Remove the overlay and check the server connection.");
                    alert.show();
                });
            }
            throw e;
        }
    }

    /**
     * Signals that this classifier is shutting down. In-flight tile requests
     * will detect this and exit without counting errors or showing dialogs.
     * Called by {@link OverlayService} before stopping the overlay.
     */
    public void shutdown() {
        shuttingDown = true;
    }

    /**
     * Cleans up resources used by this classifier (shared temp directory).
     * Called by {@link OverlayService} when the overlay is removed.
     */
    public void cleanup() {
        shuttingDown = true;
        try {
            if (sharedTempDir != null && Files.exists(sharedTempDir)) {
                Files.walk(sharedTempDir)
                        .sorted(Comparator.reverseOrder())
                        .forEach(path -> {
                            try { Files.deleteIfExists(path); }
                            catch (IOException ignored) {}
                        });
                logger.debug("Cleaned up shared temp dir: {}", sharedTempDir);
            }
        } catch (IOException e) {
            logger.warn("Failed to clean up shared temp dir: {}", sharedTempDir, e);
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

        // Scale calibration by downsample factor so QuPath requests tiles at the correct resolution
        if (downsample > 1.0) {
            cal = cal.createScaledInstance(downsample, downsample);
        }

        // Build classification labels map
        Map<Integer, PathClass> labels = new LinkedHashMap<>();
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();
        List<ImageChannel> channels = new ArrayList<>();

        for (ClassifierMetadata.ClassInfo classInfo : classes) {
            int color = parseClassColor(classInfo.color(), classInfo.index());
            PathClass pathClass = PathClass.fromString(classInfo.name(), color);
            // Use the actual color from the resolved PathClass (may differ from metadata
            // if QuPath already has a cached PathClass with a different color)
            int resolvedColor = pathClass.getColor();
            resolvedClassColors.put(classInfo.index(), resolvedColor);
            labels.put(classInfo.index(), pathClass);
            channels.add(ImageChannel.getInstance(classInfo.name(), resolvedColor));
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
     * Uses colors resolved from PathClass cache (populated by buildPixelMetadata) so that
     * overlay colors match the annotation class colors the user sees in QuPath.
     */
    private IndexColorModel buildColorModel() {
        byte[] r = new byte[256];
        byte[] g = new byte[256];
        byte[] b = new byte[256];
        byte[] a = new byte[256];

        for (Map.Entry<Integer, Integer> entry : resolvedClassColors.entrySet()) {
            int idx = entry.getKey();
            if (idx < 0 || idx >= 256) continue;
            int color = entry.getValue();
            r[idx] = (byte) ColorTools.red(color);
            g[idx] = (byte) ColorTools.green(color);
            b[idx] = (byte) ColorTools.blue(color);
            a[idx] = (byte) 255;
        }

        return new IndexColorModel(8, 256, r, g, b, a);
    }

    /** Distinct color palette for fallback when class metadata lacks colors. */
    private static final int[][] FALLBACK_PALETTE = {
            {255, 0, 0}, {0, 170, 0}, {0, 0, 255}, {255, 255, 0},
            {255, 0, 255}, {0, 255, 255}, {255, 136, 0}, {136, 0, 255}
    };

    /**
     * Parses a hex color string to a packed RGB integer (QuPath format).
     * Falls back to a distinct palette color for the given class index.
     */
    private static int parseClassColor(String colorStr, int classIndex) {
        if (colorStr == null || colorStr.isEmpty() || "#808080".equals(colorStr)) {
            // Use distinct fallback color instead of gray
            int[] c = FALLBACK_PALETTE[classIndex % FALLBACK_PALETTE.length];
            return ColorTools.packRGB(c[0], c[1], c[2]);
        }
        try {
            String hex = colorStr.startsWith("#") ? colorStr.substring(1) : colorStr;
            int rgb = Integer.parseInt(hex, 16);
            return ColorTools.packRGB(
                    (rgb >> 16) & 0xFF,
                    (rgb >> 8) & 0xFF,
                    rgb & 0xFF);
        } catch (NumberFormatException e) {
            int[] c = FALLBACK_PALETTE[classIndex % FALLBACK_PALETTE.length];
            return ColorTools.packRGB(c[0], c[1], c[2]);
        }
    }

    /**
     * Parses a hex color string to a packed RGB integer (QuPath format).
     */
    private static int parseClassColor(String colorStr) {
        return parseClassColor(colorStr, 0);
    }
}
