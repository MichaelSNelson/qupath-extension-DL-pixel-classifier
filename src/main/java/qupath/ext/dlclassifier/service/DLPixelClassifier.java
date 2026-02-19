package qupath.ext.dlclassifier.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.controller.InferenceWorkflow;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.ClassifierClient.PixelInferenceResult;
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
import java.awt.image.DataBuffer;
import java.awt.image.IndexColorModel;
import java.awt.image.Raster;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
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
    private final int contextScale;
    private final PixelClassifierMetadata pixelMetadata;
    private final IndexColorModel colorModel;
    private final ClassifierBackend backend;
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

    // ==================== Image-Level Normalization ====================

    /** Number of tiles sampled across the image for normalization stats. */
    private static final int STATS_GRID_SIZE = 4;
    /** Maximum pixel samples per channel for stats computation. */
    private static final int STATS_TARGET_SAMPLES = 100_000;

    /** Cached channel config with precomputed normalization stats (lazy init). */
    private volatile ChannelConfiguration channelConfigWithStats;
    private final Object statsLock = new Object();

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
        this.contextScale = metadata.getContextScale();
        this.pixelMetadata = buildPixelMetadata(imageData);
        this.colorModel = buildColorModel();
        this.backend = BackendFactory.getBackend();

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

        // Lazily compute image-level normalization stats on first tile request
        // (double-checked locking for thread safety)
        if (channelConfigWithStats == null) {
            synchronized (statsLock) {
                if (channelConfigWithStats == null) {
                    channelConfigWithStats = computeChannelConfigWithStats(server);
                }
            }
        }

        // Read the tile from the image server
        BufferedImage tileImage = server.readRegion(request);
        if (tileImage == null) {
            throw new IOException("Failed to read tile at " + request);
        }

        // Encode detail tile as raw binary (uint8 fast path for simple RGB, float32 for N-channel)
        String dtype;
        byte[] detailBytes;
        int detailChannels;
        if (InferenceWorkflow.isSimpleRgb(tileImage)) {
            dtype = "uint8";
            detailBytes = InferenceWorkflow.encodeTileRaw(tileImage);
            detailChannels = 3;
        } else {
            dtype = "float32";
            detailBytes = InferenceWorkflow.encodeTileRawFloat(tileImage,
                    channelConfig.getSelectedChannels());
            detailChannels = channelConfig.getSelectedChannels().isEmpty()
                    ? tileImage.getRaster().getNumBands()
                    : channelConfig.getSelectedChannels().size();
        }

        // When multi-scale context is enabled, extract a context tile and concatenate
        byte[] rawBytes;
        int numChannels;
        if (contextScale > 1) {
            BufferedImage contextImage = readContextTile(server, request);
            byte[] contextBytes;
            if ("uint8".equals(dtype)) {
                contextBytes = InferenceWorkflow.encodeTileRaw(contextImage);
            } else {
                contextBytes = InferenceWorkflow.encodeTileRawFloat(contextImage,
                        channelConfig.getSelectedChannels());
            }
            // Concatenate detail + context bytes
            rawBytes = new byte[detailBytes.length + contextBytes.length];
            System.arraycopy(detailBytes, 0, rawBytes, 0, detailBytes.length);
            System.arraycopy(contextBytes, 0, rawBytes, detailBytes.length, contextBytes.length);
            numChannels = detailChannels * 2;
        } else {
            rawBytes = detailBytes;
            numChannels = detailChannels;
        }

        String tileId = String.format("%d_%d_%d_%d",
                request.getX(), request.getY(), request.getWidth(), request.getHeight());

        try {
            // Use binary pixel inference (single-tile batch)
            // Pass channelConfigWithStats which includes precomputed normalization stats
            int reflectionPadding = DLClassifierPreferences.getOverlayReflectionPadding();
            PixelInferenceResult result = backend.runPixelInferenceBinary(
                    modelDirPath, rawBytes, List.of(tileId),
                    tileImage.getHeight(), tileImage.getWidth(), numChannels,
                    dtype, channelConfigWithStats, inferenceConfig, sharedTempDir,
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
                result = backend.runPixelInference(
                        modelDirPath, tiles, channelConfigWithStats, inferenceConfig,
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

            // "thread death" is transient (Python worker thread contention under high
            // concurrency). QuPath re-requests the tile on repaint, so don't count it
            // toward the circuit breaker -- it would trip on startup when many tiles
            // are requested simultaneously.
            String msg = e.getMessage() != null ? e.getMessage() : "";
            if (msg.toLowerCase().contains("thread death")) {
                logger.debug("Transient thread death for tile, will retry on repaint");
                throw e;
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
     * Reads a context tile centered on the same location as the given detail request,
     * but covering a larger area (contextScale times in each dimension) and downsampled
     * to the same pixel dimensions as the detail tile.
     */
    private BufferedImage readContextTile(ImageServer<BufferedImage> server,
                                          RegionRequest detailRequest) throws IOException {
        int detailX = detailRequest.getX();
        int detailY = detailRequest.getY();
        int detailW = detailRequest.getWidth();
        int detailH = detailRequest.getHeight();

        // Context region covers contextScale times the area in each dimension
        int contextW = detailW * contextScale;
        int contextH = detailH * contextScale;

        // Center the context region on the detail tile's center
        int centerX = detailX + detailW / 2;
        int centerY = detailY + detailH / 2;
        int cx = centerX - contextW / 2;
        int cy = centerY - contextH / 2;

        // Clamp to image bounds
        cx = Math.max(0, Math.min(cx, server.getWidth() - contextW));
        cy = Math.max(0, Math.min(cy, server.getHeight() - contextH));
        int clampedW = Math.min(contextW, server.getWidth() - cx);
        int clampedH = Math.min(contextH, server.getHeight() - cy);

        // Read at higher downsample so output has same pixel dimensions as detail tile
        double contextDownsample = detailRequest.getDownsample() * contextScale;
        RegionRequest contextRequest = RegionRequest.createInstance(
                server.getPath(), contextDownsample,
                cx, cy, clampedW, clampedH,
                detailRequest.getZ(), detailRequest.getT());

        BufferedImage contextImage = server.readRegion(contextRequest);
        if (contextImage == null) {
            throw new IOException("Failed to read context tile at " + contextRequest);
        }
        return contextImage;
    }

    // ==================== Image-Level Normalization Stats ====================

    /**
     * Computes or retrieves precomputed normalization statistics and returns a
     * ChannelConfiguration with the stats attached.
     * <p>
     * Priority order:
     * <ol>
     *   <li>Training dataset stats from model metadata (Phase 2, most accurate)</li>
     *   <li>Image-level stats via tile sampling (Phase 1, ~1-3s one-time cost)</li>
     *   <li>Per-tile normalization (fallback, can cause tile boundary artifacts)</li>
     * </ol>
     * <p>
     * FIXED_RANGE normalization always uses the user-specified values directly.
     *
     * @param server the image server to sample from
     * @return channelConfig with precomputed stats, or original if not applicable
     */
    private ChannelConfiguration computeChannelConfigWithStats(ImageServer<BufferedImage> server) {
        // FIXED_RANGE uses user-specified values, no sampling needed
        if (channelConfig.getNormalizationStrategy() ==
                ChannelConfiguration.NormalizationStrategy.FIXED_RANGE) {
            logger.info("FIXED_RANGE normalization: skipping image-level stats");
            return channelConfig;
        }

        // Priority 1: Use training dataset stats from model metadata
        if (metadata.hasNormalizationStats()) {
            logger.info("Using training dataset normalization stats from model metadata " +
                    "({} channels)", metadata.getNormalizationStats().size());
            return channelConfig.withPrecomputedStats(metadata.getNormalizationStats());
        }

        // Priority 2: Compute image-level stats via sampling
        try {
            List<Map<String, Double>> stats = computeImageNormalizationStats(server);
            if (stats != null && !stats.isEmpty()) {
                logger.info("Computed image-level normalization stats from {} sample tiles " +
                        "({} channels)", STATS_GRID_SIZE * STATS_GRID_SIZE, stats.size());
                return channelConfig.withPrecomputedStats(stats);
            }
        } catch (Exception e) {
            logger.warn("Failed to compute image-level normalization stats, " +
                    "falling back to per-tile normalization: {}", e.getMessage());
        }

        // Priority 3: Fall back to original config (per-tile normalization)
        return channelConfig;
    }

    /**
     * Samples the image to compute per-channel normalization statistics.
     * <p>
     * Reads tiles from a grid of sample locations across the image,
     * collects pixel values using reservoir sampling, and computes
     * aggregate statistics for each channel.
     *
     * @param server the image server to sample from
     * @return list of per-channel stat maps, or null on failure
     */
    private List<Map<String, Double>> computeImageNormalizationStats(
            ImageServer<BufferedImage> server) throws IOException {
        int imgWidth = server.getWidth();
        int imgHeight = server.getHeight();

        // Tile dimensions at full resolution (pre-downsample)
        int tileW = (int) (metadata.getInputWidth() * downsample);
        int tileH = (int) (metadata.getInputHeight() * downsample);
        tileW = Math.min(tileW, imgWidth);
        tileH = Math.min(tileH, imgHeight);

        // Determine number of channels we'll be extracting
        List<Integer> selectedChannels = channelConfig.getSelectedChannels();
        int numChannels = selectedChannels.isEmpty()
                ? server.nChannels()
                : selectedChannels.size();

        // Collect pixel samples per channel
        List<List<Float>> channelSamples = new ArrayList<>();
        for (int c = 0; c < numChannels; c++) {
            channelSamples.add(new ArrayList<>());
        }

        // Compute grid step sizes (full resolution coordinates)
        int gridSize = Math.min(STATS_GRID_SIZE, Math.max(1,
                Math.min(imgWidth / tileW, imgHeight / tileH)));
        int stepX = (imgWidth - tileW) / Math.max(1, gridSize - 1);
        int stepY = (imgHeight - tileH) / Math.max(1, gridSize - 1);

        // Estimate subsample rate to keep total around TARGET_SAMPLES per channel
        int pixelsPerTile = metadata.getInputWidth() * metadata.getInputHeight();
        int totalEstimatedPixels = pixelsPerTile * gridSize * gridSize;
        int subsampleRate = Math.max(1, totalEstimatedPixels / STATS_TARGET_SAMPLES);

        int sampledTiles = 0;
        for (int gy = 0; gy < gridSize; gy++) {
            for (int gx = 0; gx < gridSize; gx++) {
                int x = Math.min(gx * stepX, imgWidth - tileW);
                int y = Math.min(gy * stepY, imgHeight - tileH);

                RegionRequest req = RegionRequest.createInstance(
                        server.getPath(), downsample, x, y, tileW, tileH);
                BufferedImage tile = server.readRegion(req);
                if (tile == null) continue;

                Raster raster = tile.getRaster();
                int bands = raster.getNumBands();
                int dataType = raster.getDataBuffer().getDataType();
                boolean isUint8 = (dataType == DataBuffer.TYPE_BYTE);

                // Determine which bands to sample
                int w = tile.getWidth();
                int h = tile.getHeight();
                int pixelIndex = 0;

                for (int py = 0; py < h; py++) {
                    for (int px = 0; px < w; px++) {
                        pixelIndex++;
                        if (pixelIndex % subsampleRate != 0) continue;

                        for (int c = 0; c < numChannels; c++) {
                            int band = selectedChannels.isEmpty() ? c : selectedChannels.get(c);
                            if (band >= bands) continue;

                            float val;
                            if (isUint8) {
                                // Match the uint8->float conversion done in the backend
                                val = (raster.getSample(px, py, band) & 0xFF) / 255.0f;
                            } else {
                                val = raster.getSampleFloat(px, py, band);
                            }
                            channelSamples.get(c).add(val);
                        }
                    }
                }
                sampledTiles++;
            }
        }

        if (sampledTiles == 0) {
            logger.warn("No tiles could be sampled for normalization stats");
            return null;
        }

        // Compute per-channel statistics from collected samples
        List<Map<String, Double>> channelStats = new ArrayList<>();
        for (int c = 0; c < numChannels; c++) {
            List<Float> samples = channelSamples.get(c);
            if (samples.isEmpty()) {
                // Default stats for empty channel
                Map<String, Double> stats = new HashMap<>();
                stats.put("p1", 0.0);
                stats.put("p99", 1.0);
                stats.put("min", 0.0);
                stats.put("max", 1.0);
                stats.put("mean", 0.5);
                stats.put("std", 0.25);
                channelStats.add(stats);
                continue;
            }

            // Sort for percentile computation
            float[] arr = new float[samples.size()];
            for (int i = 0; i < arr.length; i++) arr[i] = samples.get(i);
            Arrays.sort(arr);

            int n = arr.length;
            double p1 = arr[Math.max(0, (int) (n * 0.01))];
            double p99 = arr[Math.min(n - 1, (int) (n * 0.99))];
            double min = arr[0];
            double max = arr[n - 1];

            // Compute mean and std
            double sum = 0;
            double sumSq = 0;
            for (float v : arr) {
                sum += v;
                sumSq += (double) v * v;
            }
            double mean = sum / n;
            double std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));

            Map<String, Double> stats = new HashMap<>();
            stats.put("p1", p1);
            stats.put("p99", p99);
            stats.put("min", min);
            stats.put("max", max);
            stats.put("mean", mean);
            stats.put("std", std);
            channelStats.add(stats);

            logger.debug("Channel {} stats: p1={}, p99={}, min={}, max={}, mean={}, std={}",
                    c, String.format("%.4f", p1), String.format("%.4f", p99),
                    String.format("%.4f", min), String.format("%.4f", max),
                    String.format("%.4f", mean), String.format("%.4f", std));
        }

        return channelStats;
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
     * <p>
     * Tile overlap (inputPadding) is computed from the preferred physical distance
     * in micrometers using the image's pixel calibration. This ensures consistent
     * CNN context at tile boundaries regardless of objective/resolution.
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
        int padding = computePhysicalOverlap(cal, tileSize);

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
     * Computes tile overlap in pixels from a target physical distance.
     * <p>
     * Uses the image's pixel calibration to convert the preferred overlap distance
     * (in micrometers) to pixels. Falls back to a minimum of 64 pixels when
     * calibration is unavailable.
     *
     * @param cal      pixel calibration (may lack micron info)
     * @param tileSize tile size in pixels (overlap is capped at tileSize/2)
     * @return overlap in pixels
     */
    static int computePhysicalOverlap(PixelCalibration cal, int tileSize) {
        double targetOverlapUm = DLClassifierPreferences.getOverlayOverlapUm();
        int minOverlap = 64;

        if (cal == null || !cal.hasPixelSizeMicrons()) {
            return minOverlap;
        }

        double pixelSizeUm = cal.getAveragedPixelSizeMicrons();
        int overlapPx = (int) Math.ceil(targetOverlapUm / pixelSizeUm);

        // Clamp: at least minOverlap, at most tileSize/2
        return Math.max(minOverlap, Math.min(overlapPx, tileSize / 2));
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
