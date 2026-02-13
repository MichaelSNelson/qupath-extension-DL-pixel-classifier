package qupath.ext.dlclassifier.controller;

import javafx.application.Platform;
import javafx.scene.control.Alert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.DLClassifierChecks;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.ext.dlclassifier.service.DLPixelClassifier;
import qupath.ext.dlclassifier.service.OverlayService;
import qupath.ext.dlclassifier.ui.InferenceDialog;
import qupath.ext.dlclassifier.ui.ProgressMonitorController;
import qupath.ext.dlclassifier.utilities.BitDepthConverter;
import qupath.ext.dlclassifier.utilities.ChannelNormalizer;
import qupath.ext.dlclassifier.utilities.OutputGenerator;
import qupath.ext.dlclassifier.utilities.TileProcessor;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.scripting.QP;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.roi.interfaces.ROI;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Workflow for applying a trained classifier to images.
 * <p>
 * This workflow:
 * <ol>
 *   <li>Loads a trained classifier</li>
 *   <li>Generates tiles from the image or annotations</li>
 *   <li>Runs inference on the server</li>
 *   <li>Merges results and generates output (measurements, objects, or overlay)</li>
 * </ol>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class InferenceWorkflow {

    private static final Logger logger = LoggerFactory.getLogger(InferenceWorkflow.class);

    private QuPathGUI qupath;

    public InferenceWorkflow() {
        this.qupath = QuPathGUI.getInstance();
    }

    // ==================== Headless Result Record ====================

    /**
     * Result of a headless inference run.
     *
     * @param processedAnnotations number of annotations processed
     * @param processedTiles       total number of tiles processed
     * @param objectsCreated       number of objects created (for OBJECTS output)
     * @param success              whether the inference completed successfully
     * @param message              summary or error message
     */
    public record InferenceResult(
            int processedAnnotations,
            int processedTiles,
            int objectsCreated,
            boolean success,
            String message
    ) {}

    // ==================== Builder API ====================

    /**
     * Creates a new builder for configuring and running inference without GUI.
     * <p>
     * Example usage:
     * <pre>{@code
     * InferenceResult result = InferenceWorkflow.builder()
     *     .classifier(metadata)
     *     .config(inferenceConfig)
     *     .channels(channelConfig)
     *     .annotations(annotationList)
     *     .build()
     *     .run();
     * }</pre>
     *
     * @return a new InferenceBuilder
     */
    public static InferenceBuilder builder() {
        return new InferenceBuilder();
    }

    /**
     * Builder for configuring headless inference runs.
     */
    public static class InferenceBuilder {
        private ClassifierMetadata classifier;
        private InferenceConfig config;
        private ChannelConfiguration channels;
        private List<PathObject> annotations;
        private ImageData<BufferedImage> imageData;

        private InferenceBuilder() {}

        /** Sets the classifier metadata (required). */
        public InferenceBuilder classifier(ClassifierMetadata classifier) {
            this.classifier = classifier;
            return this;
        }

        /** Sets the inference configuration (required). */
        public InferenceBuilder config(InferenceConfig config) {
            this.config = config;
            return this;
        }

        /** Sets the channel configuration (required). */
        public InferenceBuilder channels(ChannelConfiguration channels) {
            this.channels = channels;
            return this;
        }

        /** Sets the annotations to classify (required). */
        public InferenceBuilder annotations(List<PathObject> annotations) {
            this.annotations = annotations;
            return this;
        }

        /**
         * Sets the image data to use. If not provided, falls back to
         * {@code QP.getCurrentImageData()} at run time.
         */
        public InferenceBuilder imageData(ImageData<BufferedImage> imageData) {
            this.imageData = imageData;
            return this;
        }

        /**
         * Validates parameters and builds an {@link InferenceRunner}.
         *
         * @return a runner ready to execute inference
         * @throws IllegalStateException if required parameters are missing
         */
        public InferenceRunner build() {
            Objects.requireNonNull(classifier, "Classifier metadata is required");
            Objects.requireNonNull(config, "InferenceConfig is required");
            Objects.requireNonNull(channels, "ChannelConfiguration is required");
            if (annotations == null || annotations.isEmpty()) {
                throw new IllegalStateException("At least one annotation is required");
            }
            return new InferenceRunner(classifier, config, channels, annotations, imageData);
        }
    }

    /**
     * Executes inference synchronously without GUI dependencies.
     */
    public static class InferenceRunner {
        private final ClassifierMetadata classifier;
        private final InferenceConfig config;
        private final ChannelConfiguration channels;
        private final List<PathObject> annotations;
        private final ImageData<BufferedImage> imageData;

        private InferenceRunner(ClassifierMetadata classifier,
                                InferenceConfig config,
                                ChannelConfiguration channels,
                                List<PathObject> annotations,
                                ImageData<BufferedImage> imageData) {
            this.classifier = classifier;
            this.config = config;
            this.channels = channels;
            this.annotations = new ArrayList<>(annotations);
            this.imageData = imageData;
        }

        /**
         * Runs inference synchronously and returns the result.
         *
         * @return the inference result
         */
        public InferenceResult run() {
            ImageData<BufferedImage> imgData = this.imageData;
            if (imgData == null) {
                imgData = QP.getCurrentImageData();
            }
            if (imgData == null) {
                logger.warn("No image data available for inference");
                return new InferenceResult(0, 0, 0, false, "No image data available");
            }

            try {
                ImageServer<BufferedImage> server = imgData.getServer();
                TileProcessor tileProcessor = new TileProcessor(config);

                ClassifierClient client = new ClassifierClient(
                        DLClassifierPreferences.getServerHost(),
                        DLClassifierPreferences.getServerPort()
                );

                int processedAnnotations = 0;
                int processedTiles = 0;
                int objectsCreated = 0;

                for (PathObject annotation : annotations) {
                    ROI region = annotation.getROI();
                    int tilesForRegion = processRegionCore(
                            region, annotation, tileProcessor, client,
                            classifier, channels, config, server, imgData,
                            null // no progress monitor
                    );
                    processedTiles += tilesForRegion;
                    processedAnnotations++;
                }

                imgData.getHierarchy().fireHierarchyChangedEvent(
                        imgData.getHierarchy().getRootObject());

                String message = String.format(
                        "Classification completed: %d annotation(s), %d tile(s)",
                        processedAnnotations, processedTiles);
                return new InferenceResult(processedAnnotations, processedTiles,
                        objectsCreated, true, message);

            } catch (Exception e) {
                logger.error("Headless inference failed", e);
                return new InferenceResult(0, 0, 0, false,
                        "Inference failed: " + e.getMessage());
            }
        }
    }

    /**
     * Starts the inference workflow.
     */
    public void start() {
        logger.info("Starting inference workflow");

        // Validate prerequisites
        if (!validatePrerequisites()) {
            return;
        }

        // Show inference dialog
        Platform.runLater(this::showInferenceDialog);
    }

    /**
     * Validates that all prerequisites for inference are met.
     */
    private boolean validatePrerequisites() {
        ImageData<BufferedImage> imageData = qupath.getImageData();
        if (imageData == null) {
            showError("No Image", "Please open an image before applying a classifier.");
            return false;
        }

        if (!DLClassifierChecks.checkServerHealth()) {
            showError("Server Unavailable",
                    "Cannot connect to classification server.\n" +
                            "Please start the Python server and check settings.");
            return false;
        }

        return true;
    }

    /**
     * Shows the inference configuration dialog.
     */
    private void showInferenceDialog() {
        InferenceDialog.showDialog()
                .thenAccept(result -> {
                    if (result != null) {
                        logger.info("Inference dialog completed. Classifier: {}",
                                result.classifier().getName());

                        // Get target objects
                        ImageData<BufferedImage> imageData = qupath.getImageData();
                        List<PathObject> targetObjects;

                        if (result.applyToSelected()) {
                            Collection<PathObject> selected = imageData.getHierarchy().getSelectionModel()
                                    .getSelectedObjects();
                            targetObjects = selected.stream()
                                    .filter(o -> o.isAnnotation() || o.isTMACore())
                                    .toList();

                            if (targetObjects.isEmpty()) {
                                showError("No Selection",
                                        "No annotations selected. Please select annotations to classify.");
                                return;
                            }
                        } else {
                            targetObjects = new ArrayList<>(imageData.getHierarchy().getAnnotationObjects());
                            if (targetObjects.isEmpty()) {
                                showError("No Annotations",
                                        "No annotations found. Please create annotations to classify.");
                                return;
                            }
                        }

                        // Run inference with progress
                        runInferenceWithProgress(
                                result.classifier(),
                                result.inferenceConfig(),
                                result.channelConfig(),
                                targetObjects
                        );
                    }
                })
                .exceptionally(ex -> {
                    logger.error("Inference dialog failed", ex);
                    showError("Error", "Failed to show inference dialog: " + ex.getMessage());
                    return null;
                });
    }

    /**
     * Runs inference on the specified region with progress monitoring.
     *
     * @param metadata        classifier metadata
     * @param inferenceConfig inference configuration
     * @param channelConfig   channel configuration
     * @param targetObjects   objects to classify
     */
    public void runInferenceWithProgress(ClassifierMetadata metadata,
                                         InferenceConfig inferenceConfig,
                                         ChannelConfiguration channelConfig,
                                         List<PathObject> targetObjects) {
        // Create progress monitor
        ProgressMonitorController progress = ProgressMonitorController.forInference();
        progress.setOnCancel(v -> {
            logger.info("Inference cancellation requested");
        });
        progress.show();

        CompletableFuture.runAsync(() -> {
            try {
                progress.setStatus("Preparing inference...");
                progress.log("Classifier: " + metadata.getName());
                progress.log("Processing " + targetObjects.size() + " annotation(s)");

                ImageData<BufferedImage> imageData = qupath.getImageData();
                ImageServer<BufferedImage> server = imageData.getServer();

                // Create tile processor
                TileProcessor tileProcessor = new TileProcessor(inferenceConfig);

                // Create client
                progress.setStatus("Connecting to server...");
                ClassifierClient client = new ClassifierClient(
                        DLClassifierPreferences.getServerHost(),
                        DLClassifierPreferences.getServerPort()
                );
                progress.log("Connected to server");

                // Count total tiles
                int totalTiles = 0;
                for (PathObject obj : targetObjects) {
                    List<TileProcessor.TileSpec> specs = tileProcessor.generateTiles(obj.getROI(), server);
                    totalTiles += specs.size();
                }
                progress.log("Total tiles to process: " + totalTiles);

                // Process each annotation
                int processedAnnotations = 0;
                int processedTiles = 0;

                for (PathObject annotation : targetObjects) {
                    if (progress.isCancelled()) {
                        progress.complete(false, "Inference cancelled by user");
                        return;
                    }

                    String annotationName = annotation.getName() != null ?
                            annotation.getName() : "Annotation " + (processedAnnotations + 1);
                    progress.setStatus("Processing: " + annotationName);
                    progress.log("Processing annotation: " + annotationName);

                    ROI region = annotation.getROI();
                    int tilesForRegion = processRegionWithProgress(
                            region, annotation, tileProcessor, client, metadata,
                            channelConfig, inferenceConfig, server, imageData, progress
                    );

                    processedTiles += tilesForRegion;
                    processedAnnotations++;

                    progress.setOverallProgress((double) processedAnnotations / targetObjects.size());
                    progress.log("Completed " + processedAnnotations + "/" + targetObjects.size() + " annotations");
                }

                // Fire hierarchy update
                imageData.getHierarchy().fireHierarchyChangedEvent(
                        imageData.getHierarchy().getRootObject());

                progress.complete(true, String.format(
                        "Classification completed!\nProcessed %d annotation(s), %d tile(s)",
                        processedAnnotations, processedTiles));

            } catch (Exception e) {
                logger.error("Inference failed", e);
                progress.log("ERROR: " + e.getMessage());
                progress.complete(false, "Inference failed: " + e.getMessage());
            }
        });
    }

    /**
     * Runs inference on the specified region.
     *
     * @param metadata        classifier metadata
     * @param inferenceConfig inference configuration
     * @param channelConfig   channel configuration
     * @param targetObjects   objects to classify (null for whole image)
     * @deprecated Use {@link #runInferenceWithProgress} instead
     */
    @Deprecated
    public void runInference(ClassifierMetadata metadata,
                             InferenceConfig inferenceConfig,
                             ChannelConfiguration channelConfig,
                             List<PathObject> targetObjects) {
        runInferenceWithProgress(metadata, inferenceConfig, channelConfig, targetObjects);
    }

    /**
     * Processes a single region with progress updates.
     *
     * @return the number of tiles processed
     */
    private int processRegionWithProgress(ROI region,
                                          PathObject parentObject,
                                          TileProcessor tileProcessor,
                                          ClassifierClient client,
                                          ClassifierMetadata metadata,
                                          ChannelConfiguration channelConfig,
                                          InferenceConfig inferenceConfig,
                                          ImageServer<BufferedImage> server,
                                          ImageData<BufferedImage> imageData,
                                          ProgressMonitorController progress) throws IOException {
        return processRegionCore(region, parentObject, tileProcessor, client,
                metadata, channelConfig, inferenceConfig, server, imageData, progress);
    }

    /**
     * Core region processing logic shared by GUI and headless paths.
     * <p>
     * When {@code progress} is {@code null}, progress updates and cancellation
     * checks are skipped, enabling headless execution.
     *
     * @param region          the ROI to process
     * @param parentObject    the parent annotation
     * @param tileProcessor   tile processor for generating tiles
     * @param client          classifier server client
     * @param metadata        classifier metadata
     * @param channelConfig   channel configuration
     * @param inferenceConfig inference configuration
     * @param server          image server
     * @param imageData       image data
     * @param progress        progress monitor (nullable for headless execution)
     * @return the number of tiles processed
     * @throws IOException if tile processing or server communication fails
     */
    static int processRegionCore(ROI region,
                                 PathObject parentObject,
                                 TileProcessor tileProcessor,
                                 ClassifierClient client,
                                 ClassifierMetadata metadata,
                                 ChannelConfiguration channelConfig,
                                 InferenceConfig inferenceConfig,
                                 ImageServer<BufferedImage> server,
                                 ImageData<BufferedImage> imageData,
                                 ProgressMonitorController progress) throws IOException {
        // Generate tiles
        List<TileProcessor.TileSpec> tileSpecs = tileProcessor.generateTiles(region, server);
        logger.info("Generated {} tiles for region", tileSpecs.size());
        if (progress != null) progress.log("Generated " + tileSpecs.size() + " tiles");

        // OVERLAY mode uses on-demand tile rendering via QuPath's PixelClassifier
        // interface, so skip batch tile processing entirely
        if (inferenceConfig.getOutputType() == InferenceConfig.OutputType.OVERLAY) {
            DLPixelClassifier pixelClassifier = new DLPixelClassifier(
                    metadata, channelConfig, inferenceConfig, imageData);
            OverlayService.getInstance().applyClassifierOverlay(imageData, pixelClassifier);
            if (progress != null) {
                progress.log("Classification overlay applied - tiles rendered on demand");
            }
            return 0;
        }

        // Branch on output type: MEASUREMENTS uses aggregated tile-level inference,
        // OBJECTS needs full pixel-level probability maps
        boolean usePixelInference = inferenceConfig.getOutputType() != InferenceConfig.OutputType.MEASUREMENTS;

        // Process in batches
        int batchSize = tileProcessor.getMaxTilesInMemory();
        List<float[][][]> allResults = new ArrayList<>();
        Path tempDir = null;

        try {
            if (usePixelInference) {
                tempDir = Files.createTempDirectory("dl-pixel-inference-");
                logger.info("Using pixel-level inference, temp dir: {}", tempDir);
            }

            for (int i = 0; i < tileSpecs.size(); i += batchSize) {
                if (progress != null && progress.isCancelled()) {
                    return i;
                }

                int end = Math.min(i + batchSize, tileSpecs.size());
                List<TileProcessor.TileSpec> batch = tileSpecs.subList(i, end);

                if (progress != null) {
                    progress.setDetail(String.format("Processing tiles %d-%d of %d", i + 1, end, tileSpecs.size()));
                    progress.setCurrentProgress((double) i / tileSpecs.size());
                }

                // Read and encode tiles
                List<ClassifierClient.TileData> tileDataList = new ArrayList<>();
                for (TileProcessor.TileSpec spec : batch) {
                    BufferedImage tileImage = tileProcessor.readTile(spec, server);
                    String encoded = encodeTile(tileImage);
                    tileDataList.add(new ClassifierClient.TileData(
                            String.valueOf(spec.index()),
                            encoded,
                            spec.x(),
                            spec.y()
                    ));
                }

                if (usePixelInference) {
                    // Pixel-level inference: get full probability maps for each tile
                    ClassifierClient.PixelInferenceResult pixelResult = client.runPixelInference(
                            metadata.getId(),
                            tileDataList,
                            channelConfig,
                            inferenceConfig,
                            tempDir
                    );

                    if (pixelResult != null && pixelResult.outputPaths() != null) {
                        int tileSize = inferenceConfig.getTileSize();
                        for (ClassifierClient.TileData tile : tileDataList) {
                            String outputPath = pixelResult.outputPaths().get(tile.id());
                            if (outputPath != null) {
                                float[][][] probMap = ClassifierClient.readProbabilityMap(
                                        Path.of(outputPath),
                                        pixelResult.numClasses(),
                                        tileSize,
                                        tileSize
                                );
                                allResults.add(probMap);
                            }
                        }
                    }
                } else {
                    // Aggregated tile-level inference for MEASUREMENTS output
                    ClassifierClient.InferenceResult result = client.runInference(
                            metadata.getId(),
                            tileDataList,
                            channelConfig,
                            inferenceConfig
                    );

                    if (result != null && result.predictions() != null) {
                        for (float[] probs : result.predictions().values()) {
                            float[][][] tileResult = new float[1][1][probs.length];
                            tileResult[0][0] = probs;
                            allResults.add(tileResult);
                        }
                    }
                }
            }

            if (progress != null) progress.setCurrentProgress(1.0);

            // Create output generator
            OutputGenerator outputGenerator = new OutputGenerator(imageData, metadata, inferenceConfig);

            // Generate output based on type
            switch (inferenceConfig.getOutputType()) {
                case MEASUREMENTS:
                    outputGenerator.addMeasurements(parentObject, allResults, tileSpecs);
                    if (progress != null) progress.log("Added measurements to annotation");
                    break;

                case OBJECTS:
                    // Use the new merged-map approach for proper cross-tile object handling
                    int numClasses = metadata.getClasses().size();
                    List<PathObject> objects = outputGenerator.createObjectsFromTiles(
                            tileProcessor,
                            allResults,
                            tileSpecs,
                            region,
                            numClasses,
                            inferenceConfig.getObjectType()
                    );
                    imageData.getHierarchy().addObjects(objects);
                    if (progress != null) {
                        progress.log("Created " + objects.size() + " " +
                                inferenceConfig.getObjectType().name().toLowerCase() + " objects");
                    }
                    break;

                case OVERLAY:
                    // Should not reach here - OVERLAY exits early above
                    logger.warn("OVERLAY case reached in switch - this should not happen");
                    break;
            }

            return tileSpecs.size();
        } finally {
            // Clean up temp directory for pixel inference
            if (tempDir != null) {
                try {
                    Files.walk(tempDir)
                            .sorted(Comparator.reverseOrder())
                            .forEach(path -> {
                                try {
                                    Files.deleteIfExists(path);
                                } catch (IOException e) {
                                    logger.warn("Failed to delete temp file: {}", path, e);
                                }
                            });
                } catch (IOException e) {
                    logger.warn("Failed to clean up temp directory: {}", tempDir, e);
                }
            }
        }
    }

    /**
     * Processes a single region.
     * @deprecated Use {@link #processRegionWithProgress} instead
     */
    @Deprecated
    private void processRegion(ROI region,
                               TileProcessor tileProcessor,
                               ClassifierClient client,
                               ClassifierMetadata metadata,
                               ChannelConfiguration channelConfig,
                               InferenceConfig inferenceConfig,
                               ImageServer<BufferedImage> server) throws IOException {
        // Generate tiles
        List<TileProcessor.TileSpec> tileSpecs = tileProcessor.generateTiles(region, server);
        logger.info("Generated {} tiles for region", tileSpecs.size());

        // Process in batches
        int batchSize = tileProcessor.getMaxTilesInMemory();

        for (int i = 0; i < tileSpecs.size(); i += batchSize) {
            int end = Math.min(i + batchSize, tileSpecs.size());
            List<TileProcessor.TileSpec> batch = tileSpecs.subList(i, end);

            // Read and encode tiles
            List<ClassifierClient.TileData> tileDataList = new ArrayList<>();
            for (TileProcessor.TileSpec spec : batch) {
                BufferedImage tileImage = tileProcessor.readTile(spec, server);
                String encoded = encodeTile(tileImage);
                tileDataList.add(new ClassifierClient.TileData(
                        String.valueOf(spec.index()),
                        encoded,
                        spec.x(),
                        spec.y()
                ));
            }

            // Run inference
            client.runInference(
                    metadata.getId(),
                    tileDataList,
                    channelConfig,
                    inferenceConfig
            );
        }
    }

    /**
     * Encodes a tile image to base64.
     */
    private static String encodeTile(BufferedImage image) throws IOException {
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        javax.imageio.ImageIO.write(image, "png", baos);
        return Base64.getEncoder().encodeToString(baos.toByteArray());
    }

    /**
     * Shows an error dialog.
     */
    private void showError(String title, String message) {
        Platform.runLater(() -> {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle(title);
            alert.setHeaderText(null);
            alert.setContentText(message);
            alert.showAndWait();
        });
    }
}
