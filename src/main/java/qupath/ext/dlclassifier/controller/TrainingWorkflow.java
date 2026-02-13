package qupath.ext.dlclassifier.controller;

import javafx.application.Platform;
import javafx.scene.control.Alert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.DLClassifierChecks;
import qupath.ext.dlclassifier.classifier.ClassifierHandler;
import qupath.ext.dlclassifier.classifier.ClassifierRegistry;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.TrainingConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.ext.dlclassifier.ui.ProgressMonitorController;
import qupath.ext.dlclassifier.ui.TrainingDialog;
import qupath.ext.dlclassifier.utilities.AnnotationExtractor;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.scripting.QP;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.ProjectImageEntry;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Workflow for training a new deep learning pixel classifier.
 * <p>
 * This workflow guides the user through:
 * <ol>
 *   <li>Selecting classification classes from annotations</li>
 *   <li>Configuring channel selection and normalization</li>
 *   <li>Setting training hyperparameters</li>
 *   <li>Exporting training data</li>
 *   <li>Training the model on the server</li>
 *   <li>Saving the trained classifier</li>
 * </ol>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class TrainingWorkflow {

    private static final Logger logger = LoggerFactory.getLogger(TrainingWorkflow.class);

    private QuPathGUI qupath;

    public TrainingWorkflow() {
        this.qupath = QuPathGUI.getInstance();
    }

    // ==================== Headless Result Record ====================

    /**
     * Result of a headless training run.
     *
     * @param classifierId    the saved classifier ID
     * @param classifierName  the classifier display name
     * @param finalLoss       final training loss
     * @param finalAccuracy   final training accuracy
     * @param epochsCompleted number of epochs completed
     * @param success         whether training completed successfully
     * @param message         summary or error message
     */
    public record TrainingResult(
            String classifierId,
            String classifierName,
            double finalLoss,
            double finalAccuracy,
            int epochsCompleted,
            boolean success,
            String message
    ) {}

    // ==================== Builder API ====================

    /**
     * Creates a new builder for configuring and running training without GUI.
     * <p>
     * Example usage:
     * <pre>{@code
     * TrainingResult result = TrainingWorkflow.builder()
     *     .name("Collagen_Classifier")
     *     .config(trainingConfig)
     *     .channels(channelConfig)
     *     .classes(List.of("Background", "Collagen"))
     *     .build()
     *     .run();
     * }</pre>
     *
     * @return a new TrainingBuilder
     */
    public static TrainingBuilder builder() {
        return new TrainingBuilder();
    }

    /**
     * Builder for configuring headless training runs.
     */
    public static class TrainingBuilder {
        private String name;
        private String description = "";
        private TrainingConfig config;
        private ChannelConfiguration channels;
        private List<String> classes;
        private ImageData<BufferedImage> imageData;

        private TrainingBuilder() {}

        /** Sets the classifier name (required). */
        public TrainingBuilder name(String name) {
            this.name = name;
            return this;
        }

        /** Sets the classifier description (optional, defaults to empty). */
        public TrainingBuilder description(String description) {
            this.description = description;
            return this;
        }

        /** Sets the training configuration (required). */
        public TrainingBuilder config(TrainingConfig config) {
            this.config = config;
            return this;
        }

        /** Sets the channel configuration (required). */
        public TrainingBuilder channels(ChannelConfiguration channels) {
            this.channels = channels;
            return this;
        }

        /** Sets the class names for training (required, minimum 2). */
        public TrainingBuilder classes(List<String> classes) {
            this.classes = classes;
            return this;
        }

        /**
         * Sets the image data to use. If not provided, falls back to
         * {@code QP.getCurrentImageData()} at run time.
         */
        public TrainingBuilder imageData(ImageData<BufferedImage> imageData) {
            this.imageData = imageData;
            return this;
        }

        /**
         * Validates parameters and builds a {@link TrainingRunner}.
         *
         * @return a runner ready to execute training
         * @throws IllegalStateException if required parameters are missing
         */
        public TrainingRunner build() {
            Objects.requireNonNull(name, "Classifier name is required");
            Objects.requireNonNull(config, "TrainingConfig is required");
            Objects.requireNonNull(channels, "ChannelConfiguration is required");
            if (classes == null || classes.size() < 2) {
                throw new IllegalStateException("At least 2 class names are required");
            }
            return new TrainingRunner(name, description, config, channels, classes, imageData);
        }
    }

    /**
     * Executes training synchronously without GUI dependencies.
     */
    public static class TrainingRunner {
        private final String name;
        private final String description;
        private final TrainingConfig config;
        private final ChannelConfiguration channels;
        private final List<String> classes;
        private final ImageData<BufferedImage> imageData;

        private TrainingRunner(String name, String description,
                               TrainingConfig config, ChannelConfiguration channels,
                               List<String> classes, ImageData<BufferedImage> imageData) {
            this.name = name;
            this.description = description;
            this.config = config;
            this.channels = channels;
            this.classes = new ArrayList<>(classes);
            this.imageData = imageData;
        }

        /**
         * Runs training synchronously and returns the result.
         *
         * @return the training result
         */
        public TrainingResult run() {
            ImageData<BufferedImage> imgData = this.imageData;
            if (imgData == null) {
                imgData = QP.getCurrentImageData();
            }
            if (imgData == null) {
                logger.warn("No image data available for training");
                return new TrainingResult(null, name, 0.0, 0.0, 0, false,
                        "No image data available");
            }

            ClassifierHandler handler = ClassifierRegistry.getHandler(config.getModelType())
                    .orElse(ClassifierRegistry.getDefaultHandler());

            return trainCore(name, description, handler, config, channels, classes,
                    imgData, null, null);
        }
    }

    /**
     * Starts the training workflow.
     */
    public void start() {
        logger.info("Starting training workflow");

        // Validate prerequisites
        if (!validatePrerequisites()) {
            return;
        }

        // Show training dialog
        Platform.runLater(this::showTrainingDialog);
    }

    /**
     * Validates that all prerequisites for training are met.
     */
    private boolean validatePrerequisites() {
        // Check image is open
        ImageData<BufferedImage> imageData = qupath.getImageData();
        if (imageData == null) {
            showError("No Image", "Please open an image before training a classifier.");
            return false;
        }

        // Check for annotations
        List<PathObject> annotations = imageData.getHierarchy().getAnnotationObjects()
                .stream()
                .filter(a -> a.getPathClass() != null)
                .toList();

        if (annotations.isEmpty()) {
            showError("No Annotations",
                    "No classified annotations found.\n" +
                            "Please create annotations with foreground and background classes.");
            return false;
        }

        // Check server availability
        if (!DLClassifierChecks.checkServerHealth()) {
            showError("Server Unavailable",
                    "Cannot connect to classification server.\n" +
                            "Please start the Python server and check settings.");
            return false;
        }

        return true;
    }

    /**
     * Shows the training configuration dialog.
     */
    private void showTrainingDialog() {
        TrainingDialog.showDialog()
                .thenAccept(result -> {
                    if (result != null) {
                        logger.info("Training dialog completed. Classifier: {}", result.classifierName());

                        // Get the classifier handler
                        ClassifierHandler handler = ClassifierRegistry.getHandler(
                                result.trainingConfig().getModelType())
                                .orElse(ClassifierRegistry.getDefaultHandler());

                        // Start training with progress monitor
                        trainClassifierWithProgress(
                                result.classifierName(),
                                result.description(),
                                handler,
                                result.trainingConfig(),
                                result.channelConfig(),
                                result.selectedClasses(),
                                result.selectedImages()
                        );
                    }
                })
                .exceptionally(ex -> {
                    logger.error("Training dialog failed", ex);
                    showError("Error", "Failed to show training dialog: " + ex.getMessage());
                    return null;
                });
    }

    /**
     * Executes the training process with progress monitoring.
     *
     * @param classifierName name for the classifier
     * @param description    classifier description
     * @param handler        the classifier handler
     * @param trainingConfig training configuration
     * @param channelConfig  channel configuration
     * @param classNames     list of class names
     * @param selectedImages project images to train from, or null for current image only
     */
    public void trainClassifierWithProgress(String classifierName,
                                            String description,
                                            ClassifierHandler handler,
                                            TrainingConfig trainingConfig,
                                            ChannelConfiguration channelConfig,
                                            List<String> classNames,
                                            List<ProjectImageEntry<BufferedImage>> selectedImages) {
        // Create progress monitor
        ProgressMonitorController progress = ProgressMonitorController.forTraining();
        progress.setOnCancel(v -> logger.info("Training cancellation requested"));
        progress.show();

        CompletableFuture.runAsync(() -> {
            TrainingResult result = trainCore(classifierName, description, handler,
                    trainingConfig, channelConfig, classNames,
                    qupath.getImageData(), selectedImages, progress);

            if (result.success()) {
                progress.complete(true, String.format(
                        "Classifier trained successfully!\nFinal loss: %.4f\nAccuracy: %.2f%%",
                        result.finalLoss(), result.finalAccuracy() * 100));
            } else {
                progress.complete(false, result.message());
            }
        });
    }

    /**
     * Core training logic shared by GUI and headless paths.
     * <p>
     * When {@code progress} is {@code null}, progress updates and cancellation
     * checks are skipped, enabling headless execution.
     *
     * @param classifierName name for the classifier
     * @param description    classifier description
     * @param handler        the classifier handler
     * @param trainingConfig training configuration
     * @param channelConfig  channel configuration
     * @param classNames     list of class names
     * @param imageData      image data for extracting training patches (single-image mode)
     * @param selectedImages project images for multi-image training, or null for single-image
     * @param progress       progress monitor (nullable for headless execution)
     * @return the training result
     */
    static TrainingResult trainCore(String classifierName,
                                    String description,
                                    ClassifierHandler handler,
                                    TrainingConfig trainingConfig,
                                    ChannelConfiguration channelConfig,
                                    List<String> classNames,
                                    ImageData<BufferedImage> imageData,
                                    List<ProjectImageEntry<BufferedImage>> selectedImages,
                                    ProgressMonitorController progress) {
        try {
            if (progress != null) {
                progress.setStatus("Exporting training data...");
                progress.setCurrentProgress(-1);
            }
            logger.info("Starting training for classifier: {}", classifierName);
            if (progress != null) progress.log("Starting training for classifier: " + classifierName);

            // Export training data
            Path tempDir = Files.createTempDirectory("dl-training");
            logger.info("Exporting training data to: {}", tempDir);
            if (progress != null) progress.log("Export directory: " + tempDir);

            int patchCount;
            if (selectedImages != null && !selectedImages.isEmpty()) {
                // Multi-image project export
                if (progress != null) {
                    progress.log("Exporting from " + selectedImages.size() + " project images...");
                }
                AnnotationExtractor.ExportResult exportResult = AnnotationExtractor.exportFromProject(
                        selectedImages,
                        trainingConfig.getTileSize(),
                        channelConfig,
                        classNames,
                        tempDir,
                        trainingConfig.getValidationSplit()
                );
                patchCount = exportResult.totalPatches();
            } else {
                // Single-image export
                AnnotationExtractor extractor = new AnnotationExtractor(
                        imageData,
                        trainingConfig.getTileSize(),
                        channelConfig
                );
                AnnotationExtractor.ExportResult exportResult = extractor.exportTrainingData(
                        tempDir, classNames, trainingConfig.getValidationSplit());
                patchCount = exportResult.totalPatches();
            }
            if (progress != null) progress.log("Exported " + patchCount + " training patches");

            if (progress != null && progress.isCancelled()) {
                return new TrainingResult(null, classifierName, 0, 0, 0, false,
                        "Training cancelled by user");
            }

            // Create client and start training
            if (progress != null) progress.setStatus("Connecting to server...");
            ClassifierClient client = new ClassifierClient(
                    DLClassifierPreferences.getServerHost(),
                    DLClassifierPreferences.getServerPort()
            );
            if (progress != null) {
                progress.log("Connected to server at " + DLClassifierPreferences.getServerHost() +
                        ":" + DLClassifierPreferences.getServerPort());
                progress.setStatus("Training model...");
                progress.setOverallProgress(0);
            }

            ClassifierClient.TrainingResult serverResult = client.startTraining(
                    trainingConfig,
                    channelConfig,
                    classNames,
                    tempDir,
                    trainingProgress -> {
                        if (progress != null && progress.isCancelled()) {
                            return;
                        }
                        if (progress != null) {
                            double progressValue = (double) trainingProgress.epoch() / trainingProgress.totalEpochs();
                            progress.setOverallProgress(progressValue);
                            progress.setDetail(String.format("Epoch %d/%d - Loss: %.4f",
                                    trainingProgress.epoch(), trainingProgress.totalEpochs(), trainingProgress.loss()));
                            progress.updateTrainingMetrics(
                                    trainingProgress.epoch(),
                                    trainingProgress.loss(),
                                    trainingProgress.valLoss()
                            );
                            progress.log(String.format("Epoch %d: train_loss=%.4f, val_loss=%.4f",
                                    trainingProgress.epoch(), trainingProgress.loss(), trainingProgress.valLoss()));
                        }
                    },
                    () -> progress != null && progress.isCancelled()
            );

            if (serverResult.isCancelled()) {
                if (progress != null) progress.log("Training cancelled");
                return new TrainingResult(null, classifierName, 0, 0, 0, false,
                        "Training cancelled by user");
            }

            logger.info("Training completed. Model saved to: {}", serverResult.modelPath());
            if (progress != null) progress.log("Training completed. Model path: " + serverResult.modelPath());

            // Build and save metadata
            if (progress != null) progress.setStatus("Saving classifier...");

            String classifierId = classifierName.toLowerCase().replaceAll("[^a-z0-9_-]", "_") +
                    "_" + System.currentTimeMillis();

            List<ClassifierMetadata.ClassInfo> classInfoList = new ArrayList<>();
            for (int i = 0; i < classNames.size(); i++) {
                classInfoList.add(new ClassifierMetadata.ClassInfo(i, classNames.get(i), "#808080"));
            }

            ClassifierMetadata metadata = ClassifierMetadata.builder()
                    .id(classifierId)
                    .name(classifierName)
                    .description(description)
                    .modelType(trainingConfig.getModelType())
                    .backbone(trainingConfig.getBackbone())
                    .inputChannels(channelConfig.getSelectedChannels().size())
                    .expectedChannelNames(channelConfig.getChannelNames())
                    .inputSize(trainingConfig.getTileSize(), trainingConfig.getTileSize())
                    .classes(classInfoList)
                    .normalizationStrategy(channelConfig.getNormalizationStrategy())
                    .bitDepthTrained(channelConfig.getBitDepth())
                    .trainingEpochs(trainingConfig.getEpochs())
                    .finalLoss(serverResult.finalLoss())
                    .finalAccuracy(serverResult.finalAccuracy())
                    .build();

            // Save the classifier
            ModelManager modelManager = new ModelManager();
            modelManager.saveClassifier(metadata, Path.of(serverResult.modelPath()));
            if (progress != null) progress.log("Classifier saved: " + metadata.getId());

            return new TrainingResult(
                    classifierId,
                    classifierName,
                    serverResult.finalLoss(),
                    serverResult.finalAccuracy(),
                    trainingConfig.getEpochs(),
                    true,
                    "Training completed successfully"
            );

        } catch (Exception e) {
            logger.error("Training failed", e);
            if (progress != null) progress.log("ERROR: " + e.getMessage());
            return new TrainingResult(null, classifierName, 0, 0, 0, false,
                    "Training failed: " + e.getMessage());
        }
    }

    /**
     * Executes the training process.
     *
     * @param handler       the classifier handler
     * @param trainingConfig training configuration
     * @param channelConfig channel configuration
     * @param classNames    list of class names
     * @deprecated Use {@link #trainClassifierWithProgress} instead
     */
    @Deprecated
    public void trainClassifier(ClassifierHandler handler,
                                TrainingConfig trainingConfig,
                                ChannelConfiguration channelConfig,
                                List<String> classNames) {
        trainClassifierWithProgress("Untitled", "", handler, trainingConfig, channelConfig, classNames, null);
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
