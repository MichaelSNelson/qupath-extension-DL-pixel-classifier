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

import javafx.geometry.Insets;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
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

    /**
     * Parameters for resuming training.
     *
     * @param totalEpochs new total epoch count
     * @param learningRate new learning rate
     * @param batchSize new batch size
     */
    public record ResumeParams(int totalEpochs, double learningRate, int batchSize) {}

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
        // Check project is open (required for saving classifiers and accessing image data)
        if (qupath.getProject() == null) {
            showError("No Project",
                    "A QuPath project must be open to train a classifier.\n\n" +
                    "Classifiers are saved within the project, and training data\n" +
                    "is exported from project images.\n\n" +
                    "Please create or open a project first.");
            return false;
        }

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
        // Check for unsaved changes before training
        if (!checkUnsavedChanges(selectedImages)) {
            return;
        }

        // Create progress monitor
        ProgressMonitorController progress = ProgressMonitorController.forTraining();
        progress.setOnCancel(v -> logger.info("Training cancellation requested"));
        progress.show();

        // Shared state for pause/resume
        final String[] currentJobId = {null};

        // Wire pause callback
        progress.setOnPause(v -> {
            if (currentJobId[0] != null) {
                try {
                    ClassifierClient client = new ClassifierClient(
                            DLClassifierPreferences.getServerHost(),
                            DLClassifierPreferences.getServerPort());
                    client.pauseTraining(currentJobId[0]);
                } catch (Exception e) {
                    logger.error("Failed to pause training", e);
                    progress.log("ERROR: Failed to pause: " + e.getMessage());
                }
            }
        });

        // Wire resume callback
        progress.setOnResume(v -> {
            CompletableFuture.runAsync(() -> handleResume(
                    currentJobId[0], classifierName, description, handler,
                    trainingConfig, channelConfig, classNames,
                    selectedImages, progress, currentJobId));
        });

        CompletableFuture.runAsync(() -> {
            TrainingResult result = trainCore(classifierName, description, handler,
                    trainingConfig, channelConfig, classNames,
                    qupath.getImageData(), selectedImages, progress, currentJobId);

            if (result.success()) {
                progress.complete(true, String.format(
                        "Classifier trained successfully!\nFinal loss: %.4f\nAccuracy: %.2f%%",
                        result.finalLoss(), result.finalAccuracy() * 100));
            } else if (result.message() != null && result.message().contains("paused")) {
                // Paused state is handled by showPausedState - don't close
                logger.info("Training paused, waiting for user action");
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
        return trainCore(classifierName, description, handler, trainingConfig,
                channelConfig, classNames, imageData, selectedImages, progress, null);
    }

    /**
     * Core training logic shared by GUI and headless paths.
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
     * @param jobIdHolder    optional array to receive the job ID (element 0 is set)
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
                                    ProgressMonitorController progress,
                                    String[] jobIdHolder) {
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
                        trainingConfig.getValidationSplit(),
                        trainingConfig.getLineStrokeWidth()
                );
                patchCount = exportResult.totalPatches();
            } else {
                // Single-image export
                AnnotationExtractor extractor = new AnnotationExtractor(
                        imageData,
                        trainingConfig.getTileSize(),
                        channelConfig,
                        trainingConfig.getLineStrokeWidth()
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

                // Log frozen layer configuration
                List<String> frozen = trainingConfig.getFrozenLayers();
                if (frozen != null && !frozen.isEmpty()) {
                    progress.log("Transfer learning: " + frozen.size() + " layer groups frozen: "
                            + String.join(", ", frozen));
                } else if (trainingConfig.isUsePretrainedWeights()) {
                    progress.log("Transfer learning: pretrained weights loaded, all layers trainable");
                }
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
                            progress.setDetail(String.format("Epoch %d/%d - Loss: %.4f - mIoU: %.4f",
                                    trainingProgress.epoch(), trainingProgress.totalEpochs(),
                                    trainingProgress.loss(), trainingProgress.meanIoU()));
                            progress.updateTrainingMetrics(
                                    trainingProgress.epoch(),
                                    trainingProgress.loss(),
                                    trainingProgress.valLoss(),
                                    trainingProgress.perClassIoU(),
                                    trainingProgress.perClassLoss()
                            );

                            // Log with per-class breakdown
                            StringBuilder logMsg = new StringBuilder();
                            logMsg.append(String.format(
                                    "Epoch %d: train_loss=%.4f, val_loss=%.4f, acc=%.1f%%, mIoU=%.4f",
                                    trainingProgress.epoch(), trainingProgress.loss(),
                                    trainingProgress.valLoss(), trainingProgress.accuracy() * 100,
                                    trainingProgress.meanIoU()));
                            progress.log(logMsg.toString());

                            if (trainingProgress.perClassIoU() != null && !trainingProgress.perClassIoU().isEmpty()) {
                                StringBuilder iouLine = new StringBuilder("  IoU:");
                                for (var entry : trainingProgress.perClassIoU().entrySet()) {
                                    iouLine.append(String.format(" %s=%.3f", entry.getKey(), entry.getValue()));
                                }
                                progress.log(iouLine.toString());
                            }
                            if (trainingProgress.perClassLoss() != null && !trainingProgress.perClassLoss().isEmpty()) {
                                StringBuilder lossLine = new StringBuilder("  Loss:");
                                for (var entry : trainingProgress.perClassLoss().entrySet()) {
                                    lossLine.append(String.format(" %s=%.4f", entry.getKey(), entry.getValue()));
                                }
                                progress.log(lossLine.toString());
                            }
                        }
                    },
                    () -> progress != null && progress.isCancelled(),
                    jobId -> {
                        if (jobIdHolder != null && jobIdHolder.length > 0) {
                            jobIdHolder[0] = jobId;
                        }
                    }
            );

            if (serverResult.isPaused()) {
                if (progress != null) {
                    progress.log("Training paused at epoch " + serverResult.lastEpoch());
                    progress.showPausedState(serverResult.lastEpoch(), serverResult.totalEpochs());
                }
                return new TrainingResult(null, classifierName, 0, 0, 0, false,
                        "Training paused at epoch " + serverResult.lastEpoch());
            }

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
     * Handles the resume flow after training has been paused.
     * <p>
     * This method runs on a background thread and uses Platform.runLater
     * for any UI interactions (dialogs, unsaved changes check).
     */
    private void handleResume(String jobId,
                              String classifierName,
                              String description,
                              ClassifierHandler handler,
                              TrainingConfig trainingConfig,
                              ChannelConfiguration channelConfig,
                              List<String> classNames,
                              List<ProjectImageEntry<BufferedImage>> selectedImages,
                              ProgressMonitorController progress,
                              String[] currentJobId) {
        try {
            // 1. Check for unsaved changes (on FX thread)
            CompletableFuture<Boolean> unsavedCheck = new CompletableFuture<>();
            Platform.runLater(() -> {
                boolean proceed = checkUnsavedChanges(selectedImages);
                unsavedCheck.complete(proceed);
            });
            if (!unsavedCheck.get()) {
                // User cancelled -- stay in paused state
                return;
            }

            // 2. Show resume param dialog (on FX thread)
            CompletableFuture<Optional<ResumeParams>> paramsFuture = new CompletableFuture<>();
            Platform.runLater(() -> {
                Optional<ResumeParams> params = showResumeParamDialog(
                        trainingConfig.getEpochs(),
                        trainingConfig.getLearningRate(),
                        trainingConfig.getBatchSize());
                paramsFuture.complete(params);
            });
            Optional<ResumeParams> paramsOpt = paramsFuture.get();
            if (paramsOpt.isEmpty()) {
                // User cancelled dialog -- stay in paused state
                return;
            }
            ResumeParams params = paramsOpt.get();

            // 3. Re-export training data
            progress.showResumedState();
            progress.setStatus("Re-exporting training data...");
            progress.log("Re-exporting annotations (includes any new/modified annotations)...");

            Path tempDir = Files.createTempDirectory("dl-training-resume");
            int patchCount;
            if (selectedImages != null && !selectedImages.isEmpty()) {
                AnnotationExtractor.ExportResult exportResult = AnnotationExtractor.exportFromProject(
                        selectedImages,
                        trainingConfig.getTileSize(),
                        channelConfig,
                        classNames,
                        tempDir,
                        trainingConfig.getValidationSplit(),
                        trainingConfig.getLineStrokeWidth()
                );
                patchCount = exportResult.totalPatches();
            } else {
                ImageData<BufferedImage> imageData = qupath.getImageData();
                AnnotationExtractor extractor = new AnnotationExtractor(
                        imageData, trainingConfig.getTileSize(), channelConfig,
                        trainingConfig.getLineStrokeWidth());
                AnnotationExtractor.ExportResult exportResult = extractor.exportTrainingData(
                        tempDir, classNames, trainingConfig.getValidationSplit());
                patchCount = exportResult.totalPatches();
            }
            progress.log("Re-exported " + patchCount + " training patches");
            progress.setStatus("Resuming training...");

            // Log frozen layer configuration
            List<String> frozen = trainingConfig.getFrozenLayers();
            if (frozen != null && !frozen.isEmpty()) {
                progress.log("Transfer learning: " + frozen.size() + " layer groups frozen: "
                        + String.join(", ", frozen));
            } else if (trainingConfig.isUsePretrainedWeights()) {
                progress.log("Transfer learning: pretrained weights loaded, all layers trainable");
            }

            // 4. Call client.resumeTraining
            ClassifierClient client = new ClassifierClient(
                    DLClassifierPreferences.getServerHost(),
                    DLClassifierPreferences.getServerPort());

            ClassifierClient.TrainingResult serverResult = client.resumeTraining(
                    jobId,
                    tempDir,
                    params.totalEpochs(),
                    params.learningRate(),
                    params.batchSize(),
                    trainingProgress -> {
                        if (progress.isCancelled()) return;
                        double progressValue = (double) trainingProgress.epoch() / trainingProgress.totalEpochs();
                        progress.setOverallProgress(progressValue);
                        progress.setDetail(String.format("Epoch %d/%d - Loss: %.4f - mIoU: %.4f",
                                trainingProgress.epoch(), trainingProgress.totalEpochs(),
                                trainingProgress.loss(), trainingProgress.meanIoU()));
                        progress.updateTrainingMetrics(
                                trainingProgress.epoch(),
                                trainingProgress.loss(),
                                trainingProgress.valLoss(),
                                trainingProgress.perClassIoU(),
                                trainingProgress.perClassLoss());

                        StringBuilder logMsg = new StringBuilder();
                        logMsg.append(String.format(
                                "Epoch %d: train_loss=%.4f, val_loss=%.4f, acc=%.1f%%, mIoU=%.4f",
                                trainingProgress.epoch(), trainingProgress.loss(),
                                trainingProgress.valLoss(), trainingProgress.accuracy() * 100,
                                trainingProgress.meanIoU()));
                        progress.log(logMsg.toString());

                        if (trainingProgress.perClassIoU() != null && !trainingProgress.perClassIoU().isEmpty()) {
                            StringBuilder iouLine = new StringBuilder("  IoU:");
                            for (var entry : trainingProgress.perClassIoU().entrySet()) {
                                iouLine.append(String.format(" %s=%.3f", entry.getKey(), entry.getValue()));
                            }
                            progress.log(iouLine.toString());
                        }
                        if (trainingProgress.perClassLoss() != null && !trainingProgress.perClassLoss().isEmpty()) {
                            StringBuilder lossLine = new StringBuilder("  Loss:");
                            for (var entry : trainingProgress.perClassLoss().entrySet()) {
                                lossLine.append(String.format(" %s=%.4f", entry.getKey(), entry.getValue()));
                            }
                            progress.log(lossLine.toString());
                        }
                    },
                    progress::isCancelled
            );

            // 5. Handle result -- may be paused again, completed, or cancelled
            if (serverResult.isPaused()) {
                progress.log("Training paused again at epoch " + serverResult.lastEpoch());
                progress.showPausedState(serverResult.lastEpoch(), serverResult.totalEpochs());
                currentJobId[0] = serverResult.jobId();
            } else if (serverResult.isCancelled()) {
                progress.complete(false, "Training cancelled by user");
            } else {
                // Completed -- save the classifier
                progress.setStatus("Saving classifier...");
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
                        .trainingEpochs(params.totalEpochs())
                        .finalLoss(serverResult.finalLoss())
                        .finalAccuracy(serverResult.finalAccuracy())
                        .build();

                ModelManager modelManager = new ModelManager();
                modelManager.saveClassifier(metadata, Path.of(serverResult.modelPath()));
                progress.log("Classifier saved: " + metadata.getId());

                progress.complete(true, String.format(
                        "Classifier trained successfully!\nFinal loss: %.4f\nAccuracy: %.2f%%",
                        serverResult.finalLoss(), serverResult.finalAccuracy() * 100));
            }

        } catch (Exception e) {
            logger.error("Resume failed", e);
            progress.log("ERROR: Resume failed: " + e.getMessage());
            progress.complete(false, "Resume failed: " + e.getMessage());
        }
    }

    /**
     * Shows a dialog to configure resume parameters.
     *
     * @param currentEpochs current total epochs setting
     * @param currentLR     current learning rate
     * @param currentBatch  current batch size
     * @return optional resume params, or empty if user cancelled
     */
    private Optional<ResumeParams> showResumeParamDialog(int currentEpochs,
                                                          double currentLR,
                                                          int currentBatch) {
        Dialog<ResumeParams> dialog = new Dialog<>();
        dialog.setTitle("Resume Training");
        dialog.setHeaderText("Adjust parameters for resumed training");
        dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new Insets(20));

        Spinner<Integer> epochSpinner = new Spinner<>(
                new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 1000, currentEpochs));
        epochSpinner.setEditable(true);

        TextField lrField = new TextField(String.valueOf(currentLR));
        lrField.setPrefWidth(120);

        Spinner<Integer> batchSpinner = new Spinner<>(
                new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 128, currentBatch));
        batchSpinner.setEditable(true);

        grid.add(new Label("Total Epochs:"), 0, 0);
        grid.add(epochSpinner, 1, 0);
        grid.add(new Label("Learning Rate:"), 0, 1);
        grid.add(lrField, 1, 1);
        grid.add(new Label("Batch Size:"), 0, 2);
        grid.add(batchSpinner, 1, 2);

        dialog.getDialogPane().setContent(grid);

        dialog.setResultConverter(buttonType -> {
            if (buttonType == ButtonType.OK) {
                try {
                    double lr = Double.parseDouble(lrField.getText().trim());
                    return new ResumeParams(epochSpinner.getValue(), lr, batchSpinner.getValue());
                } catch (NumberFormatException e) {
                    logger.warn("Invalid learning rate value: {}", lrField.getText());
                    return null;
                }
            }
            return null;
        });

        return dialog.showAndWait();
    }

    /**
     * Checks for unsaved changes in the current image and warns the user.
     * <p>
     * In single-image mode, training uses the live in-memory ImageData, so
     * unsaved annotations are included. In multi-image mode, training reads
     * from saved .qpdata files on disk, so unsaved changes would be missed.
     * <p>
     * This method must be called on the JavaFX Application Thread.
     *
     * @param selectedImages the selected project images, or null for single-image mode
     * @return true if training should proceed, false if user cancelled
     */
    private boolean checkUnsavedChanges(List<ProjectImageEntry<BufferedImage>> selectedImages) {
        ImageData<BufferedImage> currentImageData = qupath.getImageData();
        if (currentImageData == null) {
            return true;
        }

        boolean isMultiImage = selectedImages != null && !selectedImages.isEmpty();
        boolean hasUnsavedChanges = currentImageData.isChanged();

        if (!hasUnsavedChanges) {
            return true;
        }

        if (isMultiImage) {
            // Multi-image mode reads from saved .qpdata files -- unsaved changes are missed
            return Dialogs.showConfirmDialog(
                    "Unsaved Changes",
                    "The current image has unsaved annotation changes.\n\n" +
                    "Multi-image training reads from saved project data.\n" +
                    "Unsaved changes in the current image will NOT be\n" +
                    "included in training.\n\n" +
                    "Save your changes first (File -> Save) or click OK\n" +
                    "to continue without the unsaved changes."
            );
        } else {
            // Single-image mode uses live in-memory data -- unsaved annotations are included
            logger.info("Current image has unsaved changes - these will be included in single-image training");
            return true;
        }
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
