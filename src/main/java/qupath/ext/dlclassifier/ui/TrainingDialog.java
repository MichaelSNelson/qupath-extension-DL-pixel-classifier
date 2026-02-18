package qupath.ext.dlclassifier.ui;

import javafx.animation.PauseTransition;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.event.ActionEvent;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.input.Clipboard;
import javafx.scene.input.ClipboardContent;
import javafx.scene.chart.PieChart;
import javafx.scene.layout.*;
import javafx.stage.Modality;
import javafx.util.Duration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.classifier.ClassifierHandler;
import qupath.ext.dlclassifier.classifier.ClassifierRegistry;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.TrainingConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.scripting.ScriptGenerator;
import qupath.ext.dlclassifier.service.BackendFactory;
import qupath.ext.dlclassifier.service.ClassifierBackend;
import qupath.lib.gui.QuPathGUI;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.scripting.QP;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

/**
 * Dialog for configuring deep learning classifier training.
 * <p>
 * This dialog provides a comprehensive interface for:
 * <ul>
 *   <li>Classifier naming and description</li>
 *   <li>Model architecture selection</li>
 *   <li>Training hyperparameter configuration</li>
 *   <li>Channel selection for multi-channel images</li>
 *   <li>Annotation class selection for training data</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class TrainingDialog {

    private static final Logger logger = LoggerFactory.getLogger(TrainingDialog.class);

    /**
     * Result of the training dialog.
     *
     * @param classifierName  name for the classifier
     * @param description     classifier description
     * @param trainingConfig  training configuration
     * @param channelConfig   channel configuration
     * @param selectedClasses selected class names
     * @param selectedImages  project images to train from, or null for current image only
     * @param classColors     map of class name to packed RGB color (from QuPath PathClass)
     */
    public record TrainingDialogResult(
            String classifierName,
            String description,
            TrainingConfig trainingConfig,
            ChannelConfiguration channelConfig,
            List<String> selectedClasses,
            List<ProjectImageEntry<BufferedImage>> selectedImages,
            Map<String, Integer> classColors
    ) {
        /** Returns true if training should use multiple project images. */
        public boolean isMultiImage() {
            return selectedImages != null && !selectedImages.isEmpty();
        }
    }

    private TrainingDialog() {
        // Utility class
    }

    /**
     * Shows the training configuration dialog.
     *
     * @return CompletableFuture with the result, or cancelled if user cancels
     */
    public static CompletableFuture<TrainingDialogResult> showDialog() {
        CompletableFuture<TrainingDialogResult> future = new CompletableFuture<>();

        Platform.runLater(() -> {
            try {
                TrainingDialogBuilder builder = new TrainingDialogBuilder();
                Optional<TrainingDialogResult> result = builder.buildAndShow();
                if (result.isPresent()) {
                    future.complete(result.get());
                } else {
                    future.cancel(true);
                }
            } catch (Exception e) {
                logger.error("Error showing training dialog", e);
                future.completeExceptionally(e);
            }
        });

        return future;
    }

    /**
     * Inner builder class for constructing the dialog.
     */
    private static class TrainingDialogBuilder {

        private Dialog<TrainingDialogResult> dialog;
        private final Map<String, String> validationErrors = new LinkedHashMap<>();

        // Basic info fields
        private TextField classifierNameField;
        private TextArea descriptionField;

        // Model architecture
        private ComboBox<String> architectureCombo;
        private ComboBox<String> backboneCombo;

        // Training parameters
        private Spinner<Integer> epochsSpinner;
        private Spinner<Integer> batchSizeSpinner;
        private Spinner<Double> learningRateSpinner;
        private Spinner<Integer> validationSplitSpinner;

        // Tiling parameters
        private Spinner<Integer> tileSizeSpinner;
        private Spinner<Integer> overlapSpinner;
        private ComboBox<String> downsampleCombo;
        private ComboBox<String> contextScaleCombo;
        private Spinner<Integer> lineStrokeWidthSpinner;

        // Channel selection
        private ChannelSelectionPanel channelPanel;

        // Class selection
        private ListView<ClassItem> classListView;
        private PieChart classDistributionChart;

        // Augmentation
        private CheckBox flipHorizontalCheck;
        private CheckBox flipVerticalCheck;
        private CheckBox rotationCheck;
        private CheckBox colorJitterCheck;
        private CheckBox elasticCheck;

        // Training strategy
        private ComboBox<String> schedulerCombo;
        private ComboBox<String> lossFunctionCombo;
        private ComboBox<String> earlyStoppingMetricCombo;
        private Spinner<Integer> earlyStoppingPatienceSpinner;
        private CheckBox mixedPrecisionCheck;

        // Transfer learning
        private LayerFreezePanel layerFreezePanel;
        private CheckBox usePretrainedCheck;
        private ClassifierBackend backend;

        // Image source selection
        private ListView<ImageSelectionItem> imageSelectionList;
        private List<TitledPane> gatedSections = new ArrayList<>();
        private boolean classesLoaded = false;
        private Button loadClassesButton;

        // Error display
        private VBox errorSummaryPanel;
        private VBox errorListBox;
        private Button okButton;

        public Optional<TrainingDialogResult> buildAndShow() {
            dialog = new Dialog<>();
            dialog.initModality(Modality.APPLICATION_MODAL);
            dialog.setTitle("Train DL Pixel Classifier");
            dialog.setResizable(true);

            // Create header
            createHeader();

            // Create button types
            ButtonType trainType = new ButtonType("Start Training", ButtonBar.ButtonData.OK_DONE);
            ButtonType cancelType = new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE);
            ButtonType copyScriptType = new ButtonType("Copy as Script", ButtonBar.ButtonData.LEFT);
            dialog.getDialogPane().getButtonTypes().addAll(copyScriptType, trainType, cancelType);

            okButton = (Button) dialog.getDialogPane().lookupButton(trainType);
            okButton.setDisable(true);

            // Wire up the "Copy as Script" button
            Button copyScriptButton = (Button) dialog.getDialogPane().lookupButton(copyScriptType);
            copyScriptButton.addEventFilter(ActionEvent.ACTION, event -> {
                event.consume(); // Prevent dialog from closing
                copyTrainingScript(copyScriptButton);
            });

            // Create content
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            // Initialize backend for server communication
            try {
                backend = BackendFactory.getBackend();
            } catch (Exception e) {
                logger.warn("Could not initialize classifier backend: {}", e.getMessage());
            }

            // Create channel and class sections first so their fields exist
            // before the model section's backbone listener fires
            TitledPane channelSection = createChannelSection();
            TitledPane classSection = createClassSection();

            // Image source section is always enabled
            TitledPane imageSourceSection = createImageSourceSection();

            // All other sections are gated behind "Load Classes"
            TitledPane basicInfoSection = createBasicInfoSection();
            TitledPane modelSection = createModelSection();
            TitledPane transferSection = createTransferLearningSection();
            TitledPane trainingSection = createTrainingSection();
            TitledPane strategySection = createTrainingStrategySection();
            TitledPane augmentationSection = createAugmentationSection();

            gatedSections.addAll(List.of(
                    basicInfoSection, modelSection, transferSection,
                    trainingSection, strategySection,
                    channelSection, classSection, augmentationSection
            ));

            // Build layout: image source first, then gated sections, then error panel
            content.getChildren().addAll(
                    imageSourceSection,
                    basicInfoSection,
                    modelSection,
                    transferSection,
                    trainingSection,
                    strategySection,
                    channelSection,
                    classSection,
                    augmentationSection,
                    createErrorSummaryPanel()
            );

            // Disable gated sections until classes are loaded
            setGatedSectionsEnabled(false);

            ScrollPane scrollPane = new ScrollPane(content);
            scrollPane.setFitToWidth(true);
            scrollPane.setPrefHeight(600);
            scrollPane.setPrefWidth(550);

            dialog.getDialogPane().setContent(scrollPane);

            // Generate default classifier name
            String timestamp = java.time.LocalDate.now().toString().replace("-", "");
            classifierNameField.setText("Classifier_" + timestamp);

            // Trigger initial layer load now that all UI components exist
            updateLayerFreezePanel();

            // Initial validation
            updateValidation();

            // Set result converter
            dialog.setResultConverter(button -> {
                if (button != trainType) {
                    return null;
                }
                return buildResult();
            });

            return dialog.showAndWait();
        }

        private void createHeader() {
            VBox headerBox = new VBox(5);
            headerBox.setPadding(new Insets(10));

            Label titleLabel = new Label("Configure Classifier Training");
            titleLabel.setStyle("-fx-font-size: 14px; -fx-font-weight: bold;");

            Label subtitleLabel = new Label("Train a deep learning model to classify pixels in your images");
            subtitleLabel.setStyle("-fx-text-fill: #666;");

            headerBox.getChildren().addAll(titleLabel, subtitleLabel, new Separator());
            dialog.getDialogPane().setHeader(headerBox);
        }

        private TitledPane createBasicInfoSection() {
            GridPane grid = new GridPane();
            grid.setHgap(10);
            grid.setVgap(8);
            grid.setPadding(new Insets(10));

            ColumnConstraints labelCol = new ColumnConstraints();
            labelCol.setMinWidth(140);
            labelCol.setPrefWidth(150);
            ColumnConstraints fieldCol = new ColumnConstraints();
            fieldCol.setHgrow(Priority.ALWAYS);
            grid.getColumnConstraints().addAll(labelCol, fieldCol);

            int row = 0;

            // Classifier name
            classifierNameField = new TextField();
            classifierNameField.setPromptText("e.g., Collagen_Classifier_v1");
            classifierNameField.setPrefWidth(300);
            TooltipHelper.install(classifierNameField,
                    "Unique identifier for this classifier.\n" +
                    "Used as the filename when saving.\n" +
                    "Only letters, numbers, underscore, and hyphen allowed.");
            classifierNameField.textProperty().addListener((obs, old, newVal) -> validateClassifierName(newVal));

            grid.add(new Label("Classifier Name:"), 0, row);
            grid.add(classifierNameField, 1, row);
            row++;

            // Description
            descriptionField = new TextArea();
            descriptionField.setPromptText("Optional description of what this classifier detects...");
            descriptionField.setPrefRowCount(2);
            descriptionField.setWrapText(true);
            TooltipHelper.install(descriptionField,
                    "Optional free-text description of what this classifier detects.\n" +
                    "Stored in classifier metadata for documentation.\n" +
                    "Example: 'Collagen vs. epithelium in H&E stained liver sections'");

            grid.add(new Label("Description:"), 0, row);
            grid.add(descriptionField, 1, row);

            TitledPane pane = new TitledPane("CLASSIFIER INFO", grid);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Basic identification for the trained classifier model"));
            return pane;
        }

        private TitledPane createImageSourceSection() {
            VBox content = new VBox(8);
            content.setPadding(new Insets(10));

            Label info = new Label("Select project images to include in training:");
            info.setStyle("-fx-text-fill: #666;");

            imageSelectionList = new ListView<>();
            imageSelectionList.setCellFactory(lv -> new CheckBoxListCell<>(
                    item -> item.selected,
                    item -> item.imageName
            ));
            imageSelectionList.setPrefHeight(150);
            TooltipHelper.install(imageSelectionList,
                    "Check the project images to include in training.\n" +
                    "Only images with classified annotations are shown.\n" +
                    "Patches from all selected images are combined into one training set.");

            // Populate project images that have classified annotations
            Project<BufferedImage> project = QuPathGUI.getInstance().getProject();
            if (project != null) {
                for (ProjectImageEntry<BufferedImage> entry : project.getImageList()) {
                    try {
                        ImageData<BufferedImage> data = entry.readImageData();
                        long annotationCount = data.getHierarchy().getAnnotationObjects().stream()
                                .filter(a -> a.getPathClass() != null)
                                .count();
                        data.getServer().close();
                        if (annotationCount > 0) {
                            ImageSelectionItem item = new ImageSelectionItem(entry, annotationCount);
                            // When image selection changes, update button state and mark classes stale
                            item.selected.addListener((obs, old, newVal) -> {
                                updateLoadClassesButtonState();
                                if (classesLoaded) {
                                    markClassesStale();
                                }
                            });
                            imageSelectionList.getItems().add(item);
                        }
                    } catch (Exception e) {
                        logger.debug("Could not read image '{}': {}",
                                entry.getImageName(), e.getMessage());
                    }
                }
            }

            // Select All / Select None buttons
            Button selectAllImagesBtn = new Button("Select All");
            TooltipHelper.install(selectAllImagesBtn, "Select all project images for training");
            selectAllImagesBtn.setOnAction(e ->
                    imageSelectionList.getItems().forEach(item -> item.selected.set(true)));

            Button selectNoneImagesBtn = new Button("Select None");
            TooltipHelper.install(selectNoneImagesBtn, "Deselect all project images");
            selectNoneImagesBtn.setOnAction(e ->
                    imageSelectionList.getItems().forEach(item -> item.selected.set(false)));

            // Load Classes button
            loadClassesButton = new Button("Load Classes from Selected Images");
            loadClassesButton.setStyle("-fx-font-weight: bold;");
            loadClassesButton.setMaxWidth(Double.MAX_VALUE);
            TooltipHelper.install(loadClassesButton,
                    "Read annotations from the selected images and populate\n" +
                    "the class list with the union of all classes found.\n" +
                    "Also initializes channel configuration from the first image.");
            loadClassesButton.setOnAction(e -> loadClassesFromSelectedImages());

            HBox imageButtonBox = new HBox(10, selectAllImagesBtn, selectNoneImagesBtn);

            content.getChildren().addAll(info, imageSelectionList, imageButtonBox, loadClassesButton);

            // Show a message if no annotated images found
            if (imageSelectionList.getItems().isEmpty()) {
                Label noImagesLabel = new Label("No project images with classified annotations found.");
                noImagesLabel.setStyle("-fx-text-fill: #cc6600; -fx-font-style: italic;");
                content.getChildren().add(1, noImagesLabel);
            }

            // Initialize button state
            updateLoadClassesButtonState();

            TitledPane pane = new TitledPane("TRAINING DATA SOURCE", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Select project images and load classes for training"));
            return pane;
        }

        private TitledPane createModelSection() {
            GridPane grid = new GridPane();
            grid.setHgap(10);
            grid.setVgap(8);
            grid.setPadding(new Insets(10));

            ColumnConstraints labelCol = new ColumnConstraints();
            labelCol.setMinWidth(140);
            labelCol.setPrefWidth(150);
            ColumnConstraints fieldCol = new ColumnConstraints();
            fieldCol.setHgrow(Priority.ALWAYS);
            grid.getColumnConstraints().addAll(labelCol, fieldCol);

            int row = 0;

            // Architecture selection
            List<String> architectures = new ArrayList<>(ClassifierRegistry.getAllTypes());
            architectureCombo = new ComboBox<>(FXCollections.observableArrayList(architectures));
            // Restore last used architecture from preferences, falling back to first in list
            String savedArchitecture = DLClassifierPreferences.getLastArchitecture();
            if (architectures.contains(savedArchitecture)) {
                architectureCombo.setValue(savedArchitecture);
            } else {
                architectureCombo.setValue(architectures.isEmpty() ? "unet" : architectures.get(0));
            }
            TooltipHelper.installWithLink(architectureCombo,
                    "Segmentation architecture:\n\n" +
                    "UNet: Symmetric encoder-decoder with skip connections.\n" +
                    "  Best general-purpose choice. Good default for most tasks.\n\n" +
                    "DeepLabV3+: Atrous convolutions for multi-scale context.\n" +
                    "  Better for large structures that span the entire tile.\n\n" +
                    "FPN: Feature pyramid for objects at varying scales.\n" +
                    "  Good when structures vary significantly in size.\n\n" +
                    "PSPNet: Pyramid pooling for global context.",
                    "https://arxiv.org/abs/1505.04597");
            architectureCombo.valueProperty().addListener((obs, old, newVal) -> updateBackboneOptions(newVal));

            grid.add(new Label("Architecture:"), 0, row);
            grid.add(architectureCombo, 1, row);
            row++;

            // Backbone selection
            backboneCombo = new ComboBox<>();
            TooltipHelper.installWithLink(backboneCombo,
                    "Pretrained encoder network that extracts features:\n\n" +
                    "resnet34: Good balance of speed and accuracy. Best default.\n" +
                    "resnet50: Deeper, more capacity, slower. For complex tasks.\n" +
                    "efficientnet-b0: Lightweight, fast inference.\n\n" +
                    "Histology encoders (marked 'Histology') use weights pretrained\n" +
                    "on millions of tissue patches instead of ImageNet. They provide\n" +
                    "better feature extraction for tissue classification and require\n" +
                    "less layer freezing. ~100MB download on first use (cached).",
                    "https://smp.readthedocs.io/en/latest/encoders.html");
            updateBackboneOptions(architectureCombo.getValue());

            grid.add(new Label("Encoder:"), 0, row);
            grid.add(backboneCombo, 1, row);

            TitledPane pane = new TitledPane("MODEL ARCHITECTURE", grid);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Select the neural network architecture and encoder"));
            return pane;
        }

        private TitledPane createTransferLearningSection() {
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            // Use pretrained checkbox
            usePretrainedCheck = new CheckBox("Use pretrained weights");
            usePretrainedCheck.setSelected(true);
            TooltipHelper.installWithLink(usePretrainedCheck,
                    "Initialize encoder with pretrained weights.\n" +
                    "Dramatically improves convergence and final accuracy,\n" +
                    "especially with limited training data. Recommended for\n" +
                    "virtually all use cases.\n\n" +
                    "Without pretrained weights, the model starts from random\n" +
                    "initialization and requires much more data and epochs.",
                    "https://cs231n.github.io/transfer-learning/");

            Label infoLabel = new Label(
                    "Transfer learning uses pretrained weights from ImageNet. " +
                    "Freeze early layers to preserve general features, train later layers to adapt to your data."
            );
            infoLabel.setWrapText(true);
            infoLabel.setStyle("-fx-text-fill: #666666; -fx-font-size: 11px;");

            // Update info label dynamically when backbone changes
            backboneCombo.valueProperty().addListener((obs, old, newVal) -> {
                if (newVal != null && newVal.contains("_")) {
                    // Histology encoder selected
                    infoLabel.setText(
                            "This backbone uses histology-pretrained weights (tissue patches). " +
                            "Features are already tissue-relevant, so less freezing is needed. " +
                            "~100MB download on first use (cached for future runs).");
                    usePretrainedCheck.setText("Use histology pretrained weights");
                } else {
                    infoLabel.setText(
                            "Transfer learning uses pretrained weights from ImageNet. " +
                            "Freeze early layers to preserve general features, train later layers to adapt to your data.");
                    usePretrainedCheck.setText("Use ImageNet pretrained weights");
                }
            });

            // Layer freeze panel
            layerFreezePanel = new LayerFreezePanel();
            layerFreezePanel.setBackend(backend);

            // Bind visibility to pretrained checkbox
            layerFreezePanel.disableProperty().bind(usePretrainedCheck.selectedProperty().not());

            // Update layers when architecture/backbone changes
            usePretrainedCheck.selectedProperty().addListener((obs, old, newVal) -> {
                if (newVal) {
                    updateLayerFreezePanel();
                }
            });
            backboneCombo.valueProperty().addListener((obs, old, newVal) -> updateLayerFreezePanel());

            content.getChildren().addAll(usePretrainedCheck, infoLabel, layerFreezePanel);

            TitledPane pane = new TitledPane("TRANSFER LEARNING", content);
            pane.setExpanded(false); // Collapsed by default - advanced option
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Configure pretrained weight usage and layer freezing strategy"));
            return pane;
        }

        private void updateLayerFreezePanel() {
            if (layerFreezePanel == null || !usePretrainedCheck.isSelected()) return;

            String architecture = architectureCombo.getValue();
            String encoder = backboneCombo.getValue();

            if (architecture != null && encoder != null && channelPanel != null) {
                // Guard: channel panel may not have channels selected yet during init
                if (!channelPanel.isValid()) return;

                try {
                    int numChannels = channelPanel.getChannelConfiguration().getNumChannels();
                    int numClasses = (int) classListView.getItems().stream()
                            .filter(item -> item.selected().get())
                            .count();
                    if (numClasses < 2) numClasses = 2;

                    // Load layers asynchronously to avoid blocking the FX thread on HTTP call
                    final int ch = numChannels;
                    final int cls = numClasses;
                    CompletableFuture.runAsync(() -> {
                        layerFreezePanel.loadLayers(architecture, encoder, ch, cls);
                    });
                } catch (Exception e) {
                    logger.warn("Could not update layer freeze panel: {}", e.getMessage());
                }
            }
        }

        private TitledPane createTrainingSection() {
            GridPane grid = new GridPane();
            grid.setHgap(10);
            grid.setVgap(8);
            grid.setPadding(new Insets(10));

            ColumnConstraints labelCol = new ColumnConstraints();
            labelCol.setMinWidth(140);
            labelCol.setPrefWidth(150);
            ColumnConstraints fieldCol = new ColumnConstraints();
            fieldCol.setHgrow(Priority.ALWAYS);
            grid.getColumnConstraints().addAll(labelCol, fieldCol);

            int row = 0;

            // Epochs
            epochsSpinner = new Spinner<>(1, 1000, DLClassifierPreferences.getDefaultEpochs(), 10);
            epochsSpinner.setEditable(true);
            epochsSpinner.setPrefWidth(100);
            TooltipHelper.install(epochsSpinner,
                    "Number of complete passes through the training data.\n" +
                    "More epochs allow the model to learn more but risk overfitting.\n" +
                    "Watch validation loss to determine when to stop.\n\n" +
                    "Typical range: 50-200 for small datasets, 20-100 for large.\n" +
                    "Early stopping will halt training automatically if the model\n" +
                    "stops improving, so it is safe to set a high value.");

            grid.add(new Label("Epochs:"), 0, row);
            grid.add(epochsSpinner, 1, row);
            row++;

            // Batch size
            batchSizeSpinner = new Spinner<>(1, 128, DLClassifierPreferences.getDefaultBatchSize(), 4);
            batchSizeSpinner.setEditable(true);
            batchSizeSpinner.setPrefWidth(100);
            TooltipHelper.install(batchSizeSpinner,
                    "Number of tiles processed together in each training step.\n" +
                    "Larger batches give more stable gradients but use more GPU memory.\n" +
                    "Reduce if you get CUDA out-of-memory errors.\n\n" +
                    "Typical: 4-16 depending on tile size and GPU VRAM.\n" +
                    "With 512px tiles: 4-8 for 8GB VRAM, 8-16 for 12+ GB VRAM.\n" +
                    "With 256px tiles: double the above batch sizes.");

            grid.add(new Label("Batch Size:"), 0, row);
            grid.add(batchSizeSpinner, 1, row);
            row++;

            // Learning rate
            learningRateSpinner = new Spinner<>(0.00001, 1.0, DLClassifierPreferences.getDefaultLearningRate(), 0.0001);
            learningRateSpinner.setEditable(true);
            learningRateSpinner.setPrefWidth(100);
            // Default StringConverter rounds 0.001 to "0.0" - use enough decimal places
            var lrFactory = (SpinnerValueFactory.DoubleSpinnerValueFactory) learningRateSpinner.getValueFactory();
            lrFactory.setConverter(new javafx.util.StringConverter<Double>() {
                @Override
                public String toString(Double value) {
                    return value == null ? "" : String.format("%.5f", value);
                }
                @Override
                public Double fromString(String string) {
                    try {
                        return Double.parseDouble(string.trim());
                    } catch (NumberFormatException e) {
                        return lrFactory.getValue();
                    }
                }
            });
            TooltipHelper.install(learningRateSpinner,
                    "Controls the step size during gradient descent.\n" +
                    "Too high: training diverges. Too low: training stalls.\n\n" +
                    "Default 1e-3 (0.001) is a safe starting point for Adam optimizer.\n" +
                    "Reduce to 1e-4 if loss oscillates wildly.\n" +
                    "Use 1e-5 when fine-tuning all layers (no freezing).\n" +
                    "The LR scheduler will adjust the rate during training.");

            grid.add(new Label("Learning Rate:"), 0, row);
            grid.add(learningRateSpinner, 1, row);
            row++;

            // Validation split
            validationSplitSpinner = new Spinner<>(5, 50, DLClassifierPreferences.getValidationSplit(), 5);
            validationSplitSpinner.setEditable(true);
            validationSplitSpinner.setPrefWidth(100);
            TooltipHelper.install(validationSplitSpinner,
                    "Percentage of annotated tiles held out for validation.\n" +
                    "Used to monitor overfitting during training.\n" +
                    "Higher values give more reliable validation metrics\n" +
                    "but leave less data for training.\n\n" +
                    "15-25% is typical. Use 10% for very small datasets.\n" +
                    "Use 25-30% for large datasets where you can afford it.");

            grid.add(new Label("Validation Split (%):"), 0, row);
            grid.add(validationSplitSpinner, 1, row);
            row++;

            // Tile size
            tileSizeSpinner = new Spinner<>(64, 1024, DLClassifierPreferences.getTileSize(), 64);
            tileSizeSpinner.setEditable(true);
            tileSizeSpinner.setPrefWidth(100);
            TooltipHelper.install(tileSizeSpinner,
                    "Size of square patches extracted from annotations for training.\n" +
                    "Must be divisible by 32 (encoder downsampling requirement).\n" +
                    "Larger tiles capture more context but use more memory.\n\n" +
                    "256: Good for cell-level features. Faster training.\n" +
                    "512: Good balance of context and memory. Recommended default.\n" +
                    "1024: Maximum context but requires large GPU VRAM.");

            grid.add(new Label("Tile Size:"), 0, row);
            grid.add(tileSizeSpinner, 1, row);
            row++;

            // Downsample
            downsampleCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "1x (Full resolution)",
                    "2x (Half resolution)",
                    "4x (Quarter resolution)",
                    "8x (1/8 resolution)"
            ));
            downsampleCombo.setValue("1x (Full resolution)");
            TooltipHelper.install(downsampleCombo,
                    "Controls image resolution for training.\n" +
                    "Higher downsample = more spatial context per tile but less detail.\n\n" +
                    "1x: Full resolution -- best for cell-level features.\n" +
                    "2x: Half resolution -- good for tissue structures.\n" +
                    "4x: Quarter resolution -- each 512px tile covers 2048px of tissue.\n" +
                    "8x: Low resolution -- for large-scale region classification.\n\n" +
                    "Must match at inference time for consistent results.");

            grid.add(new Label("Resolution:"), 0, row);
            grid.add(downsampleCombo, 1, row);
            row++;

            // Context scale
            contextScaleCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "None (single scale)",
                    "2x context",
                    "4x context (Recommended)",
                    "8x context"
            ));
            contextScaleCombo.setValue("None (single scale)");
            TooltipHelper.install(contextScaleCombo,
                    "Multi-scale context feeds the model two views of each location:\n" +
                    "the full-resolution tile for detail, plus a larger surrounding\n" +
                    "region (downsampled to the same pixel size) for spatial context.\n\n" +
                    "None: Single-scale input (current behavior).\n" +
                    "2x: Context covers 2x the area. Moderate additional context.\n" +
                    "4x: Context covers 4x the area. Good for tissue-level patterns.\n" +
                    "8x: Context covers 8x the area. For large-scale classification.\n\n" +
                    "Adds C extra input channels (e.g., 3ch RGB -> 6ch with context).\n" +
                    "Modest memory increase (~5-10%). Compatible with all architectures.");

            grid.add(new Label("Context Scale:"), 0, row);
            grid.add(contextScaleCombo, 1, row);
            row++;

            // Overlap
            overlapSpinner = new Spinner<>(0, 50, DLClassifierPreferences.getTileOverlap(), 5);
            overlapSpinner.setEditable(true);
            overlapSpinner.setPrefWidth(100);
            TooltipHelper.install(overlapSpinner,
                    "Overlap between adjacent training tiles as a percentage.\n" +
                    "Higher overlap generates more training patches from\n" +
                    "the same annotations but increases extraction time.\n\n" +
                    "0%: No overlap -- fastest extraction, fewer tiles.\n" +
                    "10-25%: Typical range -- good balance of diversity and speed.\n" +
                    "Higher overlap is most beneficial with limited annotations.");

            grid.add(new Label("Tile Overlap (%):"), 0, row);
            grid.add(overlapSpinner, 1, row);
            row++;

            // Line stroke width - pre-fill from QuPath's annotation stroke thickness
            int defaultStroke = 5;
            try {
                defaultStroke = (int) Math.max(1, PathPrefs.annotationStrokeThicknessProperty().get());
            } catch (Exception e) {
                logger.debug("Could not read QuPath annotation stroke thickness, using default");
            }
            lineStrokeWidthSpinner = new Spinner<>(1, 50, defaultStroke, 1);
            lineStrokeWidthSpinner.setEditable(true);
            lineStrokeWidthSpinner.setPrefWidth(100);
            TooltipHelper.install(lineStrokeWidthSpinner,
                    "Width in pixels for rendering line/polyline annotations as training masks.\n" +
                    "Pre-filled from QuPath's annotation stroke thickness.\n\n" +
                    "Thin strokes (<5px) produce sparse training signal from polyline\n" +
                    "annotations -- consider increasing for better training.\n" +
                    "Only affects line/polyline annotations; area annotations are\n" +
                    "filled completely regardless of this setting.");

            grid.add(new Label("Line Stroke Width:"), 0, row);
            grid.add(lineStrokeWidthSpinner, 1, row);

            TitledPane pane = new TitledPane("TRAINING PARAMETERS", grid);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Configure training hyperparameters and tile extraction settings"));
            return pane;
        }

        private TitledPane createTrainingStrategySection() {
            GridPane grid = new GridPane();
            grid.setHgap(10);
            grid.setVgap(8);
            grid.setPadding(new Insets(10));

            ColumnConstraints labelCol = new ColumnConstraints();
            labelCol.setMinWidth(140);
            labelCol.setPrefWidth(150);
            ColumnConstraints fieldCol = new ColumnConstraints();
            fieldCol.setHgrow(Priority.ALWAYS);
            grid.getColumnConstraints().addAll(labelCol, fieldCol);

            int row = 0;

            // LR Scheduler
            schedulerCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "One Cycle", "Cosine Annealing", "Step Decay", "None"));
            schedulerCombo.setValue(mapSchedulerToDisplay(DLClassifierPreferences.getDefaultScheduler()));
            TooltipHelper.installWithLink(schedulerCombo,
                    "Learning rate schedule during training:\n\n" +
                    "One Cycle (recommended): Smooth ramp-up then decay.\n" +
                    "  Typically finds a good LR range automatically.\n\n" +
                    "Cosine Annealing: Periodic warm restarts.\n" +
                    "  Can escape local minima but may cause LR spikes.\n\n" +
                    "Step Decay: Reduce LR by factor every N epochs.\n" +
                    "  Predictable but requires manual tuning of step schedule.\n\n" +
                    "None: Constant learning rate throughout training.",
                    "https://pytorch.org/docs/stable/optim.html");

            grid.add(new Label("LR Scheduler:"), 0, row);
            grid.add(schedulerCombo, 1, row);
            row++;

            // Loss function
            lossFunctionCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "Cross Entropy + Dice", "Cross Entropy"));
            lossFunctionCombo.setValue(mapLossFunctionToDisplay(DLClassifierPreferences.getDefaultLossFunction()));
            TooltipHelper.installWithLink(lossFunctionCombo,
                    "Loss function for training:\n\n" +
                    "CE + Dice (recommended): Combines per-pixel Cross Entropy with\n" +
                    "  region overlap Dice loss. Modern standard for segmentation.\n" +
                    "  Dice directly optimizes IoU and handles class imbalance better.\n\n" +
                    "Cross Entropy: Per-pixel classification loss only.\n" +
                    "  May over-weight easy/majority pixels. Suitable when class\n" +
                    "  balance is good and boundaries are the priority.",
                    "https://smp.readthedocs.io/en/latest/losses.html");

            grid.add(new Label("Loss Function:"), 0, row);
            grid.add(lossFunctionCombo, 1, row);
            row++;

            // Early stopping metric
            earlyStoppingMetricCombo = new ComboBox<>(FXCollections.observableArrayList(
                    "Mean IoU", "Validation Loss"));
            earlyStoppingMetricCombo.setValue(
                    mapEarlyStoppingMetricToDisplay(DLClassifierPreferences.getDefaultEarlyStoppingMetric()));
            TooltipHelper.install(earlyStoppingMetricCombo,
                    "Metric monitored for early stopping:\n\n" +
                    "Mean IoU (recommended): Stops when segmentation quality plateaus.\n" +
                    "  Directly measures overlap between prediction and ground truth.\n\n" +
                    "Validation Loss: Stops when loss stops decreasing.\n" +
                    "  Loss can oscillate while IoU still improves, so Mean IoU\n" +
                    "  is generally a more reliable stopping criterion.");

            grid.add(new Label("Early Stop Metric:"), 0, row);
            grid.add(earlyStoppingMetricCombo, 1, row);
            row++;

            // Early stopping patience
            earlyStoppingPatienceSpinner = new Spinner<>(3, 50,
                    DLClassifierPreferences.getDefaultEarlyStoppingPatience(), 1);
            earlyStoppingPatienceSpinner.setEditable(true);
            earlyStoppingPatienceSpinner.setPrefWidth(100);
            TooltipHelper.install(earlyStoppingPatienceSpinner,
                    "Number of epochs to wait without improvement before stopping.\n" +
                    "Higher patience allows the model more time to recover from\n" +
                    "temporary plateaus, but risks wasting time on a converged model.\n\n" +
                    "10-15: Good default for most training runs.\n" +
                    "20-30: For noisy training curves or cosine annealing scheduler.\n" +
                    "5: For quick experiments or when resources are limited.");

            grid.add(new Label("Early Stop Patience:"), 0, row);
            grid.add(earlyStoppingPatienceSpinner, 1, row);
            row++;

            // Mixed precision
            mixedPrecisionCheck = new CheckBox("Enable mixed precision (AMP)");
            mixedPrecisionCheck.setSelected(DLClassifierPreferences.isDefaultMixedPrecision());
            TooltipHelper.installWithLink(mixedPrecisionCheck,
                    "Use automatic mixed precision (FP16/FP32) on CUDA GPUs.\n" +
                    "Typically provides ~2x speedup with no accuracy loss.\n" +
                    "Only active when training on NVIDIA GPUs; ignored on CPU/MPS.\n\n" +
                    "Safe to leave enabled -- PyTorch automatically manages which\n" +
                    "operations use FP16 vs FP32 for numerical stability.",
                    "https://pytorch.org/docs/stable/amp.html");

            grid.add(mixedPrecisionCheck, 0, row, 2, 1);

            TitledPane pane = new TitledPane("TRAINING STRATEGY", grid);
            pane.setExpanded(false); // Collapsed by default - advanced settings
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create(
                    "Advanced training strategy: scheduler, loss function, early stopping, and mixed precision"));
            return pane;
        }

        private TitledPane createChannelSection() {
            channelPanel = new ChannelSelectionPanel();
            channelPanel.validProperty().addListener((obs, old, valid) -> {
                updateValidation();
                // Channel count affects model layer structure - reload layers when channels become valid
                if (valid) {
                    updateLayerFreezePanel();
                }
            });

            TitledPane pane = new TitledPane("CHANNEL CONFIGURATION", channelPanel);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Select and order image channels for model input"));
            return pane;
        }

        private TitledPane createClassSection() {
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            Label infoLabel = new Label("Select annotation classes to use for training:");
            infoLabel.setStyle("-fx-text-fill: #666;");

            // Pie chart showing per-class annotation area distribution
            classDistributionChart = new PieChart();
            classDistributionChart.setLegendVisible(false);
            classDistributionChart.setLabelsVisible(true);
            classDistributionChart.setLabelLineLength(10);
            classDistributionChart.setPrefHeight(180);
            classDistributionChart.setMaxHeight(180);
            classDistributionChart.setVisible(false);
            classDistributionChart.setManaged(false);

            classListView = new ListView<>();
            classListView.setCellFactory(lv -> new ClassListCell());
            classListView.setPrefHeight(120);
            TooltipHelper.install(classListView,
                    "Annotation classes found in the selected images.\n" +
                    "At least 2 classes must be selected for training.\n" +
                    "Each class should have representative annotations.\n\n" +
                    "Tip: Line/polyline annotations are recommended over area\n" +
                    "annotations. Lines focus training on class boundaries\n" +
                    "where accuracy matters most, and avoid overtraining on\n" +
                    "easy central regions.\n\n" +
                    "Use the weight spinner (right) to boost underrepresented\n" +
                    "classes. For example, set weight=2.0 for a rare class.");

            // Add select all / none / rebalance buttons
            Button selectAllBtn = new Button("Select All");
            TooltipHelper.install(selectAllBtn, "Select all annotation classes for training");
            selectAllBtn.setOnAction(e -> classListView.getItems().forEach(item -> item.selected().set(true)));

            Button selectNoneBtn = new Button("Select None");
            TooltipHelper.install(selectNoneBtn, "Deselect all annotation classes");
            selectNoneBtn.setOnAction(e -> classListView.getItems().forEach(item -> item.selected().set(false)));

            Button rebalanceBtn = new Button("Rebalance Classes");
            TooltipHelper.install(rebalanceBtn,
                    "Auto-set weight multipliers to compensate for class imbalance.\n\n" +
                    "Classes with fewer annotated pixels receive higher weights so the\n" +
                    "model pays equal attention to all classes during training.\n\n" +
                    "Works with both area and line annotations. For lines, pixel\n" +
                    "coverage is estimated from line length x stroke width.\n\n" +
                    "Note: Rebalancing weights helps but does NOT replace having\n" +
                    "sufficient training data. Adding more annotations for under-\n" +
                    "represented classes will produce better results than relying\n" +
                    "on weight compensation alone.");
            rebalanceBtn.setOnAction(e -> rebalanceClassWeights());

            HBox buttonBox = new HBox(10, selectAllBtn, selectNoneBtn, rebalanceBtn);

            content.getChildren().addAll(infoLabel, classDistributionChart, classListView, buttonBox);

            TitledPane pane = new TitledPane("ANNOTATION CLASSES", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Select which annotation classes to include in training"));
            return pane;
        }

        private TitledPane createAugmentationSection() {
            VBox content = new VBox(8);
            content.setPadding(new Insets(10));

            flipHorizontalCheck = new CheckBox("Horizontal flip");
            flipHorizontalCheck.setSelected(DLClassifierPreferences.isAugFlipHorizontal());
            TooltipHelper.install(flipHorizontalCheck,
                    "Randomly mirror tiles left-right during training.\n" +
                    "Effective for tissue where orientation is arbitrary.\n" +
                    "Almost always beneficial; disable only if horizontal\n" +
                    "orientation carries meaning in your images.");

            flipVerticalCheck = new CheckBox("Vertical flip");
            flipVerticalCheck.setSelected(DLClassifierPreferences.isAugFlipVertical());
            TooltipHelper.install(flipVerticalCheck,
                    "Randomly mirror tiles top-bottom during training.\n" +
                    "Same rationale as horizontal flip.\n" +
                    "Safe to enable for most histopathology images.");

            rotationCheck = new CheckBox("Random rotation (90 deg)");
            rotationCheck.setSelected(DLClassifierPreferences.isAugRotation());
            TooltipHelper.install(rotationCheck,
                    "Randomly rotate tiles by 0/90/180/270 degrees.\n" +
                    "Preserves pixel alignment (no interpolation artifacts).\n" +
                    "Beneficial when tissue structures have no preferred\n" +
                    "orientation. Combines well with flips for 8x augmentation.");

            colorJitterCheck = new CheckBox("Color jitter");
            colorJitterCheck.setSelected(DLClassifierPreferences.isAugColorJitter());
            TooltipHelper.install(colorJitterCheck,
                    "Randomly perturb brightness, contrast, and saturation.\n" +
                    "Helps the model generalize across staining variations\n" +
                    "and illumination differences between slides.\n\n" +
                    "Recommended for H&E brightfield images.\n" +
                    "May not be appropriate for fluorescence or quantitative\n" +
                    "intensity-based classification.");

            elasticCheck = new CheckBox("Elastic deformation");
            elasticCheck.setSelected(DLClassifierPreferences.isAugElasticDeform());
            TooltipHelper.installWithLink(elasticCheck,
                    "Apply smooth random spatial deformations to tiles.\n" +
                    "Simulates tissue distortion and cutting artifacts.\n" +
                    "Computationally expensive but effective for histopathology.\n" +
                    "May reduce training speed by ~30%.\n\n" +
                    "Most beneficial when training data is limited and the\n" +
                    "model needs to handle shape variations in the tissue.",
                    "https://albumentations.ai/docs/");

            content.getChildren().addAll(
                    flipHorizontalCheck,
                    flipVerticalCheck,
                    rotationCheck,
                    colorJitterCheck,
                    elasticCheck
            );

            TitledPane pane = new TitledPane("DATA AUGMENTATION", content);
            pane.setExpanded(false); // Collapsed by default
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(TooltipHelper.create("Configure data augmentation to improve model generalization"));
            return pane;
        }

        private VBox createErrorSummaryPanel() {
            errorSummaryPanel = new VBox(5);
            errorSummaryPanel.setStyle(
                    "-fx-background-color: #fff3cd; " +
                    "-fx-border-color: #ffc107; " +
                    "-fx-border-width: 1px; " +
                    "-fx-padding: 10px;"
            );
            errorSummaryPanel.setVisible(false);
            errorSummaryPanel.setManaged(false);

            Label errorTitle = new Label("Please fix the following errors:");
            errorTitle.setStyle("-fx-font-weight: bold; -fx-text-fill: #856404;");

            errorListBox = new VBox(3);

            errorSummaryPanel.getChildren().addAll(errorTitle, errorListBox);
            return errorSummaryPanel;
        }

        /**
         * Loads classes from all selected project images as a union.
         * Runs image reading on a background thread to avoid blocking the FX thread.
         */
        private void loadClassesFromSelectedImages() {
            List<ImageSelectionItem> selectedItems = imageSelectionList.getItems().stream()
                    .filter(item -> item.selected.get())
                    .collect(Collectors.toList());

            if (selectedItems.isEmpty()) {
                return;
            }

            loadClassesButton.setDisable(true);
            loadClassesButton.setText("Loading classes...");

            // Capture stroke width on FX thread before async work -- used to
            // estimate pixel coverage for line/polyline annotations
            final int strokeWidth = lineStrokeWidthSpinner.getValue();

            CompletableFuture.runAsync(() -> {
                // Accumulate classes and effective pixel coverage across all selected images.
                // For area annotations, coverage = ROI area.
                // For line annotations, coverage = path length * stroke width.
                // This allows rebalancing to work with line annotations, which are
                // the recommended annotation style for boundary-focused training.
                Map<String, PathClass> classMap = new TreeMap<>();
                Map<String, Double> classAreas = new LinkedHashMap<>();
                ImageData<BufferedImage> firstImageData = null;

                for (ImageSelectionItem selItem : selectedItems) {
                    try {
                        ImageData<BufferedImage> data = selItem.entry.readImageData();

                        // Keep the first image's data for channel initialization
                        if (firstImageData == null) {
                            firstImageData = data;
                        }

                        for (PathObject annotation : data.getHierarchy().getAnnotationObjects()) {
                            PathClass pathClass = annotation.getPathClass();
                            if (pathClass != null && !pathClass.isDerivedClass()) {
                                classMap.putIfAbsent(pathClass.getName(), pathClass);

                                // Estimate effective pixel coverage:
                                // - Line/polyline ROIs have getArea()=0, so use
                                //   path length * stroke width as the pixel count
                                // - Area ROIs use geometric area directly
                                // - Mixed annotations (some lines, some areas) sum naturally
                                var roi = annotation.getROI();
                                double coverage;
                                if (roi.isLine()) {
                                    coverage = roi.getLength() * strokeWidth;
                                } else {
                                    coverage = roi.getArea();
                                }
                                classAreas.merge(pathClass.getName(), coverage, Double::sum);
                            }
                        }

                        // Close server for all images except the first
                        // (ChannelSelectionPanel stores currentServer for lazy bit depth lookup)
                        if (data != firstImageData) {
                            try {
                                data.getServer().close();
                            } catch (Exception e) {
                                logger.debug("Error closing image server: {}", e.getMessage());
                            }
                        }
                    } catch (Exception e) {
                        logger.warn("Could not read image '{}': {}",
                                selItem.entry.getImageName(), e.getMessage());
                    }
                }

                final ImageData<BufferedImage> channelImageData = firstImageData;
                final Map<String, PathClass> finalClassMap = classMap;
                final Map<String, Double> finalClassAreas = classAreas;

                Platform.runLater(() -> {
                    // Initialize channel panel from first image
                    if (channelImageData != null) {
                        channelPanel.setImageData(channelImageData);
                        channelPanel.autoConfigureForImageType(
                                channelImageData.getImageType(),
                                channelImageData.getServer().nChannels());

                        // Auto-disable color jitter for non-brightfield images
                        if (!isBrightfield(channelImageData)) {
                            colorJitterCheck.setSelected(false);
                            colorJitterCheck.setDisable(true);
                            TooltipHelper.install(colorJitterCheck,
                                    "Color jitter is disabled for non-brightfield images.\n" +
                                    "This augmentation perturbs brightness/contrast/saturation\n" +
                                    "which could corrupt quantitative fluorescence intensity data.");
                        } else {
                            colorJitterCheck.setDisable(false);
                        }
                    }

                    // Populate class list with union of all classes
                    classListView.getItems().clear();
                    for (Map.Entry<String, PathClass> entry : finalClassMap.entrySet()) {
                        PathClass pathClass = entry.getValue();
                        double area = finalClassAreas.getOrDefault(entry.getKey(), 0.0);
                        ClassItem classItem = new ClassItem(
                                pathClass.getName(), pathClass.getColor(), true, area);
                        classItem.selected().addListener((obs, old, newVal) -> {
                            refreshPieChart();
                            updateValidation();
                        });
                        classListView.getItems().add(classItem);
                    }

                    refreshPieChart();

                    // Enable gated sections
                    classesLoaded = true;
                    setGatedSectionsEnabled(true);

                    // Reset button state
                    loadClassesButton.setText("Load Classes from Selected Images");
                    updateLoadClassesButtonState();
                    updateValidation();

                    logger.info("Loaded {} classes from {} images",
                            finalClassMap.size(), selectedItems.size());
                });
            });
        }

        /** Enables/disables the Load Classes button based on whether any images are checked. */
        private void updateLoadClassesButtonState() {
            boolean anySelected = imageSelectionList.getItems().stream()
                    .anyMatch(item -> item.selected.get());
            loadClassesButton.setDisable(!anySelected);
        }

        /** Visual indicator when images change after classes were already loaded. */
        private void markClassesStale() {
            loadClassesButton.setText("Reload Classes (images changed)");
            loadClassesButton.setStyle(
                    "-fx-font-weight: bold; -fx-text-fill: #cc6600;");
        }

        /** Enables or disables all gated sections (everything except image source). */
        private void setGatedSectionsEnabled(boolean enabled) {
            for (TitledPane pane : gatedSections) {
                pane.setDisable(!enabled);
                if (!enabled) {
                    pane.setExpanded(false);
                }
            }
        }

        private boolean isBrightfield(ImageData<BufferedImage> imageData) {
            ImageData.ImageType type = imageData.getImageType();
            return type == ImageData.ImageType.BRIGHTFIELD_H_E
                    || type == ImageData.ImageType.BRIGHTFIELD_H_DAB
                    || type == ImageData.ImageType.BRIGHTFIELD_OTHER;
        }

        private void updateBackboneOptions(String architecture) {
            ClassifierHandler handler = ClassifierRegistry.getHandler(architecture)
                    .orElse(ClassifierRegistry.getDefaultHandler());

            Map<String, Object> params = handler.getArchitectureParams(null);
            Object backbones = params.get("available_backbones");

            List<String> backboneList = new ArrayList<>();
            if (backbones instanceof List<?>) {
                for (Object b : (List<?>) backbones) {
                    backboneList.add(b.toString());
                }
            } else {
                backboneList.addAll(List.of("resnet34", "resnet50", "efficientnet-b0"));
            }

            backboneCombo.setItems(FXCollections.observableArrayList(backboneList));

            // Show display names via custom cell factory
            backboneCombo.setCellFactory(lv -> new ListCell<>() {
                @Override
                protected void updateItem(String item, boolean empty) {
                    super.updateItem(item, empty);
                    if (empty || item == null) {
                        setText(null);
                    } else {
                        setText(qupath.ext.dlclassifier.classifier.handlers.UNetHandler
                                .getBackboneDisplayName(item));
                    }
                }
            });
            backboneCombo.setButtonCell(new ListCell<>() {
                @Override
                protected void updateItem(String item, boolean empty) {
                    super.updateItem(item, empty);
                    if (empty || item == null) {
                        setText(null);
                    } else {
                        setText(qupath.ext.dlclassifier.classifier.handlers.UNetHandler
                                .getBackboneDisplayName(item));
                    }
                }
            });

            // Restore last used backbone from preferences if it's available for this architecture
            String savedBackbone = DLClassifierPreferences.getLastBackbone();
            if (backboneList.contains(savedBackbone)) {
                backboneCombo.setValue(savedBackbone);
            } else if (!backboneList.isEmpty()) {
                backboneCombo.setValue(backboneList.get(0));
            }
        }

        private void validateClassifierName(String name) {
            if (name == null || name.trim().isEmpty()) {
                validationErrors.put("name", "Classifier name is required");
            } else if (!name.matches("[a-zA-Z0-9_-]+")) {
                validationErrors.put("name", "Classifier name can only contain letters, numbers, underscore, and hyphen");
            } else {
                validationErrors.remove("name");
            }
            updateErrorSummary();
        }

        private void updateValidation() {
            // Check that classes have been loaded
            if (!classesLoaded) {
                validationErrors.put("classesLoaded",
                        "Select images and click 'Load Classes from Selected Images'");
                // Clear channel/class errors since they are not relevant yet
                validationErrors.remove("channels");
                validationErrors.remove("classes");
                validationErrors.remove("images");
                updateErrorSummary();
                return;
            }
            validationErrors.remove("classesLoaded");

            // Check at least 1 image is selected
            long selectedImageCount = imageSelectionList.getItems().stream()
                    .filter(item -> item.selected.get())
                    .count();
            if (selectedImageCount < 1) {
                validationErrors.put("images", "At least 1 image must be selected for training");
            } else {
                validationErrors.remove("images");
            }

            // Check channels
            if (!channelPanel.isValid()) {
                validationErrors.put("channels", "Invalid channel configuration");
            } else {
                validationErrors.remove("channels");
            }

            // Check classes
            long selectedClasses = classListView.getItems().stream()
                    .filter(item -> item.selected().get())
                    .count();
            if (selectedClasses < 2) {
                validationErrors.put("classes", "At least 2 classes must be selected for training");
            } else {
                validationErrors.remove("classes");
            }

            updateErrorSummary();
        }

        private void updateErrorSummary() {
            if (validationErrors.isEmpty()) {
                errorSummaryPanel.setVisible(false);
                errorSummaryPanel.setManaged(false);
                okButton.setDisable(false);
            } else {
                errorListBox.getChildren().clear();
                validationErrors.forEach((fieldId, errorMsg) -> {
                    Label errorLabel = new Label("- " + errorMsg);
                    errorLabel.setStyle("-fx-text-fill: #856404;");
                    errorListBox.getChildren().add(errorLabel);
                });

                errorSummaryPanel.setVisible(true);
                errorSummaryPanel.setManaged(true);
                okButton.setDisable(true);
            }
        }

        private void refreshPieChart() {
            if (classDistributionChart == null) return;
            classDistributionChart.getData().clear();

            double totalArea = 0;
            List<ClassItem> selectedItems = new ArrayList<>();
            for (ClassItem item : classListView.getItems()) {
                if (item.selected().get() && item.annotationArea() > 0) {
                    selectedItems.add(item);
                    totalArea += item.annotationArea();
                }
            }

            if (totalArea == 0 || selectedItems.isEmpty()) {
                classDistributionChart.setVisible(false);
                classDistributionChart.setManaged(false);
                return;
            }

            classDistributionChart.setVisible(true);
            classDistributionChart.setManaged(true);

            for (ClassItem item : selectedItems) {
                double pct = (item.annotationArea() / totalArea) * 100.0;
                String label = String.format("%s (%.1f%%)", item.name(), pct);
                PieChart.Data data = new PieChart.Data(label, item.annotationArea());
                classDistributionChart.getData().add(data);
            }

            // Apply QuPath class colors to pie slices
            for (int i = 0; i < selectedItems.size(); i++) {
                ClassItem item = selectedItems.get(i);
                PieChart.Data data = classDistributionChart.getData().get(i);
                if (item.color() != null) {
                    int r = (item.color() >> 16) & 0xFF;
                    int g = (item.color() >> 8) & 0xFF;
                    int b = item.color() & 0xFF;
                    String style = "-fx-pie-color: rgb(" + r + "," + g + "," + b + ");";
                    // Node may not exist yet if chart hasn't been laid out
                    if (data.getNode() != null) {
                        data.getNode().setStyle(style);
                    }
                    data.nodeProperty().addListener((obs, oldNode, newNode) -> {
                        if (newNode != null) newNode.setStyle(style);
                    });
                }
            }
        }

        private void rebalanceClassWeights() {
            List<ClassItem> allItems = classListView.getItems();
            logger.info("Rebalance: classListView has {} items", allItems.size());

            if (allItems.isEmpty()) {
                Dialogs.showWarningNotification("Rebalance",
                        "No classes loaded. Click 'Load Classes from Selected Images' first.");
                return;
            }

            List<ClassItem> selected = allItems.stream()
                    .filter(item -> item.selected().get())
                    .collect(Collectors.toList());

            if (selected.isEmpty()) {
                Dialogs.showWarningNotification("Rebalance",
                        "No classes are selected. Check at least 2 classes to rebalance.");
                logger.warn("Rebalance: no selected classes");
                return;
            }

            // Log per-class areas for diagnostics
            for (ClassItem item : selected) {
                logger.info("Rebalance: class '{}' annotationArea={}", item.name(), item.annotationArea());
            }

            // Collect non-zero areas and sort for median calculation
            List<Double> areas = selected.stream()
                    .map(ClassItem::annotationArea)
                    .filter(a -> a > 0)
                    .sorted()
                    .collect(Collectors.toList());

            if (areas.isEmpty()) {
                Dialogs.showWarningNotification("Rebalance",
                        "All selected classes have zero estimated pixel coverage.\n" +
                        "Try reloading classes (coverage is estimated from annotation\n" +
                        "area or line length x stroke width).");
                logger.warn("Rebalance: all selected classes have zero coverage -- cannot compute weights. " +
                        "Are annotations point ROIs? Line/area annotations are required.");
                return;
            }

            // Compute median area
            double median;
            int n = areas.size();
            if (n % 2 == 0) {
                median = (areas.get(n / 2 - 1) + areas.get(n / 2)) / 2.0;
            } else {
                median = areas.get(n / 2);
            }
            logger.info("Rebalance: median area = {}", median);

            // Set inverse-frequency weights clamped to spinner range [0.1, 10.0]
            // The property->spinner listener in ClassListCell updates spinners automatically
            for (ClassItem item : selected) {
                double area = item.annotationArea();
                double weight;
                if (area > 0) {
                    weight = Math.max(0.1, Math.min(10.0, median / area));
                } else {
                    weight = 1.0;
                }
                item.weightMultiplier().set(weight);
            }

            // Build user-visible summary
            StringBuilder sb = new StringBuilder();
            for (ClassItem item : selected) {
                if (sb.length() > 0) sb.append(", ");
                sb.append(item.name())
                  .append("=").append(String.format("%.2f", item.weightMultiplier().get()));
            }
            String summary = sb.toString();
            logger.info("Rebalanced class weights: {}", summary);
            Dialogs.showInfoNotification("Rebalance",
                    "Weights updated: " + summary);
        }

        private TrainingDialogResult buildResult() {
            // Save dialog settings to preferences for next session
            DLClassifierPreferences.setLastArchitecture(architectureCombo.getValue());
            DLClassifierPreferences.setLastBackbone(backboneCombo.getValue());
            DLClassifierPreferences.setValidationSplit(validationSplitSpinner.getValue());
            DLClassifierPreferences.setAugFlipHorizontal(flipHorizontalCheck.isSelected());
            DLClassifierPreferences.setAugFlipVertical(flipVerticalCheck.isSelected());
            DLClassifierPreferences.setAugRotation(rotationCheck.isSelected());
            DLClassifierPreferences.setAugColorJitter(colorJitterCheck.isSelected());
            DLClassifierPreferences.setAugElasticDeform(elasticCheck.isSelected());

            // Get frozen layers for transfer learning
            List<String> frozenLayers = new ArrayList<>();
            if (usePretrainedCheck.isSelected() && layerFreezePanel != null) {
                frozenLayers = layerFreezePanel.getFrozenLayerNames();
            }

            // Extract weight multipliers and class colors from selected class items
            Map<String, Double> classWeightMultipliers = new LinkedHashMap<>();
            Map<String, Integer> classColors = new LinkedHashMap<>();
            for (ClassItem item : classListView.getItems()) {
                if (item.selected().get()) {
                    double multiplier = item.weightMultiplier().get();
                    if (multiplier != 1.0) {
                        classWeightMultipliers.put(item.name(), multiplier);
                    }
                    if (item.color() != null) {
                        classColors.put(item.name(), item.color());
                    }
                }
            }

            // Save training strategy preferences
            DLClassifierPreferences.setDefaultScheduler(mapSchedulerFromDisplay(schedulerCombo.getValue()));
            DLClassifierPreferences.setDefaultLossFunction(mapLossFunctionFromDisplay(lossFunctionCombo.getValue()));
            DLClassifierPreferences.setDefaultEarlyStoppingMetric(
                    mapEarlyStoppingMetricFromDisplay(earlyStoppingMetricCombo.getValue()));
            DLClassifierPreferences.setDefaultEarlyStoppingPatience(earlyStoppingPatienceSpinner.getValue());
            DLClassifierPreferences.setDefaultMixedPrecision(mixedPrecisionCheck.isSelected());

            // Build training config
            TrainingConfig trainingConfig = TrainingConfig.builder()
                    .classifierType(architectureCombo.getValue())
                    .backbone(backboneCombo.getValue())
                    .epochs(epochsSpinner.getValue())
                    .batchSize(batchSizeSpinner.getValue())
                    .learningRate(learningRateSpinner.getValue())
                    .validationSplit(validationSplitSpinner.getValue() / 100.0)
                    .tileSize(tileSizeSpinner.getValue())
                    .overlap(overlapSpinner.getValue())
                    .downsample(parseDownsample(downsampleCombo.getValue()))
                    .contextScale(parseContextScale(contextScaleCombo.getValue()))
                    .augmentation(buildAugmentationConfig())
                    .usePretrainedWeights(usePretrainedCheck.isSelected())
                    .frozenLayers(frozenLayers)
                    .lineStrokeWidth(lineStrokeWidthSpinner.getValue())
                    .classWeightMultipliers(classWeightMultipliers)
                    .schedulerType(mapSchedulerFromDisplay(schedulerCombo.getValue()))
                    .lossFunction(mapLossFunctionFromDisplay(lossFunctionCombo.getValue()))
                    .earlyStoppingMetric(mapEarlyStoppingMetricFromDisplay(earlyStoppingMetricCombo.getValue()))
                    .earlyStoppingPatience(earlyStoppingPatienceSpinner.getValue())
                    .mixedPrecision(mixedPrecisionCheck.isSelected())
                    .build();

            // Get channel config
            ChannelConfiguration channelConfig = channelPanel.getChannelConfiguration();

            // Get selected classes
            List<String> selectedClasses = classListView.getItems().stream()
                    .filter(item -> item.selected().get())
                    .map(ClassItem::name)
                    .collect(Collectors.toList());

            // Always collect selected images from the list
            List<ProjectImageEntry<BufferedImage>> selectedImages = imageSelectionList.getItems().stream()
                    .filter(item -> item.selected.get())
                    .map(item -> item.entry)
                    .collect(Collectors.toList());

            return new TrainingDialogResult(
                    classifierNameField.getText().trim(),
                    descriptionField.getText().trim(),
                    trainingConfig,
                    channelConfig,
                    selectedClasses,
                    selectedImages,
                    classColors
            );
        }

        private Map<String, Boolean> buildAugmentationConfig() {
            Map<String, Boolean> config = new LinkedHashMap<>();
            config.put("flip_horizontal", flipHorizontalCheck.isSelected());
            config.put("flip_vertical", flipVerticalCheck.isSelected());
            config.put("rotation_90", rotationCheck.isSelected());
            config.put("color_jitter", colorJitterCheck.isSelected());
            config.put("elastic_deformation", elasticCheck.isSelected());
            return config;
        }

        /**
         * Parses downsample value from ComboBox display string.
         */
        private static double parseDownsample(String displayValue) {
            if (displayValue == null) return 1.0;
            if (displayValue.startsWith("8x")) return 8.0;
            if (displayValue.startsWith("4x")) return 4.0;
            if (displayValue.startsWith("2x")) return 2.0;
            return 1.0;
        }

        /**
         * Parses context scale value from ComboBox display string.
         */
        private static int parseContextScale(String displayValue) {
            if (displayValue == null) return 1;
            if (displayValue.startsWith("8x")) return 8;
            if (displayValue.startsWith("4x")) return 4;
            if (displayValue.startsWith("2x")) return 2;
            return 1;
        }

        // ==================== Display/Value Mapping Helpers ====================

        private static String mapSchedulerToDisplay(String value) {
            if (value == null) return "One Cycle";
            return switch (value) {
                case "onecycle" -> "One Cycle";
                case "cosine" -> "Cosine Annealing";
                case "step" -> "Step Decay";
                case "none" -> "None";
                default -> "One Cycle";
            };
        }

        private static String mapSchedulerFromDisplay(String display) {
            if (display == null) return "onecycle";
            return switch (display) {
                case "One Cycle" -> "onecycle";
                case "Cosine Annealing" -> "cosine";
                case "Step Decay" -> "step";
                case "None" -> "none";
                default -> "onecycle";
            };
        }

        private static String mapLossFunctionToDisplay(String value) {
            if ("cross_entropy".equals(value)) return "Cross Entropy";
            return "Cross Entropy + Dice";
        }

        private static String mapLossFunctionFromDisplay(String display) {
            if ("Cross Entropy".equals(display)) return "cross_entropy";
            return "ce_dice";
        }

        private static String mapEarlyStoppingMetricToDisplay(String value) {
            if ("val_loss".equals(value)) return "Validation Loss";
            return "Mean IoU";
        }

        private static String mapEarlyStoppingMetricFromDisplay(String display) {
            if ("Validation Loss".equals(display)) return "val_loss";
            return "mean_iou";
        }

        private void copyTrainingScript(Button sourceButton) {
            String name = classifierNameField.getText().trim();
            if (name.isEmpty()) {
                showCopyFeedback(sourceButton, "Enter a classifier name first");
                return;
            }

            // Get frozen layers for transfer learning
            List<String> frozenLayers = new ArrayList<>();
            if (usePretrainedCheck.isSelected() && layerFreezePanel != null) {
                frozenLayers = layerFreezePanel.getFrozenLayerNames();
            }

            TrainingConfig config = TrainingConfig.builder()
                    .classifierType(architectureCombo.getValue())
                    .backbone(backboneCombo.getValue())
                    .epochs(epochsSpinner.getValue())
                    .batchSize(batchSizeSpinner.getValue())
                    .learningRate(learningRateSpinner.getValue())
                    .validationSplit(validationSplitSpinner.getValue() / 100.0)
                    .tileSize(tileSizeSpinner.getValue())
                    .overlap(overlapSpinner.getValue())
                    .downsample(parseDownsample(downsampleCombo.getValue()))
                    .contextScale(parseContextScale(contextScaleCombo.getValue()))
                    .augmentation(buildAugmentationConfig())
                    .usePretrainedWeights(usePretrainedCheck.isSelected())
                    .frozenLayers(frozenLayers)
                    .lineStrokeWidth(lineStrokeWidthSpinner.getValue())
                    .schedulerType(mapSchedulerFromDisplay(schedulerCombo.getValue()))
                    .lossFunction(mapLossFunctionFromDisplay(lossFunctionCombo.getValue()))
                    .earlyStoppingMetric(mapEarlyStoppingMetricFromDisplay(earlyStoppingMetricCombo.getValue()))
                    .earlyStoppingPatience(earlyStoppingPatienceSpinner.getValue())
                    .mixedPrecision(mixedPrecisionCheck.isSelected())
                    .build();

            ChannelConfiguration channelConfig = channelPanel.getChannelConfiguration();

            List<String> selectedClasses = classListView.getItems().stream()
                    .filter(item -> item.selected().get())
                    .map(ClassItem::name)
                    .collect(Collectors.toList());

            String script = ScriptGenerator.generateTrainingScript(
                    name, descriptionField.getText().trim(),
                    config, channelConfig, selectedClasses);

            Clipboard clipboard = Clipboard.getSystemClipboard();
            ClipboardContent content = new ClipboardContent();
            content.putString(script);
            clipboard.setContent(content);

            showCopyFeedback(sourceButton, "Script copied to clipboard!");
        }

        private void showCopyFeedback(Button button, String message) {
            Tooltip tooltip = new Tooltip(message);
            tooltip.setAutoHide(true);
            tooltip.show(button,
                    button.localToScreen(button.getBoundsInLocal()).getMinX(),
                    button.localToScreen(button.getBoundsInLocal()).getMinY() - 30);
            PauseTransition pause = new PauseTransition(Duration.seconds(2));
            pause.setOnFinished(e -> tooltip.hide());
            pause.play();
        }
    }

    /**
     * Represents a class item in the list.
     */
    private record ClassItem(String name, Integer color,
                              javafx.beans.property.BooleanProperty selected,
                              javafx.beans.property.DoubleProperty weightMultiplier,
                              double annotationArea) {
        public ClassItem(String name, Integer color, boolean selected, double annotationArea) {
            this(name, color,
                    new javafx.beans.property.SimpleBooleanProperty(selected),
                    new javafx.beans.property.SimpleDoubleProperty(1.0),
                    annotationArea);
        }
    }

    /**
     * Custom cell renderer for class items with weight multiplier spinner.
     * Properly manages bidirectional binding between the spinner and the
     * ClassItem's weightMultiplier property, cleaning up on cell reuse.
     */
    private static class ClassListCell extends ListCell<ClassItem> {
        private final HBox content;
        private final CheckBox checkBox;
        private final javafx.scene.shape.Rectangle colorBox;
        private final Label weightLabel;
        private final Spinner<Double> weightSpinner;

        // Track current bindings for cleanup on cell reuse
        private javafx.beans.property.BooleanProperty boundSelectedProperty;
        private javafx.beans.value.ChangeListener<Boolean> itemToCheckboxListener;
        private javafx.beans.value.ChangeListener<Number> weightToSpinnerListener;
        private javafx.beans.value.ChangeListener<Double> spinnerToWeightListener;
        private javafx.beans.property.DoubleProperty boundWeightProperty;

        public ClassListCell() {
            checkBox = new CheckBox();
            colorBox = new javafx.scene.shape.Rectangle(16, 16);
            colorBox.setStroke(javafx.scene.paint.Color.BLACK);
            colorBox.setStrokeWidth(1);

            weightLabel = new Label("Weight:");
            weightLabel.setStyle("-fx-text-fill: #666;");

            weightSpinner = new Spinner<>(0.1, 10.0, 1.0, 0.1);
            weightSpinner.setPrefWidth(80);
            weightSpinner.setEditable(true);
            weightSpinner.setTooltip(TooltipHelper.create(
                    "Multiplier applied to auto-computed class weight.\n" +
                    "1.0 = no change. >1.0 emphasizes this class.\n" +
                    "Use to boost underperforming or rare classes.\n\n" +
                    "Example: Set to 2.0 for a class with few annotations\n" +
                    "to give it more influence during training."));

            Region spacer = new Region();
            HBox.setHgrow(spacer, Priority.ALWAYS);

            content = new HBox(8, checkBox, colorBox, spacer, weightLabel, weightSpinner);
            content.setAlignment(Pos.CENTER_LEFT);
        }

        @Override
        protected void updateItem(ClassItem item, boolean empty) {
            super.updateItem(item, empty);

            // Clean up previous listeners to avoid leaks during cell reuse
            if (boundSelectedProperty != null && itemToCheckboxListener != null) {
                boundSelectedProperty.removeListener(itemToCheckboxListener);
            }
            checkBox.setOnAction(null);
            boundSelectedProperty = null;
            itemToCheckboxListener = null;
            if (boundWeightProperty != null && weightToSpinnerListener != null) {
                boundWeightProperty.removeListener(weightToSpinnerListener);
            }
            if (spinnerToWeightListener != null) {
                weightSpinner.valueProperty().removeListener(spinnerToWeightListener);
            }
            boundWeightProperty = null;
            weightToSpinnerListener = null;
            spinnerToWeightListener = null;

            if (empty || item == null) {
                setGraphic(null);
            } else {
                checkBox.setText(item.name());
                checkBox.setSelected(item.selected().get());
                checkBox.setOnAction(e -> item.selected().set(checkBox.isSelected()));
                boundSelectedProperty = item.selected();
                itemToCheckboxListener = (obs, oldVal, newVal) -> checkBox.setSelected(newVal);
                boundSelectedProperty.addListener(itemToCheckboxListener);

                if (item.color() != null) {
                    int r = (item.color() >> 16) & 0xFF;
                    int g = (item.color() >> 8) & 0xFF;
                    int b = item.color() & 0xFF;
                    colorBox.setFill(javafx.scene.paint.Color.rgb(r, g, b));
                } else {
                    colorBox.setFill(javafx.scene.paint.Color.GRAY);
                }

                // Set initial spinner value
                weightSpinner.getValueFactory().setValue(item.weightMultiplier().get());

                // Property -> spinner: update spinner when weight changes programmatically
                boundWeightProperty = item.weightMultiplier();
                weightToSpinnerListener = (obs, oldVal, newVal) -> {
                    if (newVal != null) {
                        weightSpinner.getValueFactory().setValue(newVal.doubleValue());
                    }
                };
                boundWeightProperty.addListener(weightToSpinnerListener);

                // Spinner -> property: update weight when user changes spinner
                spinnerToWeightListener = (obs, oldVal, newVal) -> {
                    if (newVal != null) {
                        item.weightMultiplier().set(newVal);
                    }
                };
                weightSpinner.valueProperty().addListener(spinnerToWeightListener);

                setGraphic(content);
            }
        }
    }

    /**
     * Represents a project image in the selection list.
     */
    private static class ImageSelectionItem {
        final ProjectImageEntry<BufferedImage> entry;
        final String imageName;
        final long annotationCount;
        final javafx.beans.property.BooleanProperty selected;

        ImageSelectionItem(ProjectImageEntry<BufferedImage> entry, long annotationCount) {
            this.entry = entry;
            this.imageName = entry.getImageName() + " (" + annotationCount + " annotations)";
            this.annotationCount = annotationCount;
            this.selected = new javafx.beans.property.SimpleBooleanProperty(true);
        }
    }

    /**
     * Generic checkbox list cell for items with a selected property.
     * Uses explicit listeners instead of bidirectional binding to prevent
     * linked checkbox behavior during ListView cell recycling.
     */
    private static class CheckBoxListCell<T> extends ListCell<T> {
        private final java.util.function.Function<T, javafx.beans.property.BooleanProperty> selectedExtractor;
        private final java.util.function.Function<T, String> textExtractor;
        private final CheckBox checkBox = new CheckBox();

        private javafx.beans.property.BooleanProperty boundSelectedProperty;
        private javafx.beans.value.ChangeListener<Boolean> itemToCheckboxListener;

        CheckBoxListCell(java.util.function.Function<T, javafx.beans.property.BooleanProperty> selectedExtractor,
                         java.util.function.Function<T, String> textExtractor) {
            this.selectedExtractor = selectedExtractor;
            this.textExtractor = textExtractor;
        }

        @Override
        protected void updateItem(T item, boolean empty) {
            super.updateItem(item, empty);

            // Clean up previous listeners to avoid leaks during cell reuse
            if (boundSelectedProperty != null && itemToCheckboxListener != null) {
                boundSelectedProperty.removeListener(itemToCheckboxListener);
            }
            checkBox.setOnAction(null);
            boundSelectedProperty = null;
            itemToCheckboxListener = null;

            if (empty || item == null) {
                setGraphic(null);
            } else {
                checkBox.setText(textExtractor.apply(item));
                boundSelectedProperty = selectedExtractor.apply(item);
                checkBox.setSelected(boundSelectedProperty.get());
                checkBox.setOnAction(e -> boundSelectedProperty.set(checkBox.isSelected()));
                itemToCheckboxListener = (obs, oldVal, newVal) -> checkBox.setSelected(newVal);
                boundSelectedProperty.addListener(itemToCheckboxListener);
                setGraphic(checkBox);
            }
        }
    }
}
