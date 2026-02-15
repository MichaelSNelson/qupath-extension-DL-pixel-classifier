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
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.lib.gui.QuPathGUI;
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
     */
    /**
     * @param selectedImages project images to train from, or null for current image only
     */
    public record TrainingDialogResult(
            String classifierName,
            String description,
            TrainingConfig trainingConfig,
            ChannelConfiguration channelConfig,
            List<String> selectedClasses,
            List<ProjectImageEntry<BufferedImage>> selectedImages
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
        private Spinner<Integer> lineStrokeWidthSpinner;

        // Channel selection
        private ChannelSelectionPanel channelPanel;

        // Class selection
        private ListView<ClassItem> classListView;

        // Augmentation
        private CheckBox flipHorizontalCheck;
        private CheckBox flipVerticalCheck;
        private CheckBox rotationCheck;
        private CheckBox colorJitterCheck;
        private CheckBox elasticCheck;

        // Transfer learning
        private LayerFreezePanel layerFreezePanel;
        private CheckBox usePretrainedCheck;
        private ClassifierClient client;

        // Image source selection
        private RadioButton currentImageRadio;
        private RadioButton projectImagesRadio;
        private ListView<ImageSelectionItem> imageSelectionList;

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

            // Initialize client for server communication
            try {
                client = new ClassifierClient(
                        DLClassifierPreferences.getServerHost(),
                        DLClassifierPreferences.getServerPort()
                );
            } catch (Exception e) {
                logger.warn("Could not initialize classifier client: {}", e.getMessage());
            }

            // Create channel and class sections first so their fields exist
            // before the model section's backbone listener fires
            TitledPane channelSection = createChannelSection();
            TitledPane classSection = createClassSection();

            // Create collapsible sections (visual order unchanged)
            content.getChildren().addAll(
                    createBasicInfoSection(),
                    createImageSourceSection(),
                    createModelSection(),
                    createTransferLearningSection(),
                    createTrainingSection(),
                    channelSection,
                    classSection,
                    createAugmentationSection(),
                    createErrorSummaryPanel()
            );

            ScrollPane scrollPane = new ScrollPane(content);
            scrollPane.setFitToWidth(true);
            scrollPane.setPrefHeight(600);
            scrollPane.setPrefWidth(550);

            dialog.getDialogPane().setContent(scrollPane);

            // Initialize with current image
            initializeFromCurrentImage();

            // Trigger initial layer load now that all UI components exist
            updateLayerFreezePanel();

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
            classifierNameField.setTooltip(new Tooltip(
                    "Unique identifier for this classifier.\n" +
                    "Used as the filename when saving. Only letters, numbers, underscore, and hyphen."));
            classifierNameField.textProperty().addListener((obs, old, newVal) -> validateClassifierName(newVal));

            grid.add(new Label("Classifier Name:"), 0, row);
            grid.add(classifierNameField, 1, row);
            row++;

            // Description
            descriptionField = new TextArea();
            descriptionField.setPromptText("Optional description of what this classifier detects...");
            descriptionField.setPrefRowCount(2);
            descriptionField.setWrapText(true);
            descriptionField.setTooltip(new Tooltip(
                    "Optional free-text description of what this classifier detects.\n" +
                    "Stored in classifier metadata for documentation."));

            grid.add(new Label("Description:"), 0, row);
            grid.add(descriptionField, 1, row);

            TitledPane pane = new TitledPane("CLASSIFIER INFO", grid);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(new Tooltip("Basic identification for the trained classifier model"));
            return pane;
        }

        private TitledPane createImageSourceSection() {
            VBox content = new VBox(8);
            content.setPadding(new Insets(10));

            ToggleGroup sourceGroup = new ToggleGroup();
            currentImageRadio = new RadioButton("Current image only");
            currentImageRadio.setToggleGroup(sourceGroup);
            currentImageRadio.setSelected(true);
            currentImageRadio.setTooltip(new Tooltip(
                    "Train using only annotations from the currently open image.\n" +
                    "Simpler but may overfit to a single image."));

            projectImagesRadio = new RadioButton("Selected project images");
            projectImagesRadio.setToggleGroup(sourceGroup);
            projectImagesRadio.setTooltip(new Tooltip(
                    "Train using annotations from multiple project images.\n" +
                    "Improves generalization across different tissue regions."));

            imageSelectionList = new ListView<>();
            imageSelectionList.setCellFactory(lv -> new CheckBoxListCell<>(
                    item -> item.selected,
                    item -> item.imageName
            ));
            imageSelectionList.setPrefHeight(120);
            imageSelectionList.setTooltip(new Tooltip(
                    "Check the project images to include in training.\n" +
                    "Only images with classified annotations are shown."));
            imageSelectionList.setDisable(true);
            imageSelectionList.setVisible(false);
            imageSelectionList.setManaged(false);

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
                            imageSelectionList.getItems().add(
                                    new ImageSelectionItem(entry, annotationCount));
                        }
                    } catch (Exception e) {
                        logger.debug("Could not read image '{}': {}",
                                entry.getImageName(), e.getMessage());
                    }
                }
            }

            // Disable project option if no project or no annotated images
            if (imageSelectionList.getItems().isEmpty()) {
                projectImagesRadio.setDisable(true);
                projectImagesRadio.setTooltip(new Tooltip(
                        "No project images with annotations found"));
            }

            // Toggle image list visibility
            sourceGroup.selectedToggleProperty().addListener((obs, old, toggle) -> {
                boolean showList = toggle == projectImagesRadio;
                imageSelectionList.setDisable(!showList);
                imageSelectionList.setVisible(showList);
                imageSelectionList.setManaged(showList);
            });

            Label info = new Label("Training data source:");
            info.setStyle("-fx-text-fill: #666;");

            content.getChildren().addAll(info, currentImageRadio, projectImagesRadio, imageSelectionList);

            TitledPane pane = new TitledPane("TRAINING DATA SOURCE", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(new Tooltip("Choose whether to train from one image or multiple project images"));
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
            architectureCombo.setTooltip(new Tooltip(
                    "Segmentation architecture.\n" +
                    "UNet: symmetric encoder-decoder with skip connections, good default.\n" +
                    "DeepLabV3+: atrous convolutions for multi-scale context.\n" +
                    "FPN: feature pyramid for objects at varying scales.\n" +
                    "PSPNet: pyramid pooling for global context."));
            architectureCombo.valueProperty().addListener((obs, old, newVal) -> updateBackboneOptions(newVal));

            grid.add(new Label("Architecture:"), 0, row);
            grid.add(architectureCombo, 1, row);
            row++;

            // Backbone selection
            backboneCombo = new ComboBox<>();
            backboneCombo.setTooltip(new Tooltip(
                    "Pretrained encoder network that extracts features.\n" +
                    "resnet34: good balance of speed and accuracy.\n" +
                    "resnet50: deeper, more capacity, slower.\n" +
                    "efficientnet-b0: lightweight, fast inference.\n\n" +
                    "Histology backbones (marked with 'Histology') use weights\n" +
                    "pretrained on millions of tissue patches instead of ImageNet.\n" +
                    "They provide better feature extraction for tissue classification\n" +
                    "and require less layer freezing. ~100MB download on first use."));
            updateBackboneOptions(architectureCombo.getValue());

            grid.add(new Label("Backbone:"), 0, row);
            grid.add(backboneCombo, 1, row);

            TitledPane pane = new TitledPane("MODEL ARCHITECTURE", grid);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(new Tooltip("Select the neural network architecture and encoder backbone"));
            return pane;
        }

        private TitledPane createTransferLearningSection() {
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            // Use pretrained checkbox
            usePretrainedCheck = new CheckBox("Use pretrained weights");
            usePretrainedCheck.setSelected(true);
            usePretrainedCheck.setTooltip(new Tooltip(
                    "Initialize encoder with pretrained weights.\n" +
                    "Dramatically improves convergence and final accuracy,\n" +
                    "especially with limited training data. Recommended for\n" +
                    "most use cases."));

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
            layerFreezePanel.setClient(client);

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
            pane.setTooltip(new Tooltip("Configure pretrained weight usage and layer freezing strategy"));
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

                    layerFreezePanel.loadLayers(architecture, encoder, numChannels, numClasses);
                } catch (Exception e) {
                    logger.debug("Could not update layer freeze panel: {}", e.getMessage());
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
            epochsSpinner.setTooltip(new Tooltip(
                    "Number of complete passes through the training data.\n" +
                    "More epochs allow the model to learn more but risk overfitting.\n" +
                    "Watch validation loss to determine when to stop.\n" +
                    "Typical range: 50-200 for small datasets, 20-100 for large."));

            grid.add(new Label("Epochs:"), 0, row);
            grid.add(epochsSpinner, 1, row);
            row++;

            // Batch size
            batchSizeSpinner = new Spinner<>(1, 128, DLClassifierPreferences.getDefaultBatchSize(), 4);
            batchSizeSpinner.setEditable(true);
            batchSizeSpinner.setPrefWidth(100);
            batchSizeSpinner.setTooltip(new Tooltip(
                    "Number of tiles processed together in each training step.\n" +
                    "Larger batches give more stable gradients but use more GPU memory.\n" +
                    "Reduce if you get CUDA out-of-memory errors.\n" +
                    "Typical: 4-16 depending on tile size and GPU VRAM."));

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
            learningRateSpinner.setTooltip(new Tooltip(
                    "Controls the step size during gradient descent.\n" +
                    "Too high: training diverges. Too low: training stalls.\n" +
                    "Default 1e-4 is a safe starting point for Adam optimizer.\n" +
                    "Reduce if loss oscillates; increase if training is very slow."));

            grid.add(new Label("Learning Rate:"), 0, row);
            grid.add(learningRateSpinner, 1, row);
            row++;

            // Validation split
            validationSplitSpinner = new Spinner<>(5, 50, DLClassifierPreferences.getValidationSplit(), 5);
            validationSplitSpinner.setEditable(true);
            validationSplitSpinner.setPrefWidth(100);
            validationSplitSpinner.setTooltip(new Tooltip(
                    "Percentage of annotated tiles held out for validation.\n" +
                    "Used to monitor overfitting during training.\n" +
                    "Higher values give more reliable validation metrics\n" +
                    "but leave less data for training. 15-25% is typical."));

            grid.add(new Label("Validation Split (%):"), 0, row);
            grid.add(validationSplitSpinner, 1, row);
            row++;

            // Tile size
            tileSizeSpinner = new Spinner<>(64, 1024, DLClassifierPreferences.getTileSize(), 64);
            tileSizeSpinner.setEditable(true);
            tileSizeSpinner.setPrefWidth(100);
            tileSizeSpinner.setTooltip(new Tooltip(
                    "Size of square patches extracted from annotations for training.\n" +
                    "Must be divisible by 32 (encoder downsampling requirement).\n" +
                    "Larger tiles capture more context but use more memory.\n" +
                    "256 or 512 are common choices."));

            grid.add(new Label("Tile Size:"), 0, row);
            grid.add(tileSizeSpinner, 1, row);
            row++;

            // Overlap
            overlapSpinner = new Spinner<>(0, 50, DLClassifierPreferences.getTileOverlap(), 5);
            overlapSpinner.setEditable(true);
            overlapSpinner.setPrefWidth(100);
            overlapSpinner.setTooltip(new Tooltip(
                    "Overlap between adjacent training tiles as a percentage.\n" +
                    "Higher overlap generates more training patches from\n" +
                    "the same annotations but increases extraction time.\n" +
                    "10-25% is typical."));

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
            lineStrokeWidthSpinner.setTooltip(new Tooltip(
                    "Width in pixels for rendering line/polyline annotations as training masks.\n" +
                    "Pre-filled from QuPath's annotation stroke thickness.\n" +
                    "Thin strokes (< 5px) produce sparse training signal from polyline\n" +
                    "annotations - consider increasing for better training."));

            grid.add(new Label("Line Stroke Width:"), 0, row);
            grid.add(lineStrokeWidthSpinner, 1, row);

            TitledPane pane = new TitledPane("TRAINING PARAMETERS", grid);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(new Tooltip("Configure training hyperparameters and tile extraction settings"));
            return pane;
        }

        private TitledPane createChannelSection() {
            channelPanel = new ChannelSelectionPanel();
            channelPanel.validProperty().addListener((obs, old, valid) -> updateValidation());

            TitledPane pane = new TitledPane("CHANNEL CONFIGURATION", channelPanel);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(new Tooltip("Select and order image channels for model input"));
            return pane;
        }

        private TitledPane createClassSection() {
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            Label infoLabel = new Label("Select annotation classes to use for training:");
            infoLabel.setStyle("-fx-text-fill: #666;");

            classListView = new ListView<>();
            classListView.setCellFactory(lv -> new ClassListCell());
            classListView.setPrefHeight(120);
            classListView.setTooltip(new Tooltip(
                    "Annotation classes found in the current image.\n" +
                    "At least 2 classes must be selected for training.\n" +
                    "Each class should have representative annotations."));

            // Add select all / none buttons
            Button selectAllBtn = new Button("Select All");
            selectAllBtn.setTooltip(new Tooltip("Select all annotation classes for training"));
            selectAllBtn.setOnAction(e -> classListView.getItems().forEach(item -> item.selected().set(true)));

            Button selectNoneBtn = new Button("Select None");
            selectNoneBtn.setTooltip(new Tooltip("Deselect all annotation classes"));
            selectNoneBtn.setOnAction(e -> classListView.getItems().forEach(item -> item.selected().set(false)));

            HBox buttonBox = new HBox(10, selectAllBtn, selectNoneBtn);

            content.getChildren().addAll(infoLabel, classListView, buttonBox);

            TitledPane pane = new TitledPane("ANNOTATION CLASSES", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(new Tooltip("Select which annotation classes to include in training"));
            return pane;
        }

        private TitledPane createAugmentationSection() {
            VBox content = new VBox(8);
            content.setPadding(new Insets(10));

            flipHorizontalCheck = new CheckBox("Horizontal flip");
            flipHorizontalCheck.setSelected(DLClassifierPreferences.isAugFlipHorizontal());
            flipHorizontalCheck.setTooltip(new Tooltip(
                    "Randomly mirror tiles left-right during training.\n" +
                    "Effective for tissue where orientation is arbitrary.\n" +
                    "Almost always beneficial; disable only if horizontal\n" +
                    "orientation carries meaning in your images."));

            flipVerticalCheck = new CheckBox("Vertical flip");
            flipVerticalCheck.setSelected(DLClassifierPreferences.isAugFlipVertical());
            flipVerticalCheck.setTooltip(new Tooltip(
                    "Randomly mirror tiles top-bottom during training.\n" +
                    "Same rationale as horizontal flip.\n" +
                    "Safe to enable for most histopathology images."));

            rotationCheck = new CheckBox("Random rotation (90 deg)");
            rotationCheck.setSelected(DLClassifierPreferences.isAugRotation());
            rotationCheck.setTooltip(new Tooltip(
                    "Randomly rotate tiles by 0/90/180/270 degrees.\n" +
                    "Preserves pixel alignment (no interpolation artifacts).\n" +
                    "Beneficial when tissue structures have no preferred\n" +
                    "orientation. Combines well with flips for 8x augmentation."));

            colorJitterCheck = new CheckBox("Color jitter");
            colorJitterCheck.setSelected(DLClassifierPreferences.isAugColorJitter());
            colorJitterCheck.setTooltip(new Tooltip(
                    "Randomly perturb brightness, contrast, and saturation.\n" +
                    "Helps the model generalize across staining variations\n" +
                    "and illumination differences between slides.\n" +
                    "May not be appropriate for fluorescence or quantitative\n" +
                    "intensity-based classification."));

            elasticCheck = new CheckBox("Elastic deformation");
            elasticCheck.setSelected(DLClassifierPreferences.isAugElasticDeform());
            elasticCheck.setTooltip(new Tooltip(
                    "Apply smooth random spatial deformations to tiles.\n" +
                    "Simulates tissue distortion and cutting artifacts.\n" +
                    "Computationally expensive but effective for histopathology.\n" +
                    "May reduce training speed by ~30%."));

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
            pane.setTooltip(new Tooltip("Configure data augmentation to improve model generalization"));
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

        private void initializeFromCurrentImage() {
            ImageData<BufferedImage> imageData = QP.getCurrentImageData();
            if (imageData != null) {
                // Set up channels
                channelPanel.setImageData(imageData);

                // Auto-select RGB for 3-channel images
                if (imageData.getServer().nChannels() == 3) {
                    // Pre-select all channels for RGB
                }

                // Populate class list from image annotations
                Set<PathClass> classes = new TreeSet<>(Comparator.comparing(PathClass::getName));
                for (PathObject annotation : imageData.getHierarchy().getAnnotationObjects()) {
                    PathClass pathClass = annotation.getPathClass();
                    if (pathClass != null && !pathClass.isDerivedClass()) {
                        classes.add(pathClass);
                    }
                }

                for (PathClass pathClass : classes) {
                    classListView.getItems().add(new ClassItem(pathClass.getName(), pathClass.getColor(), true));
                }
            }

            // Generate default name
            String timestamp = java.time.LocalDate.now().toString().replace("-", "");
            classifierNameField.setText("Classifier_" + timestamp);

            updateValidation();
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
                    .augmentation(buildAugmentationConfig())
                    .usePretrainedWeights(usePretrainedCheck.isSelected())
                    .frozenLayers(frozenLayers)
                    .lineStrokeWidth(lineStrokeWidthSpinner.getValue())
                    .build();

            // Get channel config
            ChannelConfiguration channelConfig = channelPanel.getChannelConfiguration();

            // Get selected classes
            List<String> selectedClasses = classListView.getItems().stream()
                    .filter(item -> item.selected().get())
                    .map(ClassItem::name)
                    .collect(Collectors.toList());

            // Get selected project images (null if current-image-only mode)
            List<ProjectImageEntry<BufferedImage>> selectedImages = null;
            if (projectImagesRadio.isSelected()) {
                selectedImages = imageSelectionList.getItems().stream()
                        .filter(item -> item.selected.get())
                        .map(item -> item.entry)
                        .collect(Collectors.toList());
            }

            return new TrainingDialogResult(
                    classifierNameField.getText().trim(),
                    descriptionField.getText().trim(),
                    trainingConfig,
                    channelConfig,
                    selectedClasses,
                    selectedImages
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
                    .augmentation(buildAugmentationConfig())
                    .usePretrainedWeights(usePretrainedCheck.isSelected())
                    .frozenLayers(frozenLayers)
                    .lineStrokeWidth(lineStrokeWidthSpinner.getValue())
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
    private record ClassItem(String name, Integer color, javafx.beans.property.BooleanProperty selected) {
        public ClassItem(String name, Integer color, boolean selected) {
            this(name, color, new javafx.beans.property.SimpleBooleanProperty(selected));
        }
    }

    /**
     * Custom cell renderer for class items.
     */
    private static class ClassListCell extends ListCell<ClassItem> {
        private final HBox content;
        private final CheckBox checkBox;
        private final javafx.scene.shape.Rectangle colorBox;

        public ClassListCell() {
            checkBox = new CheckBox();
            colorBox = new javafx.scene.shape.Rectangle(16, 16);
            colorBox.setStroke(javafx.scene.paint.Color.BLACK);
            colorBox.setStrokeWidth(1);

            content = new HBox(8, checkBox, colorBox);
            content.setAlignment(Pos.CENTER_LEFT);
        }

        @Override
        protected void updateItem(ClassItem item, boolean empty) {
            super.updateItem(item, empty);

            if (empty || item == null) {
                setGraphic(null);
            } else {
                checkBox.setText(item.name());
                checkBox.selectedProperty().bindBidirectional(item.selected());

                if (item.color() != null) {
                    int r = (item.color() >> 16) & 0xFF;
                    int g = (item.color() >> 8) & 0xFF;
                    int b = item.color() & 0xFF;
                    colorBox.setFill(javafx.scene.paint.Color.rgb(r, g, b));
                } else {
                    colorBox.setFill(javafx.scene.paint.Color.GRAY);
                }

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
     */
    private static class CheckBoxListCell<T> extends ListCell<T> {
        private final java.util.function.Function<T, javafx.beans.property.BooleanProperty> selectedExtractor;
        private final java.util.function.Function<T, String> textExtractor;
        private final CheckBox checkBox = new CheckBox();

        CheckBoxListCell(java.util.function.Function<T, javafx.beans.property.BooleanProperty> selectedExtractor,
                         java.util.function.Function<T, String> textExtractor) {
            this.selectedExtractor = selectedExtractor;
            this.textExtractor = textExtractor;
        }

        @Override
        protected void updateItem(T item, boolean empty) {
            super.updateItem(item, empty);
            if (empty || item == null) {
                setGraphic(null);
            } else {
                checkBox.setText(textExtractor.apply(item));
                checkBox.selectedProperty().unbind();
                checkBox.selectedProperty().bindBidirectional(selectedExtractor.apply(item));
                setGraphic(checkBox);
            }
        }
    }
}
