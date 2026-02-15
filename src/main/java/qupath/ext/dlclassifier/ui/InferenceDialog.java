package qupath.ext.dlclassifier.ui;

import javafx.animation.PauseTransition;
import javafx.application.Platform;
import javafx.beans.property.SimpleStringProperty;
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
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.InferenceConfig.OutputObjectType;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.scripting.ScriptGenerator;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.lib.images.ImageData;
import qupath.lib.scripting.QP;

import java.awt.image.BufferedImage;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.CompletableFuture;

/**
 * Dialog for configuring deep learning classifier inference.
 * <p>
 * This dialog provides an interface for:
 * <ul>
 *   <li>Classifier selection from available models</li>
 *   <li>Output type configuration (measurements, objects, overlay)</li>
 *   <li>Channel mapping for multi-channel images</li>
 *   <li>Post-processing options</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class InferenceDialog {

    private static final Logger logger = LoggerFactory.getLogger(InferenceDialog.class);

    /**
     * Result of the inference dialog.
     */
    public record InferenceDialogResult(
            ClassifierMetadata classifier,
            InferenceConfig inferenceConfig,
            ChannelConfiguration channelConfig,
            InferenceConfig.ApplicationScope applicationScope,
            boolean createBackup
    ) {}

    private InferenceDialog() {
        // Utility class
    }

    /**
     * Shows the inference configuration dialog.
     *
     * @return CompletableFuture with the result, or cancelled if user cancels
     */
    public static CompletableFuture<InferenceDialogResult> showDialog() {
        CompletableFuture<InferenceDialogResult> future = new CompletableFuture<>();

        Platform.runLater(() -> {
            try {
                InferenceDialogBuilder builder = new InferenceDialogBuilder();
                Optional<InferenceDialogResult> result = builder.buildAndShow();
                if (result.isPresent()) {
                    future.complete(result.get());
                } else {
                    future.complete(null);
                }
            } catch (Exception e) {
                logger.error("Error showing inference dialog", e);
                future.completeExceptionally(e);
            }
        });

        return future;
    }

    /**
     * Inner builder class for constructing the dialog.
     */
    private static class InferenceDialogBuilder {

        private final ModelManager modelManager = new ModelManager();
        private Dialog<InferenceDialogResult> dialog;

        // Classifier selection
        private TableView<ClassifierMetadata> classifierTable;
        private Label classifierInfoLabel;

        // Output options
        private ComboBox<InferenceConfig.OutputType> outputTypeCombo;
        private ComboBox<OutputObjectType> objectTypeCombo;
        private Spinner<Double> minObjectSizeSpinner;
        private Spinner<Double> holeFillingSpinner;
        private Spinner<Double> smoothingSpinner;

        // Channel configuration
        private ChannelSelectionPanel channelPanel;

        // Processing options
        private Spinner<Integer> tileSizeSpinner;
        private Spinner<Integer> overlapSpinner;
        private Spinner<Double> overlapPercentSpinner;
        private Label overlapWarningLabel;
        private ComboBox<InferenceConfig.BlendMode> blendModeCombo;
        private CheckBox useGPUCheck;

        // Channel section (for brightfield auto-configuration)
        private TitledPane channelSectionPane;

        // Scope options
        private RadioButton applyToSelectedRadio;
        private RadioButton applyToAllRadio;
        private RadioButton applyToWholeImageRadio;
        private CheckBox createBackupCheck;

        private Button okButton;

        public Optional<InferenceDialogResult> buildAndShow() {
            dialog = new Dialog<>();
            dialog.initModality(Modality.APPLICATION_MODAL);
            dialog.setTitle("Apply DL Pixel Classifier");
            dialog.setResizable(true);

            // Create header
            createHeader();

            // Create button types
            ButtonType applyType = new ButtonType("Apply", ButtonBar.ButtonData.OK_DONE);
            ButtonType cancelType = new ButtonType("Cancel", ButtonBar.ButtonData.CANCEL_CLOSE);
            ButtonType copyScriptType = new ButtonType("Copy as Script", ButtonBar.ButtonData.LEFT);
            dialog.getDialogPane().getButtonTypes().addAll(copyScriptType, applyType, cancelType);

            okButton = (Button) dialog.getDialogPane().lookupButton(applyType);
            okButton.setDisable(true);

            // Wire up the "Copy as Script" button
            Button copyScriptButton = (Button) dialog.getDialogPane().lookupButton(copyScriptType);
            copyScriptButton.addEventFilter(ActionEvent.ACTION, event -> {
                event.consume(); // Prevent dialog from closing
                copyInferenceScript(copyScriptButton);
            });

            // Create content
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            content.getChildren().addAll(
                    createClassifierSection(),
                    createOutputSection(),
                    createChannelSection(),
                    createProcessingSection(),
                    createScopeSection()
            );

            ScrollPane scrollPane = new ScrollPane(content);
            scrollPane.setFitToWidth(true);
            scrollPane.setPrefHeight(550);
            scrollPane.setPrefWidth(600);

            dialog.getDialogPane().setContent(scrollPane);

            // Load available classifiers
            loadClassifiers();

            // Initialize with current image
            initializeFromCurrentImage();

            // Set result converter
            dialog.setResultConverter(button -> {
                if (button != applyType) {
                    return null;
                }
                return buildResult();
            });

            return dialog.showAndWait();
        }

        private void createHeader() {
            VBox headerBox = new VBox(5);
            headerBox.setPadding(new Insets(10));

            Label titleLabel = new Label("Apply Classification to Image");
            titleLabel.setStyle("-fx-font-size: 14px; -fx-font-weight: bold;");

            Label subtitleLabel = new Label("Select a trained classifier and configure output options");
            subtitleLabel.setStyle("-fx-text-fill: #666;");

            headerBox.getChildren().addAll(titleLabel, subtitleLabel, new Separator());
            dialog.getDialogPane().setHeader(headerBox);
        }

        private TitledPane createClassifierSection() {
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            // Create classifier table
            classifierTable = new TableView<>();
            classifierTable.setPrefHeight(150);
            classifierTable.setPlaceholder(new Label("No classifiers available. Train a classifier first."));
            classifierTable.setTooltip(new Tooltip(
                    "Available trained classifiers.\n" +
                    "Select one to apply to the current image."));

            TableColumn<ClassifierMetadata, String> nameCol = new TableColumn<>("Name");
            nameCol.setCellValueFactory(data -> new SimpleStringProperty(data.getValue().getName()));
            nameCol.setPrefWidth(150);

            TableColumn<ClassifierMetadata, String> typeCol = new TableColumn<>("Type");
            typeCol.setCellValueFactory(data -> new SimpleStringProperty(data.getValue().getModelType()));
            typeCol.setPrefWidth(80);

            TableColumn<ClassifierMetadata, String> channelsCol = new TableColumn<>("Channels");
            channelsCol.setCellValueFactory(data -> new SimpleStringProperty(
                    String.valueOf(data.getValue().getInputChannels())));
            channelsCol.setPrefWidth(70);

            TableColumn<ClassifierMetadata, String> classesCol = new TableColumn<>("Classes");
            classesCol.setCellValueFactory(data -> new SimpleStringProperty(
                    String.valueOf(data.getValue().getClassNames().size())));
            classesCol.setPrefWidth(60);

            TableColumn<ClassifierMetadata, String> dateCol = new TableColumn<>("Trained");
            dateCol.setCellValueFactory(data -> {
                var created = data.getValue().getCreatedAt();
                if (created != null) {
                    return new SimpleStringProperty(created.format(DateTimeFormatter.ofPattern("yyyy-MM-dd")));
                }
                return new SimpleStringProperty("-");
            });
            dateCol.setPrefWidth(90);

            classifierTable.getColumns().addAll(List.of(nameCol, typeCol, channelsCol, classesCol, dateCol));

            // Selection listener
            classifierTable.getSelectionModel().selectedItemProperty().addListener(
                    (obs, old, selected) -> onClassifierSelected(selected));

            // Info label
            classifierInfoLabel = new Label("Select a classifier to see details");
            classifierInfoLabel.setStyle("-fx-text-fill: #666; -fx-font-style: italic;");
            classifierInfoLabel.setWrapText(true);

            content.getChildren().addAll(classifierTable, classifierInfoLabel);

            TitledPane pane = new TitledPane("SELECT CLASSIFIER", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(new Tooltip("Choose a trained classifier to apply"));
            return pane;
        }

        private TitledPane createOutputSection() {
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

            // Output type - restore from preferences
            outputTypeCombo = new ComboBox<>(FXCollections.observableArrayList(InferenceConfig.OutputType.values()));
            try {
                outputTypeCombo.setValue(InferenceConfig.OutputType.valueOf(DLClassifierPreferences.getLastOutputType()));
            } catch (IllegalArgumentException e) {
                outputTypeCombo.setValue(InferenceConfig.OutputType.MEASUREMENTS);
            }
            outputTypeCombo.setTooltip(new Tooltip(
                    "How classification results are represented:\n" +
                    "MEASUREMENTS: Add per-class probability measurements to annotations.\n" +
                    "OBJECTS: Create detection/annotation objects for classified regions.\n" +
                    "OVERLAY: Display classification as a live color overlay on the viewer."));
            outputTypeCombo.valueProperty().addListener((obs, old, newVal) -> updateOutputOptions(newVal));

            grid.add(new Label("Output Type:"), 0, row);
            grid.add(outputTypeCombo, 1, row);
            row++;

            // Object type (DETECTION vs ANNOTATION) - only for OBJECTS output
            objectTypeCombo = new ComboBox<>(FXCollections.observableArrayList(OutputObjectType.values()));
            objectTypeCombo.setValue(OutputObjectType.DETECTION);
            objectTypeCombo.setTooltip(new Tooltip(
                    "QuPath object type for classified regions:\n" +
                    "DETECTION: Lightweight, non-editable objects for quantification.\n" +
                    "ANNOTATION: Editable objects that can be further classified\n" +
                    "or used as parent objects for nested analysis."));

            grid.add(new Label("Object Type:"), 0, row);
            grid.add(objectTypeCombo, 1, row);
            row++;

            // Min object size (for OBJECTS output)
            minObjectSizeSpinner = new Spinner<>(0.0, 10000.0, 10.0, 1.0);
            minObjectSizeSpinner.setEditable(true);
            minObjectSizeSpinner.setPrefWidth(100);
            minObjectSizeSpinner.setTooltip(new Tooltip(
                    "Minimum area threshold in um^2 for generated objects.\n" +
                    "Objects smaller than this are discarded as noise.\n" +
                    "Set to 0 to keep all objects regardless of size."));

            grid.add(new Label("Min Object Size (um2):"), 0, row);
            grid.add(minObjectSizeSpinner, 1, row);
            row++;

            // Hole filling
            holeFillingSpinner = new Spinner<>(0.0, 1000.0, 5.0, 1.0);
            holeFillingSpinner.setEditable(true);
            holeFillingSpinner.setPrefWidth(100);
            holeFillingSpinner.setTooltip(new Tooltip(
                    "Fill interior holes in objects smaller than this area (um^2).\n" +
                    "Removes small gaps caused by misclassified pixels\n" +
                    "within otherwise solid regions. Set to 0 to disable."));

            grid.add(new Label("Hole Filling (um2):"), 0, row);
            grid.add(holeFillingSpinner, 1, row);
            row++;

            // Smoothing - restore from preferences
            smoothingSpinner = new Spinner<>(0.0, 10.0, DLClassifierPreferences.getSmoothing(), 0.5);
            smoothingSpinner.setEditable(true);
            smoothingSpinner.setPrefWidth(100);
            smoothingSpinner.setTooltip(new Tooltip(
                    "Boundary simplification tolerance in microns.\n" +
                    "Smooths jagged object boundaries using topology-preserving\n" +
                    "simplification. Higher values produce simpler boundaries.\n" +
                    "Set to 0 for pixel-exact boundaries."));

            grid.add(new Label("Boundary Smoothing:"), 0, row);
            grid.add(smoothingSpinner, 1, row);

            // Set object-specific options based on restored output type
            updateOutputOptions(outputTypeCombo.getValue());

            TitledPane pane = new TitledPane("OUTPUT OPTIONS", grid);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(new Tooltip("Configure how classification results are generated"));
            return pane;
        }

        private TitledPane createChannelSection() {
            channelPanel = new ChannelSelectionPanel();
            channelPanel.validProperty().addListener((obs, old, valid) -> updateValidation());

            channelSectionPane = new TitledPane("CHANNEL MAPPING", channelPanel);
            channelSectionPane.setExpanded(true);
            channelSectionPane.setStyle("-fx-font-weight: bold;");
            channelSectionPane.setTooltip(new Tooltip("Map image channels to classifier input channels"));
            return channelSectionPane;
        }

        private TitledPane createProcessingSection() {
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

            // Tile size
            tileSizeSpinner = new Spinner<>(64, 1024, DLClassifierPreferences.getTileSize(), 64);
            tileSizeSpinner.setEditable(true);
            tileSizeSpinner.setPrefWidth(100);
            tileSizeSpinner.setTooltip(new Tooltip(
                    "Tile size in pixels for inference processing.\n" +
                    "Auto-set to match the classifier's training tile size.\n" +
                    "Must be divisible by 32. Larger tiles may improve\n" +
                    "context but use more GPU memory."));

            grid.add(new Label("Tile Size:"), 0, row);
            grid.add(tileSizeSpinner, 1, row);
            row++;

            // Overlap percent (preferred - percentage based)
            overlapPercentSpinner = new Spinner<>(0.0, 20.0, DLClassifierPreferences.getTileOverlapPercent(), 2.5);
            overlapPercentSpinner.setEditable(true);
            overlapPercentSpinner.setPrefWidth(100);
            overlapPercentSpinner.setTooltip(new Tooltip(
                    "Tile overlap as percentage of tile size (0-20%)\n" +
                    "Higher values improve blending but increase processing time"));
            overlapPercentSpinner.valueProperty().addListener((obs, old, newVal) -> updateOverlapWarning(newVal));

            grid.add(new Label("Tile Overlap (%):"), 0, row);
            grid.add(overlapPercentSpinner, 1, row);
            row++;

            // Overlap warning label
            overlapWarningLabel = new Label();
            overlapWarningLabel.setWrapText(true);
            overlapWarningLabel.setMaxWidth(350);
            overlapWarningLabel.setStyle("-fx-font-size: 11px;");
            grid.add(overlapWarningLabel, 0, row, 2, 1);
            row++;

            // Keep legacy overlap spinner hidden but functional
            overlapSpinner = new Spinner<>(0, 256, DLClassifierPreferences.getTileOverlap(), 8);
            // Don't add to grid - it's computed from overlapPercentSpinner

            // Blend mode - restore from preferences
            blendModeCombo = new ComboBox<>(FXCollections.observableArrayList(InferenceConfig.BlendMode.values()));
            try {
                blendModeCombo.setValue(InferenceConfig.BlendMode.valueOf(DLClassifierPreferences.getLastBlendMode()));
            } catch (IllegalArgumentException e) {
                blendModeCombo.setValue(InferenceConfig.BlendMode.LINEAR);
            }
            blendModeCombo.setTooltip(new Tooltip(
                    "Strategy for merging predictions in overlapping tile regions:\n" +
                    "LINEAR: Weighted average favoring tile centers.\n" +
                    "GAUSSIAN: Gaussian-weighted blending for smoother transitions.\n" +
                    "NONE: No blending; last tile wins (fastest, may show seams)."));

            grid.add(new Label("Blend Mode:"), 0, row);
            grid.add(blendModeCombo, 1, row);
            row++;

            // GPU
            useGPUCheck = new CheckBox("Use GPU if available");
            useGPUCheck.setSelected(DLClassifierPreferences.isUseGPU());
            useGPUCheck.setTooltip(new Tooltip(
                    "Run inference on GPU (CUDA) if available.\n" +
                    "Typically 10-50x faster than CPU.\n" +
                    "Requires CUDA-enabled PyTorch on the server.\n" +
                    "Falls back to CPU automatically if GPU is unavailable."));

            grid.add(useGPUCheck, 0, row, 2, 1);

            // Initialize warning based on default value
            updateOverlapWarning(overlapPercentSpinner.getValue());

            TitledPane pane = new TitledPane("PROCESSING OPTIONS", grid);
            pane.setExpanded(false); // Collapsed by default
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(new Tooltip("Configure tile processing, blending, and GPU settings"));
            return pane;
        }

        private void updateOverlapWarning(double overlapPercent) {
            if (overlapPercent == 0.0) {
                overlapWarningLabel.setText("WARNING: Objects will NOT be merged across tile boundaries");
                overlapWarningLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #D32F2F;");
            } else if (overlapPercent < 10.0) {
                overlapWarningLabel.setText("Note: Low overlap may result in visible seams in output");
                overlapWarningLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #F57C00;");
            } else {
                overlapWarningLabel.setText("Good overlap for seamless blending");
                overlapWarningLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #388E3C;");
            }

            // Update the pixel-based overlap value
            int tileSize = tileSizeSpinner != null ? tileSizeSpinner.getValue() : 512;
            int overlapPixels = (int) Math.round(tileSize * overlapPercent / 100.0);
            if (overlapSpinner != null) {
                overlapSpinner.getValueFactory().setValue(overlapPixels);
            }
        }

        private TitledPane createScopeSection() {
            VBox content = new VBox(10);
            content.setPadding(new Insets(10));

            // Application scope
            ToggleGroup scopeGroup = new ToggleGroup();

            // Restore saved scope from preferences
            InferenceConfig.ApplicationScope savedScope;
            try {
                savedScope = InferenceConfig.ApplicationScope.valueOf(
                        DLClassifierPreferences.getApplicationScope());
            } catch (IllegalArgumentException e) {
                savedScope = InferenceConfig.ApplicationScope.ALL_ANNOTATIONS;
            }

            applyToWholeImageRadio = new RadioButton("Apply to whole image");
            applyToWholeImageRadio.setToggleGroup(scopeGroup);
            applyToWholeImageRadio.setSelected(savedScope == InferenceConfig.ApplicationScope.WHOLE_IMAGE);
            applyToWholeImageRadio.setTooltip(new Tooltip(
                    "Classify the entire image without requiring annotations.\n" +
                    "Recommended for overlay output or full-image classification."));

            applyToAllRadio = new RadioButton("Apply to all annotations");
            applyToAllRadio.setToggleGroup(scopeGroup);
            applyToAllRadio.setSelected(savedScope == InferenceConfig.ApplicationScope.ALL_ANNOTATIONS);
            applyToAllRadio.setTooltip(new Tooltip(
                    "Classify within all annotations in the image.\n" +
                    "Processes every annotation regardless of selection state."));

            applyToSelectedRadio = new RadioButton("Apply to selected annotations only");
            applyToSelectedRadio.setToggleGroup(scopeGroup);
            applyToSelectedRadio.setSelected(savedScope == InferenceConfig.ApplicationScope.SELECTED_ANNOTATIONS);
            applyToSelectedRadio.setTooltip(new Tooltip(
                    "Only classify within the currently selected annotations.\n" +
                    "Useful for testing on a small region before full-image inference."));

            // Backup option - restore from preferences
            createBackupCheck = new CheckBox("Create backup of annotation measurements before applying");
            createBackupCheck.setSelected(DLClassifierPreferences.isCreateBackup());
            createBackupCheck.setTooltip(new Tooltip(
                    "Save a copy of existing annotation measurements before\n" +
                    "overwriting with new classification results.\n" +
                    "Recommended when re-running inference on previously classified images."));

            content.getChildren().addAll(
                    new Label("Application scope:"),
                    applyToWholeImageRadio,
                    applyToAllRadio,
                    applyToSelectedRadio,
                    new Separator(),
                    createBackupCheck
            );

            TitledPane pane = new TitledPane("APPLICATION SCOPE", content);
            pane.setExpanded(true);
            pane.setStyle("-fx-font-weight: bold;");
            pane.setTooltip(new Tooltip("Control which region to classify and backup options"));
            return pane;
        }

        private void loadClassifiers() {
            List<ClassifierMetadata> classifiers = modelManager.listClassifiers();
            classifierTable.setItems(FXCollections.observableArrayList(classifiers));

            if (!classifiers.isEmpty()) {
                classifierTable.getSelectionModel().selectFirst();
            }
        }

        private void initializeFromCurrentImage() {
            ImageData<BufferedImage> imageData = QP.getCurrentImageData();
            if (imageData != null) {
                channelPanel.setImageData(imageData);

                // Auto-select all channels and collapse section for brightfield images
                if (isBrightfield(imageData)) {
                    channelPanel.selectAllChannels();
                    channelSectionPane.setExpanded(false);
                }
            }
        }

        private boolean isBrightfield(ImageData<BufferedImage> imageData) {
            ImageData.ImageType type = imageData.getImageType();
            return type == ImageData.ImageType.BRIGHTFIELD_H_E
                    || type == ImageData.ImageType.BRIGHTFIELD_H_DAB
                    || type == ImageData.ImageType.BRIGHTFIELD_OTHER;
        }

        private void onClassifierSelected(ClassifierMetadata classifier) {
            if (classifier == null) {
                classifierInfoLabel.setText("Select a classifier to see details");
                classifierInfoLabel.setStyle("-fx-text-fill: #666; -fx-font-style: italic;");
                channelPanel.setRequiredChannelCount(-1);
                okButton.setDisable(true);
                return;
            }

            // Update info label
            StringBuilder info = new StringBuilder();
            info.append("Architecture: ").append(classifier.getModelType());
            if (classifier.getBackbone() != null) {
                info.append(" (").append(classifier.getBackbone()).append(")");
            }
            info.append("\n");
            info.append("Input: ").append(classifier.getInputChannels()).append(" channels, ");
            info.append(classifier.getInputWidth()).append("x").append(classifier.getInputHeight()).append(" tiles\n");
            info.append("Classes: ").append(String.join(", ", classifier.getClassNames()));

            classifierInfoLabel.setText(info.toString());
            classifierInfoLabel.setStyle("-fx-text-fill: #333;");

            // Update channel requirements
            channelPanel.setRequiredChannelCount(classifier.getInputChannels());
            channelPanel.setExpectedChannels(classifier.getExpectedChannelNames());

            // Update tile size to match classifier
            tileSizeSpinner.getValueFactory().setValue(classifier.getInputWidth());

            updateValidation();
        }

        private void updateOutputOptions(InferenceConfig.OutputType outputType) {
            boolean enableObjectOptions = (outputType == InferenceConfig.OutputType.OBJECTS);
            objectTypeCombo.setDisable(!enableObjectOptions);
            minObjectSizeSpinner.setDisable(!enableObjectOptions);
            holeFillingSpinner.setDisable(!enableObjectOptions);
            smoothingSpinner.setDisable(!enableObjectOptions);

            // Auto-select "Apply to whole image" when overlay is chosen
            if (outputType == InferenceConfig.OutputType.OVERLAY
                    && applyToWholeImageRadio != null) {
                applyToWholeImageRadio.setSelected(true);
            }
        }

        private void updateValidation() {
            ClassifierMetadata selected = classifierTable.getSelectionModel().getSelectedItem();
            boolean valid = selected != null && channelPanel.isValid();
            okButton.setDisable(!valid);
        }

        private InferenceDialogResult buildResult() {
            // Determine selected scope
            InferenceConfig.ApplicationScope scope;
            if (applyToWholeImageRadio.isSelected()) {
                scope = InferenceConfig.ApplicationScope.WHOLE_IMAGE;
            } else if (applyToSelectedRadio.isSelected()) {
                scope = InferenceConfig.ApplicationScope.SELECTED_ANNOTATIONS;
            } else {
                scope = InferenceConfig.ApplicationScope.ALL_ANNOTATIONS;
            }

            // Save dialog settings to preferences for next session
            DLClassifierPreferences.setLastOutputType(outputTypeCombo.getValue().name());
            DLClassifierPreferences.setLastBlendMode(blendModeCombo.getValue().name());
            DLClassifierPreferences.setSmoothing(smoothingSpinner.getValue());
            DLClassifierPreferences.setApplicationScope(scope.name());
            DLClassifierPreferences.setCreateBackup(createBackupCheck.isSelected());

            ClassifierMetadata classifier = classifierTable.getSelectionModel().getSelectedItem();

            // Calculate pixel overlap from percentage
            int overlapPixels = (int) Math.round(
                    tileSizeSpinner.getValue() * overlapPercentSpinner.getValue() / 100.0);

            InferenceConfig inferenceConfig = InferenceConfig.builder()
                    .tileSize(tileSizeSpinner.getValue())
                    .overlap(overlapPixels)
                    .overlapPercent(overlapPercentSpinner.getValue())
                    .blendMode(blendModeCombo.getValue())
                    .outputType(outputTypeCombo.getValue())
                    .objectType(objectTypeCombo.getValue())
                    .minObjectSize(minObjectSizeSpinner.getValue())
                    .holeFilling(holeFillingSpinner.getValue())
                    .smoothing(smoothingSpinner.getValue())
                    .useGPU(useGPUCheck.isSelected())
                    .build();

            ChannelConfiguration channelConfig = channelPanel.getChannelConfiguration();

            return new InferenceDialogResult(
                    classifier,
                    inferenceConfig,
                    channelConfig,
                    scope,
                    createBackupCheck.isSelected()
            );
        }

        private void copyInferenceScript(Button sourceButton) {
            ClassifierMetadata classifier = classifierTable.getSelectionModel().getSelectedItem();
            if (classifier == null) {
                showCopyFeedback(sourceButton, "No classifier selected");
                return;
            }

            // Build current config from dialog state
            int overlapPixels = (int) Math.round(
                    tileSizeSpinner.getValue() * overlapPercentSpinner.getValue() / 100.0);

            InferenceConfig config = InferenceConfig.builder()
                    .tileSize(tileSizeSpinner.getValue())
                    .overlap(overlapPixels)
                    .blendMode(blendModeCombo.getValue())
                    .outputType(outputTypeCombo.getValue())
                    .objectType(objectTypeCombo.getValue())
                    .minObjectSize(minObjectSizeSpinner.getValue())
                    .holeFilling(holeFillingSpinner.getValue())
                    .smoothing(smoothingSpinner.getValue())
                    .useGPU(useGPUCheck.isSelected())
                    .build();

            ChannelConfiguration channelConfig = channelPanel.getChannelConfiguration();

            InferenceConfig.ApplicationScope scope;
            if (applyToWholeImageRadio.isSelected()) {
                scope = InferenceConfig.ApplicationScope.WHOLE_IMAGE;
            } else if (applyToSelectedRadio.isSelected()) {
                scope = InferenceConfig.ApplicationScope.SELECTED_ANNOTATIONS;
            } else {
                scope = InferenceConfig.ApplicationScope.ALL_ANNOTATIONS;
            }

            String script = ScriptGenerator.generateInferenceScript(
                    classifier.getId(), config, channelConfig, scope);

            Clipboard clipboard = Clipboard.getSystemClipboard();
            ClipboardContent content = new ClipboardContent();
            content.putString(script);
            clipboard.setContent(content);

            showCopyFeedback(sourceButton, "Script copied to clipboard!");
        }

        private void showCopyFeedback(Button button, String message) {
            Tooltip tooltip = new Tooltip(message);
            tooltip.setAutoHide(true);
            tooltip.show(button, //
                    button.localToScreen(button.getBoundsInLocal()).getMinX(),
                    button.localToScreen(button.getBoundsInLocal()).getMinY() - 30);
            PauseTransition pause = new PauseTransition(Duration.seconds(2));
            pause.setOnFinished(e -> tooltip.hide());
            pause.play();
        }
    }
}
