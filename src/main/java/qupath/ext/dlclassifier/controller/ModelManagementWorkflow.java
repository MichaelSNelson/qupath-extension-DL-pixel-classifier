package qupath.ext.dlclassifier.controller;

import javafx.application.Platform;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;

import java.nio.file.Path;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Optional;

/**
 * Workflow for managing trained classifiers.
 * <p>
 * This workflow allows users to:
 * <ul>
 *   <li>Browse available classifiers</li>
 *   <li>View classifier details and metadata</li>
 *   <li>Delete classifiers</li>
 *   <li>Export classifiers (future)</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ModelManagementWorkflow {

    private static final Logger logger = LoggerFactory.getLogger(ModelManagementWorkflow.class);
    private static final DateTimeFormatter DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");

    private final QuPathGUI qupath;
    private final ModelManager modelManager;

    private Stage dialogStage;
    private TableView<ClassifierMetadata> classifierTable;
    private ObservableList<ClassifierMetadata> classifierList;
    private VBox detailsPane;

    public ModelManagementWorkflow() {
        this.qupath = QuPathGUI.getInstance();
        this.modelManager = new ModelManager();
    }

    /**
     * Starts the model management workflow.
     */
    public void start() {
        logger.info("Starting model management workflow");
        Platform.runLater(this::showModelManagerDialog);
    }

    /**
     * Shows the model manager dialog.
     */
    private void showModelManagerDialog() {
        dialogStage = new Stage();
        dialogStage.setTitle("DL Classifier Manager");
        dialogStage.initModality(Modality.APPLICATION_MODAL);
        dialogStage.initOwner(qupath.getStage());

        BorderPane root = new BorderPane();
        root.setPadding(new Insets(10));

        // Header
        Label headerLabel = new Label("Trained Classifiers");
        headerLabel.setStyle("-fx-font-size: 16px; -fx-font-weight: bold;");
        BorderPane.setMargin(headerLabel, new Insets(0, 0, 10, 0));
        root.setTop(headerLabel);

        // Main content: split between table and details
        SplitPane splitPane = new SplitPane();
        splitPane.setDividerPositions(0.55);

        // Left: classifier table
        VBox tableBox = createTablePane();
        splitPane.getItems().add(tableBox);

        // Right: details pane
        detailsPane = createDetailsPane();
        ScrollPane detailsScroll = new ScrollPane(detailsPane);
        detailsScroll.setFitToWidth(true);
        detailsScroll.setStyle("-fx-background-color: transparent;");
        splitPane.getItems().add(detailsScroll);

        root.setCenter(splitPane);

        // Bottom: action buttons
        HBox buttonBox = createButtonPane();
        BorderPane.setMargin(buttonBox, new Insets(10, 0, 0, 0));
        root.setBottom(buttonBox);

        Scene scene = new Scene(root, 800, 500);
        dialogStage.setScene(scene);

        // Load classifiers
        refreshClassifierList();

        dialogStage.showAndWait();
    }

    /**
     * Creates the table pane with classifier list.
     */
    private VBox createTablePane() {
        VBox box = new VBox(5);

        classifierList = FXCollections.observableArrayList();
        classifierTable = new TableView<>(classifierList);
        classifierTable.setPlaceholder(new Label("No classifiers found"));

        // Name column
        TableColumn<ClassifierMetadata, String> nameCol = new TableColumn<>("Name");
        nameCol.setCellValueFactory(data -> new SimpleStringProperty(data.getValue().getName()));
        nameCol.setPrefWidth(150);

        // Type column
        TableColumn<ClassifierMetadata, String> typeCol = new TableColumn<>("Architecture");
        typeCol.setCellValueFactory(data -> {
            ClassifierMetadata m = data.getValue();
            String arch = m.getModelType().toUpperCase();
            if (m.getBackbone() != null && !m.getBackbone().isEmpty()) {
                arch += " / " + m.getBackbone();
            }
            return new SimpleStringProperty(arch);
        });
        typeCol.setPrefWidth(140);

        // Classes column
        TableColumn<ClassifierMetadata, String> classesCol = new TableColumn<>("Classes");
        classesCol.setCellValueFactory(data ->
                new SimpleStringProperty(String.valueOf(data.getValue().getNumClasses())));
        classesCol.setPrefWidth(60);

        // Created column
        TableColumn<ClassifierMetadata, String> createdCol = new TableColumn<>("Created");
        createdCol.setCellValueFactory(data -> {
            var created = data.getValue().getCreatedAt();
            String dateStr = created != null ? created.format(DATE_FORMAT) : "Unknown";
            return new SimpleStringProperty(dateStr);
        });
        createdCol.setPrefWidth(120);

        classifierTable.getColumns().addAll(nameCol, typeCol, classesCol, createdCol);

        // Selection listener
        classifierTable.getSelectionModel().selectedItemProperty().addListener(
                (obs, oldVal, newVal) -> updateDetailsPane(newVal));

        VBox.setVgrow(classifierTable, Priority.ALWAYS);
        box.getChildren().add(classifierTable);

        // Refresh button
        Button refreshBtn = new Button("Refresh");
        refreshBtn.setOnAction(e -> refreshClassifierList());
        box.getChildren().add(refreshBtn);

        return box;
    }

    /**
     * Creates the details pane.
     */
    private VBox createDetailsPane() {
        VBox box = new VBox(10);
        box.setPadding(new Insets(10));
        box.setStyle("-fx-background-color: #f8f8f8; -fx-border-color: #ddd; -fx-border-radius: 5;");

        Label placeholder = new Label("Select a classifier to view details");
        placeholder.setStyle("-fx-text-fill: #888;");
        box.getChildren().add(placeholder);

        return box;
    }

    /**
     * Updates the details pane with classifier information.
     */
    private void updateDetailsPane(ClassifierMetadata metadata) {
        detailsPane.getChildren().clear();

        if (metadata == null) {
            Label placeholder = new Label("Select a classifier to view details");
            placeholder.setStyle("-fx-text-fill: #888;");
            detailsPane.getChildren().add(placeholder);
            return;
        }

        // Name and description
        Label nameLabel = new Label(metadata.getName());
        nameLabel.setStyle("-fx-font-size: 14px; -fx-font-weight: bold;");
        detailsPane.getChildren().add(nameLabel);

        if (metadata.getDescription() != null && !metadata.getDescription().isEmpty()) {
            Label descLabel = new Label(metadata.getDescription());
            descLabel.setWrapText(true);
            descLabel.setStyle("-fx-text-fill: #666;");
            detailsPane.getChildren().add(descLabel);
        }

        detailsPane.getChildren().add(new Separator());

        // Architecture section
        Label archHeader = new Label("Architecture");
        archHeader.setStyle("-fx-font-weight: bold;");
        detailsPane.getChildren().add(archHeader);

        GridPane archGrid = new GridPane();
        archGrid.setHgap(10);
        archGrid.setVgap(5);
        int row = 0;

        addDetailRow(archGrid, row++, "Model Type:", metadata.getModelType().toUpperCase());
        addDetailRow(archGrid, row++, "Backbone:", metadata.getBackbone());
        addDetailRow(archGrid, row++, "Input Size:",
                metadata.getInputWidth() + " x " + metadata.getInputHeight());
        addDetailRow(archGrid, row++, "Input Channels:", String.valueOf(metadata.getInputChannels()));

        detailsPane.getChildren().add(archGrid);

        // Classes section
        detailsPane.getChildren().add(new Separator());
        Label classesHeader = new Label("Classes (" + metadata.getNumClasses() + ")");
        classesHeader.setStyle("-fx-font-weight: bold;");
        detailsPane.getChildren().add(classesHeader);

        VBox classesBox = new VBox(3);
        for (ClassifierMetadata.ClassInfo classInfo : metadata.getClasses()) {
            HBox classRow = new HBox(8);
            classRow.setAlignment(Pos.CENTER_LEFT);

            // Color swatch
            Rectangle colorSwatch = new Rectangle(16, 16);
            try {
                colorSwatch.setFill(Color.web(classInfo.color()));
            } catch (Exception e) {
                colorSwatch.setFill(Color.GRAY);
            }
            colorSwatch.setStroke(Color.DARKGRAY);
            colorSwatch.setStrokeWidth(1);

            Label classLabel = new Label(classInfo.index() + ": " + classInfo.name());
            classRow.getChildren().addAll(colorSwatch, classLabel);
            classesBox.getChildren().add(classRow);
        }
        detailsPane.getChildren().add(classesBox);

        // Training info section
        if (metadata.getTrainingEpochs() > 0) {
            detailsPane.getChildren().add(new Separator());
            Label trainingHeader = new Label("Training Info");
            trainingHeader.setStyle("-fx-font-weight: bold;");
            detailsPane.getChildren().add(trainingHeader);

            GridPane trainingGrid = new GridPane();
            trainingGrid.setHgap(10);
            trainingGrid.setVgap(5);
            row = 0;

            addDetailRow(trainingGrid, row++, "Epochs:", String.valueOf(metadata.getTrainingEpochs()));
            addDetailRow(trainingGrid, row++, "Final Loss:", String.format("%.4f", metadata.getFinalLoss()));
            addDetailRow(trainingGrid, row++, "Final Accuracy:",
                    String.format("%.1f%%", metadata.getFinalAccuracy() * 100));

            if (metadata.getTrainingImageName() != null && !metadata.getTrainingImageName().isEmpty()) {
                addDetailRow(trainingGrid, row++, "Training Image:", metadata.getTrainingImageName());
            }

            detailsPane.getChildren().add(trainingGrid);
        }

        // Channel config section
        if (!metadata.getExpectedChannelNames().isEmpty()) {
            detailsPane.getChildren().add(new Separator());
            Label channelHeader = new Label("Channel Configuration");
            channelHeader.setStyle("-fx-font-weight: bold;");
            detailsPane.getChildren().add(channelHeader);

            GridPane channelGrid = new GridPane();
            channelGrid.setHgap(10);
            channelGrid.setVgap(5);
            row = 0;

            addDetailRow(channelGrid, row++, "Channels:",
                    String.join(", ", metadata.getExpectedChannelNames()));
            addDetailRow(channelGrid, row++, "Normalization:",
                    metadata.getNormalizationStrategy().name());
            addDetailRow(channelGrid, row++, "Bit Depth:",
                    metadata.getBitDepthTrained() + "-bit");

            detailsPane.getChildren().add(channelGrid);
        }

        // Model file info
        detailsPane.getChildren().add(new Separator());
        Label fileHeader = new Label("Model Files");
        fileHeader.setStyle("-fx-font-weight: bold;");
        detailsPane.getChildren().add(fileHeader);

        Optional<Path> modelPath = modelManager.getModelPath(metadata.getId());
        if (modelPath.isPresent()) {
            Path path = modelPath.get();
            String fileType = path.getFileName().toString().endsWith(".onnx") ? "ONNX" : "PyTorch";
            Label pathLabel = new Label(fileType + ": " + path.getParent().toString());
            pathLabel.setWrapText(true);
            pathLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
            detailsPane.getChildren().add(pathLabel);
        } else {
            Label noModelLabel = new Label("Model file not found");
            noModelLabel.setStyle("-fx-text-fill: #c00;");
            detailsPane.getChildren().add(noModelLabel);
        }
    }

    /**
     * Adds a label-value row to a grid.
     */
    private void addDetailRow(GridPane grid, int row, String label, String value) {
        Label labelNode = new Label(label);
        labelNode.setStyle("-fx-text-fill: #666;");
        Label valueNode = new Label(value != null ? value : "-");
        grid.add(labelNode, 0, row);
        grid.add(valueNode, 1, row);
    }

    /**
     * Creates the button pane.
     */
    private HBox createButtonPane() {
        HBox box = new HBox(10);
        box.setAlignment(Pos.CENTER_RIGHT);

        Button deleteBtn = new Button("Delete");
        deleteBtn.setOnAction(e -> deleteSelectedClassifier());
        deleteBtn.disableProperty().bind(
                classifierTable.getSelectionModel().selectedItemProperty().isNull());

        Button exportBtn = new Button("Export...");
        exportBtn.setOnAction(e -> exportSelectedClassifier());
        exportBtn.setDisable(true); // Not yet implemented
        exportBtn.setTooltip(new Tooltip("Export classifier (coming soon)"));

        Button closeBtn = new Button("Close");
        closeBtn.setOnAction(e -> dialogStage.close());

        // Spacer
        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        // Info label
        Label infoLabel = new Label("");
        infoLabel.setStyle("-fx-text-fill: #888;");

        box.getChildren().addAll(infoLabel, spacer, deleteBtn, exportBtn, closeBtn);
        return box;
    }

    /**
     * Refreshes the classifier list.
     */
    private void refreshClassifierList() {
        try {
            List<ClassifierMetadata> classifiers = modelManager.listClassifiers();
            classifierList.setAll(classifiers);
            logger.info("Loaded {} classifiers", classifiers.size());

            // Clear selection
            classifierTable.getSelectionModel().clearSelection();
            updateDetailsPane(null);

        } catch (Exception e) {
            logger.error("Failed to load classifiers", e);
            Dialogs.showErrorMessage("Error", "Failed to load classifiers: " + e.getMessage());
        }
    }

    /**
     * Deletes the selected classifier.
     */
    private void deleteSelectedClassifier() {
        ClassifierMetadata selected = classifierTable.getSelectionModel().getSelectedItem();
        if (selected == null) {
            return;
        }

        boolean confirm = Dialogs.showConfirmDialog(
                "Delete Classifier",
                "Are you sure you want to delete the classifier '" + selected.getName() + "'?\n\n" +
                        "This action cannot be undone."
        );

        if (!confirm) {
            return;
        }

        try {
            boolean deleted = modelManager.deleteClassifier(selected.getId());
            if (deleted) {
                logger.info("Deleted classifier: {}", selected.getId());
                Dialogs.showInfoNotification("Deleted", "Classifier deleted successfully");
                refreshClassifierList();
            } else {
                Dialogs.showWarningNotification("Warning", "Could not delete classifier");
            }
        } catch (Exception e) {
            logger.error("Failed to delete classifier", e);
            Dialogs.showErrorMessage("Error", "Failed to delete classifier: " + e.getMessage());
        }
    }

    /**
     * Exports the selected classifier (placeholder for future implementation).
     */
    private void exportSelectedClassifier() {
        ClassifierMetadata selected = classifierTable.getSelectionModel().getSelectedItem();
        if (selected == null) {
            return;
        }

        // TODO: Implement export as zip archive
        Dialogs.showInfoNotification(
                "Export",
                "Classifier export is not yet implemented.\n" +
                        "The classifier files can be found at:\n" +
                        modelManager.getModelPath(selected.getId())
                                .map(Path::getParent)
                                .map(Path::toString)
                                .orElse("Unknown location")
        );
    }
}
