package qupath.ext.dlclassifier.ui;

import javafx.application.Platform;
import javafx.beans.property.*;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.collections.transformation.FilteredList;
import javafx.collections.transformation.SortedList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.objects.hierarchy.events.PathObjectHierarchyEvent;
import qupath.lib.objects.hierarchy.events.PathObjectHierarchyListener;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Modeless dialog showing per-tile evaluation results from post-training analysis.
 * <p>
 * Displays tiles sorted by loss (descending) to help users identify annotation
 * errors, hard cases, and model failures. Selecting a row navigates the
 * QuPath viewer to the tile location and shows a temporary highlight rectangle
 * that auto-clears when the user starts drawing.
 * A preview pane shows the loss heatmap or disagreement map overlay with zoom.
 *
 * @author UW-LOCI
 * @since 0.3.0
 */
public class TrainingAreaIssuesDialog {

    private static final Logger logger = LoggerFactory.getLogger(TrainingAreaIssuesDialog.class);

    private final Stage stage;
    private final TableView<TileRow> table;
    private final ObservableList<TileRow> allRows;
    private final FilteredList<TileRow> filteredRows;
    private final Label summaryLabel;
    private final double downsample;
    private final int patchSize;
    private final Map<String, Integer> classColors;

    // Tile highlight tracking -- auto-removed when user interacts with hierarchy
    private PathObject currentHighlight;
    private qupath.lib.images.ImageData<?> highlightImageData;
    private PathObjectHierarchyListener hierarchyListener;

    // Preview pane components
    private final ImageView tileImageView;
    private final ImageView disagreeImageView;
    private final VBox legendBox;
    private final ComboBox<String> overlaySelector;
    private static final String OVERLAY_DISAGREEMENT = "Disagreement";
    private static final String OVERLAY_LOSS_HEATMAP = "Loss Heatmap";

    /**
     * Creates the training area issues dialog.
     *
     * @param classifierName name of the classifier for the title
     * @param results        per-tile evaluation results sorted by loss descending
     * @param downsample     downsample factor used during training
     * @param patchSize      training patch size in pixels (at the downsampled resolution)
     * @param classColors    map of class name to packed RGB color, or null
     */
    public TrainingAreaIssuesDialog(String classifierName,
                                    List<ClassifierClient.TileEvaluationResult> results,
                                    double downsample,
                                    int patchSize,
                                    Map<String, Integer> classColors) {
        this.downsample = downsample;
        this.patchSize = patchSize;
        this.classColors = classColors != null ? classColors : Map.of();
        this.stage = new Stage();
        stage.initStyle(StageStyle.DECORATED);
        var qupath = QuPathGUI.getInstance();
        if (qupath != null && qupath.getStage() != null) {
            stage.initOwner(qupath.getStage());
        }
        stage.setTitle("Training Area Issues - " + classifierName);
        stage.setResizable(true);

        // Convert results to observable rows
        allRows = FXCollections.observableArrayList();
        for (var r : results) {
            allRows.add(new TileRow(r));
        }
        filteredRows = new FilteredList<>(allRows, row -> true);
        SortedList<TileRow> sortedRows = new SortedList<>(filteredRows);

        // Summary label
        long highLoss = results.stream().filter(r -> r.loss() > 1.0).count();
        summaryLabel = new Label(String.format(
                "%d tiles evaluated | %d with loss > 1.0", results.size(), highLoss));
        summaryLabel.setStyle("-fx-font-weight: bold;");
        summaryLabel.setTooltip(TooltipHelper.create(
                "Total tiles evaluated and count of tiles with high loss.\n"
                + "Tiles are sorted by loss (worst first) to help find\n"
                + "annotation errors and hard cases."));

        // Filter controls
        ComboBox<String> splitFilter = new ComboBox<>();
        splitFilter.getItems().addAll("All", "Train", "Val");
        splitFilter.setValue("All");
        splitFilter.setTooltip(TooltipHelper.create(
                "Filter tiles by dataset split.\n"
                + "Val tiles are more diagnostic -- high loss there\n"
                + "suggests annotation problems, not just overfitting."));

        Slider thresholdSlider = new Slider(0, 10, 0);
        thresholdSlider.setShowTickLabels(true);
        thresholdSlider.setShowTickMarks(true);
        thresholdSlider.setMajorTickUnit(2);
        thresholdSlider.setMinorTickCount(1);
        thresholdSlider.setPrefWidth(200);
        thresholdSlider.setTooltip(TooltipHelper.create(
                "Show only tiles with loss above this threshold.\n"
                + "Increase to focus on the most problematic tiles."));

        Label thresholdLabel = new Label("Min Loss: 0.00");
        thresholdSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            thresholdLabel.setText(String.format("Min Loss: %.2f", newVal.doubleValue()));
            updateFilter(splitFilter.getValue(), newVal.doubleValue());
        });

        splitFilter.setOnAction(e -> updateFilter(splitFilter.getValue(),
                thresholdSlider.getValue()));

        HBox filterBox = new HBox(10,
                new Label("Filter:"), splitFilter,
                thresholdLabel, thresholdSlider);
        filterBox.setAlignment(Pos.CENTER_LEFT);

        // Table
        table = new TableView<>();
        sortedRows.comparatorProperty().bind(table.comparatorProperty());
        table.setItems(sortedRows);
        table.setTooltip(TooltipHelper.create(
                "Click a row to navigate to that tile in the viewer\n"
                + "and see the loss heatmap preview.\n"
                + "A temporary highlight rectangle marks the tile boundary\n"
                + "and clears automatically when you start drawing."));

        TableColumn<TileRow, String> imageCol = new TableColumn<>("Image");
        imageCol.setCellValueFactory(new PropertyValueFactory<>("sourceImage"));
        imageCol.setPrefWidth(120);

        TableColumn<TileRow, String> splitCol = new TableColumn<>("Split");
        splitCol.setCellValueFactory(new PropertyValueFactory<>("split"));
        splitCol.setPrefWidth(50);

        TableColumn<TileRow, Double> lossCol = new TableColumn<>("Loss");
        lossCol.setCellValueFactory(new PropertyValueFactory<>("loss"));
        lossCol.setPrefWidth(70);
        lossCol.setCellFactory(col -> new FormattedDoubleCell<>("%.3f"));
        lossCol.setSortType(TableColumn.SortType.DESCENDING);

        TableColumn<TileRow, Double> disagreeCol = new TableColumn<>("Disagree%");
        disagreeCol.setCellValueFactory(new PropertyValueFactory<>("disagreementPct"));
        disagreeCol.setPrefWidth(80);
        disagreeCol.setCellFactory(col -> new FormattedDoubleCell<>("%5.1f%%", 100.0));

        TableColumn<TileRow, Double> iouCol = new TableColumn<>("mIoU");
        iouCol.setCellValueFactory(new PropertyValueFactory<>("meanIoU"));
        iouCol.setPrefWidth(65);
        iouCol.setCellFactory(col -> new FormattedDoubleCell<>("%.3f"));

        TableColumn<TileRow, String> worstClassCol = new TableColumn<>("Worst Class");
        worstClassCol.setCellValueFactory(new PropertyValueFactory<>("worstClass"));
        worstClassCol.setPrefWidth(130);

        // Add column tooltips via graphic labels
        setColumnTooltip(imageCol, "Source image this tile was extracted from.");
        setColumnTooltip(splitCol, "Train or Val split. High-loss Val tiles\nare the best candidates for annotation review.");
        setColumnTooltip(lossCol, "Per-tile loss value. Higher = model struggled more.\nSort by this column to find the worst tiles.");
        setColumnTooltip(disagreeCol, "Percentage of pixels where the model's\nprediction differs from the ground truth annotation.");
        setColumnTooltip(iouCol, "Mean Intersection-over-Union across all\nclasses present in this tile (higher is better).");
        setColumnTooltip(worstClassCol, "Class with the lowest IoU in this tile.\nShows which class the model is struggling with\nand how poorly it performed (IoU score).");

        table.getColumns().addAll(List.of(imageCol, splitCol, lossCol, disagreeCol,
                iouCol, worstClassCol));
        table.getSortOrder().add(lossCol);

        // Single-click navigates to tile and updates preview
        table.getSelectionModel().selectedItemProperty().addListener((obs, oldRow, newRow) -> {
            if (newRow != null) {
                navigateToTile(newRow);
                updatePreview(newRow);
            } else {
                clearPreview();
            }
        });

        // Re-clicking the same row should re-navigate
        table.setOnMouseClicked(e -> {
            TileRow selected = table.getSelectionModel().getSelectedItem();
            if (selected != null) {
                navigateToTile(selected);
            }
        });

        // Status bar
        Label statusLabel = new Label("Click a row to navigate to the tile location");
        statusLabel.setStyle("-fx-text-fill: #666; -fx-font-size: 11px;");

        // Preview pane with zoom support
        tileImageView = new ImageView();
        tileImageView.setPreserveRatio(true);
        tileImageView.setSmooth(false);

        disagreeImageView = new ImageView();
        disagreeImageView.setPreserveRatio(true);
        disagreeImageView.setSmooth(false);
        disagreeImageView.setOpacity(0.6);

        // Zoomable preview: images in a Group scaled by zoom, inside a ScrollPane
        StackPane imageStack = new StackPane(tileImageView, disagreeImageView);

        ScrollPane previewScroll = new ScrollPane(imageStack);
        previewScroll.setPannable(true);
        previewScroll.setStyle("-fx-background-color: #222;");
        previewScroll.setPrefSize(280, 280);
        previewScroll.setMinSize(280, 280);
        previewScroll.setMaxSize(Double.MAX_VALUE, Double.MAX_VALUE);
        previewScroll.setFitToWidth(true);
        previewScroll.setFitToHeight(true);

        // Zoom slider
        Label zoomLabel = new Label("Zoom: 1x");
        Slider zoomSlider = new Slider(1, 8, 1);
        zoomSlider.setShowTickLabels(true);
        zoomSlider.setShowTickMarks(true);
        zoomSlider.setMajorTickUnit(1);
        zoomSlider.setMinorTickCount(0);
        zoomSlider.setSnapToTicks(true);
        zoomSlider.setPrefWidth(200);
        zoomSlider.setTooltip(TooltipHelper.create(
                "Zoom into the preview to see loss details at higher resolution.\n"
                + "Use the scrollbars or drag to pan when zoomed in.\n"
                + "Helps identify which specific pixels have high loss."));
        zoomSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            double zoom = newVal.doubleValue();
            zoomLabel.setText(String.format("Zoom: %dx", Math.round(zoom)));
            double size = 256 * zoom;
            tileImageView.setFitWidth(size);
            tileImageView.setFitHeight(size);
            disagreeImageView.setFitWidth(size);
            disagreeImageView.setFitHeight(size);
            // Disable fit-to-viewport when zoomed so scrollbars appear
            previewScroll.setFitToWidth(zoom <= 1);
            previewScroll.setFitToHeight(zoom <= 1);
        });

        // Initialize image sizes at 1x
        tileImageView.setFitWidth(256);
        tileImageView.setFitHeight(256);
        disagreeImageView.setFitWidth(256);
        disagreeImageView.setFitHeight(256);

        Label opacityLabel = new Label("Overlay: 60%");
        Slider opacitySlider = new Slider(0, 100, 60);
        opacitySlider.setPrefWidth(200);
        opacitySlider.setShowTickLabels(true);
        opacitySlider.setMajorTickUnit(25);
        opacitySlider.setTooltip(TooltipHelper.create(
                "Adjust the overlay transparency.\n"
                + "Lower values show more of the original tile;\n"
                + "higher values show more of the loss/disagreement overlay."));
        opacitySlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            double opacity = newVal.doubleValue() / 100.0;
            disagreeImageView.setOpacity(opacity);
            opacityLabel.setText(String.format("Overlay: %.0f%%", newVal.doubleValue()));
        });

        legendBox = new VBox(3);
        legendBox.setPadding(new Insets(5, 0, 0, 0));
        buildLossHeatmapLegend();

        Label previewTitle = new Label("Loss Heatmap Preview");
        previewTitle.setStyle("-fx-font-weight: bold; -fx-font-size: 12px;");

        overlaySelector = new ComboBox<>();
        overlaySelector.getItems().addAll(OVERLAY_LOSS_HEATMAP, OVERLAY_DISAGREEMENT);
        overlaySelector.setValue(OVERLAY_LOSS_HEATMAP);
        overlaySelector.setTooltip(TooltipHelper.create(
                "Loss Heatmap: per-pixel loss intensity (blue=low, red=high).\n"
                + "Disagreement: colored pixels where model prediction\n"
                + "differs from the ground truth annotation."));
        overlaySelector.setOnAction(e -> {
            String selected = overlaySelector.getValue();
            previewTitle.setText(selected + " Preview");
            if (OVERLAY_LOSS_HEATMAP.equals(selected)) {
                buildLossHeatmapLegend();
            } else {
                buildDisagreementLegend();
            }
            TileRow currentRow = table.getSelectionModel().getSelectedItem();
            if (currentRow != null) {
                updateOverlayImage(currentRow);
            }
        });

        HBox titleBar = new HBox(8, previewTitle, overlaySelector);
        titleBar.setAlignment(Pos.CENTER_LEFT);

        VBox previewPane = new VBox(8, titleBar, previewScroll,
                zoomLabel, zoomSlider,
                opacityLabel, opacitySlider, legendBox);
        previewPane.setPadding(new Insets(10, 0, 0, 10));
        previewPane.setAlignment(Pos.TOP_CENTER);
        previewPane.setMinWidth(300);
        previewPane.setPrefWidth(300);
        previewPane.setMaxWidth(300);

        // Layout: table on left, preview on right
        VBox tablePane = new VBox(10, summaryLabel, filterBox, table, statusLabel);
        tablePane.setPadding(new Insets(15));
        VBox.setVgrow(table, Priority.ALWAYS);
        HBox.setHgrow(tablePane, Priority.ALWAYS);

        HBox mainLayout = new HBox(0, tablePane, previewPane);
        mainLayout.setPadding(new Insets(0, 10, 10, 0));
        HBox.setHgrow(tablePane, Priority.ALWAYS);

        Scene scene = new Scene(mainLayout, 900, 600);
        stage.setScene(scene);

        // Clean up highlight on close
        stage.setOnHidden(e -> removeCurrentHighlight());
    }

    /**
     * Shows the dialog.
     */
    public void show() {
        Platform.runLater(() -> stage.show());
    }

    /**
     * Sets a tooltip on a table column header via a Label graphic.
     */
    private static <S, T> void setColumnTooltip(TableColumn<S, T> column, String text) {
        Label label = new Label(column.getText());
        label.setTooltip(TooltipHelper.create(text));
        column.setGraphic(label);
        column.setText("");
    }

    private void updateFilter(String splitValue, double minLoss) {
        filteredRows.setPredicate(row -> {
            if (!"All".equals(splitValue)) {
                String expected = splitValue.toLowerCase();
                if (!row.getSplit().equalsIgnoreCase(expected)) {
                    return false;
                }
            }
            return row.getLoss() >= minLoss;
        });

        long visible = filteredRows.size();
        long highLoss = filteredRows.stream().filter(r -> r.getLoss() > 1.0).count();
        summaryLabel.setText(String.format(
                "%d tiles shown | %d with loss > 1.0", visible, highLoss));
    }

    private void navigateToTile(TileRow row) {
        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null) return;

        String targetImageId = row.getSourceImageId();
        String targetImageName = row.getSourceImage();
        var project = qupath.getProject();

        if (project != null && targetImageName != null && !targetImageName.isEmpty()) {
            var currentImageData = qupath.getImageData();
            String currentImageName = currentImageData != null
                    ? currentImageData.getServer().getMetadata().getName() : null;
            boolean needsSwitch = !targetImageName.equals(currentImageName);

            if (needsSwitch) {
                for (var entry : project.getImageList()) {
                    boolean match = targetImageId != null && !targetImageId.isEmpty()
                            ? targetImageId.equals(entry.getID())
                            : targetImageName.equals(entry.getImageName());
                    if (match) {
                        Platform.runLater(() -> {
                            try {
                                qupath.openImageEntry(entry);
                                Platform.runLater(() -> centerViewerOnTile(qupath, row));
                            } catch (Exception e) {
                                logger.warn("Failed to open image: {}", e.getMessage());
                            }
                        });
                        return;
                    }
                }
                logger.warn("Could not find image '{}' in project", targetImageName);
            }
        }

        Platform.runLater(() -> centerViewerOnTile(qupath, row));
    }

    private void centerViewerOnTile(QuPathGUI qupath, TileRow row) {
        QuPathViewer viewer = qupath.getViewer();
        if (viewer == null) return;

        var imageData = viewer.getImageData();
        if (imageData == null) return;

        double regionSize = this.patchSize * downsample;
        double centerX = row.getX() + regionSize / 2.0;
        double centerY = row.getY() + regionSize / 2.0;

        viewer.setCenterPixelLocation(centerX, centerY);
        viewer.setDownsampleFactor(downsample);

        // Add temporary highlight rectangle that auto-clears on user interaction
        addTemporaryHighlight(imageData, row.getX(), row.getY(), regionSize);
    }

    /**
     * Adds a temporary highlight rectangle that auto-clears when the user
     * modifies the hierarchy (e.g., starts drawing a new annotation).
     * This prevents the highlight from interfering with drawing tools.
     */
    private void addTemporaryHighlight(qupath.lib.images.ImageData<?> imageData,
                                       int tileX, int tileY, double regionSize) {
        removeCurrentHighlight();

        try {
            var roi = ROIs.createRectangleROI(tileX, tileY, regionSize, regionSize,
                    ImagePlane.getDefaultPlane());
            var highlight = PathObjects.createAnnotationObject(roi,
                    PathClass.fromString("Region*"));
            highlight.setLocked(true);

            var hierarchy = imageData.getHierarchy();
            hierarchy.addObject(highlight);
            hierarchy.getSelectionModel().setSelectedObject(highlight);

            currentHighlight = highlight;
            highlightImageData = imageData;

            // Listen for hierarchy changes -- remove highlight when user
            // adds/removes/modifies any other object (i.e., starts drawing)
            hierarchyListener = new PathObjectHierarchyListener() {
                @Override
                public void hierarchyChanged(PathObjectHierarchyEvent event) {
                    // Ignore events caused by our own highlight management
                    if (event.getChangedObjects().size() == 1
                            && event.getChangedObjects().contains(currentHighlight)) {
                        return;
                    }
                    // User did something else -- clear our highlight
                    Platform.runLater(() -> removeCurrentHighlight());
                }
            };
            hierarchy.addListener(hierarchyListener);

        } catch (Exception e) {
            logger.debug("Failed to create tile highlight: {}", e.getMessage());
        }
    }

    /**
     * Removes the current highlight rectangle and hierarchy listener.
     */
    private void removeCurrentHighlight() {
        if (hierarchyListener != null && highlightImageData != null) {
            try {
                highlightImageData.getHierarchy().removeListener(hierarchyListener);
            } catch (Exception e) {
                // Ignore
            }
            hierarchyListener = null;
        }
        if (currentHighlight != null && highlightImageData != null) {
            try {
                highlightImageData.getHierarchy().removeObject(currentHighlight, false);
            } catch (Exception e) {
                logger.debug("Failed to remove highlight: {}", e.getMessage());
            }
            currentHighlight = null;
            highlightImageData = null;
        }
    }

    // ==================== Preview Pane ====================

    private void updatePreview(TileRow row) {
        String tilePath = row.getTileImagePath();

        if (tilePath != null && !tilePath.isEmpty()) {
            try {
                File tileFile = new File(tilePath);
                if (tileFile.exists()) {
                    Image tileImage = new Image(tileFile.toURI().toString());
                    tileImageView.setImage(tileImage);
                } else {
                    tileImageView.setImage(null);
                }
            } catch (Exception e) {
                logger.debug("Failed to load tile image: {}", e.getMessage());
                tileImageView.setImage(null);
            }
        } else {
            tileImageView.setImage(null);
        }

        updateOverlayImage(row);
    }

    private void updateOverlayImage(TileRow row) {
        String overlayPath;
        if (OVERLAY_LOSS_HEATMAP.equals(overlaySelector.getValue())) {
            overlayPath = row.getLossHeatmapPath();
        } else {
            overlayPath = row.getDisagreementImagePath();
        }

        if (overlayPath != null && !overlayPath.isEmpty()) {
            try {
                File overlayFile = new File(overlayPath);
                if (overlayFile.exists()) {
                    Image overlayImage = new Image(overlayFile.toURI().toString());
                    disagreeImageView.setImage(overlayImage);
                } else {
                    disagreeImageView.setImage(null);
                }
            } catch (Exception e) {
                logger.debug("Failed to load overlay image: {}", e.getMessage());
                disagreeImageView.setImage(null);
            }
        } else {
            disagreeImageView.setImage(null);
        }
    }

    private void clearPreview() {
        tileImageView.setImage(null);
        disagreeImageView.setImage(null);
    }

    private void buildDisagreementLegend() {
        legendBox.getChildren().clear();
        if (classColors.isEmpty()) return;

        Label legendTitle = new Label("Class Colors:");
        legendTitle.setStyle("-fx-font-size: 11px; -fx-text-fill: #888;");
        legendBox.getChildren().add(legendTitle);

        for (Map.Entry<String, Integer> entry : classColors.entrySet()) {
            int packed = entry.getValue() & 0xFFFFFF;
            int r = (packed >> 16) & 0xFF;
            int g = (packed >> 8) & 0xFF;
            int b = packed & 0xFF;

            Rectangle swatch = new Rectangle(12, 12);
            swatch.setFill(Color.rgb(r, g, b));
            swatch.setStroke(Color.gray(0.5));
            swatch.setStrokeWidth(0.5);

            Label name = new Label(entry.getKey());
            name.setStyle("-fx-font-size: 11px;");

            HBox legendItem = new HBox(5, swatch, name);
            legendItem.setAlignment(Pos.CENTER_LEFT);
            legendBox.getChildren().add(legendItem);
        }
    }

    private void buildLossHeatmapLegend() {
        legendBox.getChildren().clear();

        Label legendTitle = new Label("Loss Intensity:");
        legendTitle.setStyle("-fx-font-size: 11px; -fx-text-fill: #888;");
        legendBox.getChildren().add(legendTitle);

        Region gradientBar = new Region();
        gradientBar.setPrefHeight(14);
        gradientBar.setPrefWidth(200);
        gradientBar.setMaxWidth(200);
        gradientBar.setStyle(
                "-fx-background-color: linear-gradient(to right, #0000FF, #FFFF00, #FF0000);"
                + " -fx-border-color: #666; -fx-border-width: 0.5;");

        Label lowLabel = new Label("Low");
        lowLabel.setStyle("-fx-font-size: 10px; -fx-text-fill: #888;");
        Label highLabel = new Label("High");
        highLabel.setStyle("-fx-font-size: 10px; -fx-text-fill: #888;");
        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        HBox labels = new HBox(lowLabel, spacer, highLabel);
        labels.setMaxWidth(200);
        labels.setPrefWidth(200);

        legendBox.getChildren().addAll(gradientBar, labels);
    }

    // ==================== Helper Classes ====================

    /**
     * Table cell that formats doubles with a format string.
     */
    private static class FormattedDoubleCell<S> extends TableCell<S, Double> {
        private final String format;
        private final double multiplier;

        FormattedDoubleCell(String format) {
            this(format, 1.0);
        }

        FormattedDoubleCell(String format, double multiplier) {
            this.format = format;
            this.multiplier = multiplier;
        }

        @Override
        protected void updateItem(Double item, boolean empty) {
            super.updateItem(item, empty);
            if (empty || item == null) {
                setText(null);
            } else {
                setText(String.format(format, item * multiplier));
            }
        }
    }

    /**
     * Row model for the evaluation results table.
     */
    public static class TileRow {
        private final StringProperty sourceImage;
        private final StringProperty sourceImageId;
        private final StringProperty split;
        private final DoubleProperty loss;
        private final DoubleProperty disagreementPct;
        private final DoubleProperty meanIoU;
        private final StringProperty worstClass;
        private final IntegerProperty x;
        private final IntegerProperty y;
        private final StringProperty filename;
        private final StringProperty disagreementImagePath;
        private final StringProperty lossHeatmapPath;
        private final StringProperty tileImagePath;

        public TileRow(ClassifierClient.TileEvaluationResult result) {
            this.sourceImage = new SimpleStringProperty(result.sourceImage());
            this.sourceImageId = new SimpleStringProperty(result.sourceImageId());
            this.split = new SimpleStringProperty(result.split());
            this.loss = new SimpleDoubleProperty(result.loss());
            this.disagreementPct = new SimpleDoubleProperty(result.disagreementPct());
            this.meanIoU = new SimpleDoubleProperty(result.meanIoU());
            this.x = new SimpleIntegerProperty(result.x());
            this.y = new SimpleIntegerProperty(result.y());
            this.filename = new SimpleStringProperty(result.filename());
            this.disagreementImagePath = new SimpleStringProperty(result.disagreementImagePath());
            this.lossHeatmapPath = new SimpleStringProperty(result.lossHeatmapPath());
            this.tileImagePath = new SimpleStringProperty(result.tileImagePath());

            // Compute worst class: lowest IoU among classes actually present in the tile.
            // Null IoU values indicate the class has no ground truth pixels in this tile
            // and are excluded. Only consider classes with real IoU measurements.
            String worst = "";
            double worstIoU = Double.MAX_VALUE;
            if (result.perClassIoU() != null) {
                for (Map.Entry<String, Double> entry : result.perClassIoU().entrySet()) {
                    Double iou = entry.getValue();
                    if (iou != null && iou < worstIoU) {
                        worstIoU = iou;
                        worst = entry.getKey();
                    }
                }
            }
            this.worstClass = new SimpleStringProperty(
                    worst.isEmpty() ? "" : String.format("%s (IoU %.3f)", worst, worstIoU));
        }

        public String getSourceImage() { return sourceImage.get(); }
        public StringProperty sourceImageProperty() { return sourceImage; }

        public String getSourceImageId() { return sourceImageId.get(); }

        public String getSplit() { return split.get(); }
        public StringProperty splitProperty() { return split; }

        public double getLoss() { return loss.get(); }
        public DoubleProperty lossProperty() { return loss; }

        public double getDisagreementPct() { return disagreementPct.get(); }
        public DoubleProperty disagreementPctProperty() { return disagreementPct; }

        public double getMeanIoU() { return meanIoU.get(); }
        public DoubleProperty meanIoUProperty() { return meanIoU; }

        public String getWorstClass() { return worstClass.get(); }
        public StringProperty worstClassProperty() { return worstClass; }

        public int getX() { return x.get(); }
        public int getY() { return y.get(); }

        public String getFilename() { return filename.get(); }

        public String getDisagreementImagePath() { return disagreementImagePath.get(); }
        public String getLossHeatmapPath() { return lossHeatmapPath.get(); }
        public String getTileImagePath() { return tileImagePath.get(); }
    }
}
