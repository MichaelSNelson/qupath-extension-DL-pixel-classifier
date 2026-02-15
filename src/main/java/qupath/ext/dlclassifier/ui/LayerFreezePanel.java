package qupath.ext.dlclassifier.ui;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.service.ClassifierClient;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Panel for configuring which encoder layers to freeze during transfer learning.
 * <p>
 * This panel displays the layer structure of the selected model and allows
 * users to choose which layers to freeze (not train) vs. train. Earlier layers
 * typically capture general features and can be frozen, while later layers
 * are more task-specific and should be trained.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class LayerFreezePanel extends VBox {

    private static final Logger logger = LoggerFactory.getLogger(LayerFreezePanel.class);

    private final ObservableList<LayerItem> layers = FXCollections.observableArrayList();
    private final ListView<LayerItem> layerListView;
    private final ComboBox<String> presetCombo;
    private final Label statusLabel;

    private String currentArchitecture;
    private String currentEncoder;
    private ClassifierClient client;

    /**
     * Creates a new layer freeze panel.
     */
    public LayerFreezePanel() {
        setSpacing(10);
        setPadding(new Insets(10));

        // Header with info
        Label headerLabel = new Label("Transfer Learning Configuration");
        headerLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        Label infoLabel = new Label(
                "Early layers learn universal features (edges, textures) that transfer well. " +
                "Later layers learn high-level concepts that need retraining for histopathology."
        );
        infoLabel.setWrapText(true);
        infoLabel.setStyle("-fx-text-fill: #666666;");

        // Preset selection - based on data available (affects overfitting risk)
        HBox presetBox = new HBox(10);
        presetBox.setAlignment(Pos.CENTER_LEFT);
        Label presetLabel = new Label("Training Data:");
        presetCombo = new ComboBox<>();
        presetCombo.getItems().addAll(
                "Small (<500 tiles) - Conservative",
                "Medium (500-5000) - Balanced",
                "Large (>5000) - Full adaptation",
                "Custom"
        );
        presetCombo.setValue("Medium (500-5000) - Balanced");
        presetCombo.setOnAction(e -> applyPreset());
        presetCombo.setTooltip(new Tooltip(
                "Select a freeze strategy based on your dataset size:\n" +
                "Small (<500 tiles): Freeze most encoder layers to prevent overfitting.\n" +
                "Medium (500-5000): Balanced freeze of early layers only.\n" +
                "Large (>5000): Fine-tune nearly all layers for best adaptation.\n" +
                "Custom: Manually toggle individual layers below."));

        Button applyButton = new Button("Apply");
        applyButton.setTooltip(new Tooltip("Apply the selected freeze preset to all layers"));
        applyButton.setOnAction(e -> applyPreset());
        presetBox.getChildren().addAll(presetLabel, presetCombo, applyButton);

        // Layer list
        layerListView = new ListView<>(layers);
        layerListView.setCellFactory(lv -> new LayerCell());
        layerListView.setPrefHeight(200);
        layerListView.setTooltip(new Tooltip(
                "Model layers from early (top) to late (bottom).\n" +
                "Check a layer to freeze (skip training) it.\n" +
                "Green = early/general features, Red = late/specific features."));
        VBox.setVgrow(layerListView, Priority.ALWAYS);

        // Quick actions
        HBox actionBox = new HBox(10);
        actionBox.setAlignment(Pos.CENTER);

        Button freezeAllEncoderBtn = new Button("Freeze All Encoder");
        freezeAllEncoderBtn.setTooltip(new Tooltip(
                "Freeze all encoder layers.\n" +
                "Only the decoder will be trained.\n" +
                "Most conservative option, best for very small datasets."));
        freezeAllEncoderBtn.setOnAction(e -> setAllEncoderLayers(true));

        Button unfreezeAllBtn = new Button("Unfreeze All");
        unfreezeAllBtn.setTooltip(new Tooltip(
                "Unfreeze all layers for full fine-tuning.\n" +
                "Most aggressive option, best for large datasets.\n" +
                "Risk of overfitting with small datasets."));
        unfreezeAllBtn.setOnAction(e -> setAllLayers(false));

        Button recommendedBtn = new Button("Use Recommended");
        recommendedBtn.setTooltip(new Tooltip(
                "Apply the server's recommended freeze configuration\n" +
                "based on the selected architecture and backbone."));
        recommendedBtn.setOnAction(e -> applyRecommended());

        actionBox.getChildren().addAll(freezeAllEncoderBtn, unfreezeAllBtn, recommendedBtn);

        // Status
        statusLabel = new Label("Select architecture and encoder to view layers");
        statusLabel.setStyle("-fx-text-fill: #888888;");

        getChildren().addAll(headerLabel, infoLabel, presetBox, layerListView, actionBox, statusLabel);
    }

    /**
     * Sets the classifier client for fetching layer information.
     */
    public void setClient(ClassifierClient client) {
        this.client = client;
    }

    /**
     * Loads layers for the specified architecture and encoder.
     *
     * @param architecture model architecture (e.g., "unet")
     * @param encoder      encoder name (e.g., "resnet34")
     * @param numChannels  number of input channels
     * @param numClasses   number of output classes
     */
    public void loadLayers(String architecture, String encoder, int numChannels, int numClasses) {
        this.currentArchitecture = architecture;
        this.currentEncoder = encoder;

        layers.clear();

        if (client == null) {
            statusLabel.setText("No server connection");
            return;
        }

        try {
            statusLabel.setText("Loading layer structure...");

            List<ClassifierClient.LayerInfo> layerInfos = client.getModelLayers(
                    architecture, encoder, numChannels, numClasses);

            for (ClassifierClient.LayerInfo info : layerInfos) {
                LayerItem item = new LayerItem(
                        info.name(),
                        info.displayName(),
                        info.paramCount(),
                        info.isEncoder(),
                        info.depth(),
                        info.recommendedFreeze(),
                        info.description()
                );
                item.setFrozen(info.recommendedFreeze());
                layers.add(item);
            }

            updateStatus();
            logger.info("Loaded {} layers for {}/{}", layers.size(), architecture, encoder);

        } catch (Exception e) {
            logger.error("Failed to load layers", e);
            statusLabel.setText("Error: " + e.getMessage());
        }
    }

    /**
     * Gets the list of layer names that should be frozen.
     */
    public List<String> getFrozenLayerNames() {
        return layers.stream()
                .filter(LayerItem::isFrozen)
                .map(LayerItem::getName)
                .collect(Collectors.toList());
    }

    /**
     * Gets all layers and their freeze state.
     */
    public List<LayerItem> getLayers() {
        return new ArrayList<>(layers);
    }

    private void applyPreset() {
        if (client == null || layers.isEmpty()) return;

        String selection = presetCombo.getValue();
        String datasetSize;

        if (selection.contains("Small")) {
            datasetSize = "small";
        } else if (selection.contains("Medium")) {
            datasetSize = "medium";
        } else if (selection.contains("Large")) {
            datasetSize = "large";
        } else {
            return; // Custom - don't change
        }

        try {
            Map<Integer, Boolean> recommendations = client.getFreezeRecommendations(
                    datasetSize, currentEncoder);

            for (LayerItem layer : layers) {
                Boolean freeze = recommendations.get(layer.getDepth());
                if (freeze != null) {
                    layer.setFrozen(freeze);
                }
            }

            layerListView.refresh();
            updateStatus();
            logger.info("Applied {} preset for encoder {}: {} layers frozen",
                    datasetSize, currentEncoder, getFrozenLayerNames().size());

        } catch (Exception e) {
            logger.error("Failed to get recommendations", e);
        }
    }

    private void applyRecommended() {
        for (LayerItem layer : layers) {
            layer.setFrozen(layer.isRecommendedFreeze());
        }
        layerListView.refresh();
        updateStatus();
    }

    private void setAllEncoderLayers(boolean frozen) {
        for (LayerItem layer : layers) {
            if (layer.isEncoder()) {
                layer.setFrozen(frozen);
            }
        }
        layerListView.refresh();
        updateStatus();
    }

    private void setAllLayers(boolean frozen) {
        for (LayerItem layer : layers) {
            layer.setFrozen(frozen);
        }
        layerListView.refresh();
        updateStatus();
    }

    private void updateStatus() {
        int frozenCount = 0;
        int totalParams = 0;
        int frozenParams = 0;

        for (LayerItem layer : layers) {
            totalParams += layer.getParamCount();
            if (layer.isFrozen()) {
                frozenCount++;
                frozenParams += layer.getParamCount();
            }
        }

        int trainableParams = totalParams - frozenParams;
        double trainablePercent = totalParams > 0 ? 100.0 * trainableParams / totalParams : 0;

        statusLabel.setText(String.format(
                "%d/%d layers frozen | %,d trainable params (%.1f%%)",
                frozenCount, layers.size(), trainableParams, trainablePercent
        ));
    }

    /**
     * Custom cell for displaying layers with freeze checkbox.
     */
    private class LayerCell extends ListCell<LayerItem> {
        private final HBox container;
        private final CheckBox freezeCheck;
        private final Label nameLabel;
        private final Label paramsLabel;
        private final Label descLabel;
        private final Rectangle depthIndicator;

        public LayerCell() {
            container = new HBox(10);
            container.setAlignment(Pos.CENTER_LEFT);
            container.setPadding(new Insets(5));

            freezeCheck = new CheckBox();
            freezeCheck.setOnAction(e -> {
                LayerItem item = getItem();
                if (item != null) {
                    item.setFrozen(freezeCheck.isSelected());
                    updateStatus();
                }
            });

            depthIndicator = new Rectangle(8, 30);

            VBox textBox = new VBox(2);
            nameLabel = new Label();
            nameLabel.setStyle("-fx-font-weight: bold;");

            HBox detailBox = new HBox(10);
            paramsLabel = new Label();
            paramsLabel.setStyle("-fx-text-fill: #666666; -fx-font-size: 11px;");
            descLabel = new Label();
            descLabel.setStyle("-fx-text-fill: #888888; -fx-font-size: 11px;");
            descLabel.setMaxWidth(300);
            detailBox.getChildren().addAll(paramsLabel, descLabel);

            textBox.getChildren().addAll(nameLabel, detailBox);
            HBox.setHgrow(textBox, Priority.ALWAYS);

            container.getChildren().addAll(freezeCheck, depthIndicator, textBox);
        }

        @Override
        protected void updateItem(LayerItem item, boolean empty) {
            super.updateItem(item, empty);

            if (empty || item == null) {
                setGraphic(null);
            } else {
                freezeCheck.setSelected(item.isFrozen());
                nameLabel.setText(item.getDisplayName());
                paramsLabel.setText(formatParams(item.getParamCount()));
                descLabel.setText(item.getDescription());

                // Color based on depth (green=early/freeze, red=late/train)
                double hue = 120 - (item.getDepth() * 20); // Green to red
                hue = Math.max(0, Math.min(120, hue));
                depthIndicator.setFill(Color.hsb(hue, 0.6, 0.8));

                // Visual feedback for frozen state
                if (item.isFrozen()) {
                    container.setStyle("-fx-background-color: #f0f8ff;");
                } else {
                    container.setStyle("-fx-background-color: #fff8f0;");
                }

                setGraphic(container);
            }
        }

        private String formatParams(int params) {
            if (params >= 1_000_000) {
                return String.format("%.1fM params", params / 1_000_000.0);
            } else if (params >= 1_000) {
                return String.format("%.1fK params", params / 1_000.0);
            } else {
                return params + " params";
            }
        }
    }

    /**
     * Data class for a layer item.
     */
    public static class LayerItem {
        private final String name;
        private final String displayName;
        private final int paramCount;
        private final boolean isEncoder;
        private final int depth;
        private final boolean recommendedFreeze;
        private final String description;
        private final BooleanProperty frozen = new SimpleBooleanProperty(false);

        public LayerItem(String name, String displayName, int paramCount,
                         boolean isEncoder, int depth, boolean recommendedFreeze,
                         String description) {
            this.name = name;
            this.displayName = displayName;
            this.paramCount = paramCount;
            this.isEncoder = isEncoder;
            this.depth = depth;
            this.recommendedFreeze = recommendedFreeze;
            this.description = description;
        }

        public String getName() { return name; }
        public String getDisplayName() { return displayName; }
        public int getParamCount() { return paramCount; }
        public boolean isEncoder() { return isEncoder; }
        public int getDepth() { return depth; }
        public boolean isRecommendedFreeze() { return recommendedFreeze; }
        public String getDescription() { return description; }

        public boolean isFrozen() { return frozen.get(); }
        public void setFrozen(boolean value) { frozen.set(value); }
        public BooleanProperty frozenProperty() { return frozen; }
    }
}
