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
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageChannel;
import qupath.lib.images.servers.ImageServer;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Reusable panel for selecting and configuring image channels.
 * <p>
 * This component provides:
 * <ul>
 *   <li>Multi-select channel list with color indicators</li>
 *   <li>Channel reordering with up/down buttons</li>
 *   <li>Normalization strategy selection per channel</li>
 *   <li>Real-time validation and feedback</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ChannelSelectionPanel extends VBox {

    private static final Logger logger = LoggerFactory.getLogger(ChannelSelectionPanel.class);

    private final ListView<ChannelItem> availableList;
    private final ListView<ChannelItem> selectedList;
    private final ComboBox<ChannelConfiguration.NormalizationStrategy> normalizationCombo;
    private final Label statusLabel;
    private final BooleanProperty validProperty = new SimpleBooleanProperty(false);

    private ImageServer<BufferedImage> currentServer;
    private int requiredChannelCount = -1; // -1 means any count is valid

    /**
     * Creates a new channel selection panel.
     */
    public ChannelSelectionPanel() {
        setSpacing(10);
        setPadding(new Insets(10));

        // Available channels list
        availableList = new ListView<>();
        availableList.setCellFactory(lv -> new ChannelListCell());
        availableList.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
        availableList.setPrefHeight(150);
        availableList.setTooltip(new Tooltip(
                "Image channels available for selection.\n" +
                "Multi-select with Ctrl+click or Shift+click."));

        // Selected channels list
        selectedList = new ListView<>();
        selectedList.setCellFactory(lv -> new ChannelListCell());
        selectedList.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
        selectedList.setPrefHeight(150);
        selectedList.setTooltip(new Tooltip(
                "Channels that will be used as model input.\n" +
                "Order matters: must match the order used during training."));

        // Transfer buttons - fixed width to prevent text truncation
        double buttonWidth = 40;

        Button addButton = new Button(">");
        addButton.setMinWidth(buttonWidth);
        addButton.setTooltip(new Tooltip("Add selected channels"));
        addButton.setOnAction(e -> addSelectedChannels());

        Button removeButton = new Button("<");
        removeButton.setMinWidth(buttonWidth);
        removeButton.setTooltip(new Tooltip("Remove selected channels"));
        removeButton.setOnAction(e -> removeSelectedChannels());

        Button addAllButton = new Button(">>");
        addAllButton.setMinWidth(buttonWidth);
        addAllButton.setTooltip(new Tooltip("Add all channels"));
        addAllButton.setOnAction(e -> addAllChannels());

        Button removeAllButton = new Button("<<");
        removeAllButton.setMinWidth(buttonWidth);
        removeAllButton.setTooltip(new Tooltip("Remove all channels"));
        removeAllButton.setOnAction(e -> removeAllChannels());

        VBox transferButtons = new VBox(5, addButton, removeButton, new Separator(), addAllButton, removeAllButton);
        transferButtons.setAlignment(Pos.CENTER);

        // Reorder buttons
        Button upButton = new Button("Up");
        upButton.setTooltip(new Tooltip("Move selected channel up"));
        upButton.setOnAction(e -> moveSelectedUp());

        Button downButton = new Button("Down");
        downButton.setTooltip(new Tooltip("Move selected channel down"));
        downButton.setOnAction(e -> moveSelectedDown());

        VBox reorderButtons = new VBox(5, upButton, downButton);
        reorderButtons.setAlignment(Pos.CENTER);

        // Layout for channel lists
        VBox availableBox = new VBox(5, new Label("Available Channels:"), availableList);
        VBox selectedBox = new VBox(5, new Label("Selected Channels:"), selectedList, reorderButtons);
        HBox.setHgrow(availableBox, Priority.ALWAYS);
        HBox.setHgrow(selectedBox, Priority.ALWAYS);

        HBox listsBox = new HBox(10, availableBox, transferButtons, selectedBox);
        listsBox.setAlignment(Pos.CENTER);

        // Normalization settings
        normalizationCombo = new ComboBox<>(FXCollections.observableArrayList(
                ChannelConfiguration.NormalizationStrategy.values()));
        normalizationCombo.setValue(ChannelConfiguration.NormalizationStrategy.PERCENTILE_99);
        normalizationCombo.setTooltip(new Tooltip(
                "Per-channel intensity normalization before inference:\n" +
                "MIN_MAX: Scale to [0,1] using channel min/max values.\n" +
                "PERCENTILE_99: Scale using 1st/99th percentiles (robust to outliers).\n" +
                "Z_SCORE: Subtract mean, divide by std deviation.\n" +
                "FIXED_RANGE: Use a fixed intensity range (e.g. 0-255 for 8-bit)."));

        HBox normBox = new HBox(10,
                new Label("Normalization:"),
                normalizationCombo);
        normBox.setAlignment(Pos.CENTER_LEFT);

        // Status label
        statusLabel = new Label();
        statusLabel.setStyle("-fx-text-fill: #666;");

        // Add all components
        getChildren().addAll(listsBox, normBox, statusLabel);

        // Update validity when selection changes
        selectedList.getItems().addListener((javafx.collections.ListChangeListener<ChannelItem>) c -> updateValidity());
    }

    /**
     * Populates the panel from an image data object.
     *
     * @param imageData the image data to get channels from
     */
    public void setImageData(ImageData<BufferedImage> imageData) {
        availableList.getItems().clear();
        selectedList.getItems().clear();

        if (imageData == null || imageData.getServer() == null) {
            currentServer = null;
            updateValidity();
            return;
        }

        currentServer = imageData.getServer();
        List<ImageChannel> channels = currentServer.getMetadata().getChannels();

        for (int i = 0; i < channels.size(); i++) {
            ImageChannel channel = channels.get(i);
            Color color = javafxColorFromAwtColor(channel.getColor());
            availableList.getItems().add(new ChannelItem(i, channel.getName(), color));
        }

        logger.debug("Loaded {} channels from image", channels.size());
        updateValidity();
    }

    /**
     * Sets the required number of channels for validation.
     *
     * @param count required channel count, or -1 for any
     */
    public void setRequiredChannelCount(int count) {
        this.requiredChannelCount = count;
        updateValidity();
    }

    /**
     * Gets the current channel configuration.
     *
     * @return channel configuration based on current selections
     */
    public ChannelConfiguration getChannelConfiguration() {
        List<Integer> indices = selectedList.getItems().stream()
                .map(ChannelItem::index)
                .collect(Collectors.toList());

        List<String> names = selectedList.getItems().stream()
                .map(ChannelItem::name)
                .collect(Collectors.toList());

        int bitDepth = currentServer != null ?
                currentServer.getPixelType().getBitsPerPixel() : 8;

        return ChannelConfiguration.builder()
                .selectedChannels(indices)
                .channelNames(names)
                .bitDepth(bitDepth)
                .normalizationStrategy(normalizationCombo.getValue())
                .build();
    }

    /**
     * Sets the channel configuration from a classifier's expected channels.
     *
     * @param channelNames the expected channel names
     */
    public void setExpectedChannels(List<String> channelNames) {
        if (channelNames == null || channelNames.isEmpty()) {
            return;
        }

        // Try to match channels by name
        selectedList.getItems().clear();
        for (String name : channelNames) {
            for (ChannelItem item : availableList.getItems()) {
                if (item.name().equalsIgnoreCase(name)) {
                    selectedList.getItems().add(item);
                    break;
                }
            }
        }

        setRequiredChannelCount(channelNames.size());
        updateValidity();
    }

    /**
     * Auto-selects all available channels. Useful for brightfield images
     * where channel selection is not meaningful.
     */
    public void selectAllChannels() {
        addAllChannels();
    }

    /**
     * Gets the number of available channels.
     *
     * @return number of channels in the available list
     */
    public int getAvailableChannelCount() {
        return availableList.getItems().size();
    }

    /**
     * Gets whether the current selection is valid.
     *
     * @return true if valid
     */
    public boolean isValid() {
        return validProperty.get();
    }

    /**
     * Gets the valid property for binding.
     *
     * @return the valid property
     */
    public BooleanProperty validProperty() {
        return validProperty;
    }

    /**
     * Gets the selected channel count.
     *
     * @return number of selected channels
     */
    public int getSelectedChannelCount() {
        return selectedList.getItems().size();
    }

    private void addSelectedChannels() {
        List<ChannelItem> toAdd = new ArrayList<>(availableList.getSelectionModel().getSelectedItems());
        for (ChannelItem item : toAdd) {
            if (!selectedList.getItems().contains(item)) {
                selectedList.getItems().add(item);
            }
        }
    }

    private void removeSelectedChannels() {
        List<ChannelItem> toRemove = new ArrayList<>(selectedList.getSelectionModel().getSelectedItems());
        selectedList.getItems().removeAll(toRemove);
    }

    private void addAllChannels() {
        for (ChannelItem item : availableList.getItems()) {
            if (!selectedList.getItems().contains(item)) {
                selectedList.getItems().add(item);
            }
        }
    }

    private void removeAllChannels() {
        selectedList.getItems().clear();
    }

    private void moveSelectedUp() {
        int index = selectedList.getSelectionModel().getSelectedIndex();
        if (index > 0) {
            ChannelItem item = selectedList.getItems().remove(index);
            selectedList.getItems().add(index - 1, item);
            selectedList.getSelectionModel().select(index - 1);
        }
    }

    private void moveSelectedDown() {
        int index = selectedList.getSelectionModel().getSelectedIndex();
        if (index >= 0 && index < selectedList.getItems().size() - 1) {
            ChannelItem item = selectedList.getItems().remove(index);
            selectedList.getItems().add(index + 1, item);
            selectedList.getSelectionModel().select(index + 1);
        }
    }

    private void updateValidity() {
        int count = selectedList.getItems().size();
        boolean valid;

        if (count == 0) {
            statusLabel.setText("Please select at least one channel");
            statusLabel.setStyle("-fx-text-fill: #cc0000;");
            valid = false;
        } else if (requiredChannelCount > 0 && count != requiredChannelCount) {
            statusLabel.setText(String.format("Selected %d channels (classifier expects %d)", count, requiredChannelCount));
            statusLabel.setStyle("-fx-text-fill: #cc0000;");
            valid = false;
        } else {
            statusLabel.setText(String.format("%d channel(s) selected", count));
            statusLabel.setStyle("-fx-text-fill: #006600;");
            valid = true;
        }

        validProperty.set(valid);
    }

    private Color javafxColorFromAwtColor(Integer argb) {
        if (argb == null) {
            return Color.GRAY;
        }
        int r = (argb >> 16) & 0xFF;
        int g = (argb >> 8) & 0xFF;
        int b = argb & 0xFF;
        return Color.rgb(r, g, b);
    }

    /**
     * Represents a channel item in the list.
     */
    public record ChannelItem(int index, String name, Color color) {
        @Override
        public String toString() {
            return String.format("[%d] %s", index, name);
        }
    }

    /**
     * Custom cell renderer for channel items.
     */
    private static class ChannelListCell extends ListCell<ChannelItem> {
        private final HBox content;
        private final Rectangle colorBox;
        private final Label nameLabel;

        public ChannelListCell() {
            colorBox = new Rectangle(16, 16);
            colorBox.setStroke(Color.BLACK);
            colorBox.setStrokeWidth(1);

            nameLabel = new Label();

            content = new HBox(8, colorBox, nameLabel);
            content.setAlignment(Pos.CENTER_LEFT);
        }

        @Override
        protected void updateItem(ChannelItem item, boolean empty) {
            super.updateItem(item, empty);

            if (empty || item == null) {
                setGraphic(null);
            } else {
                colorBox.setFill(item.color());
                nameLabel.setText(item.toString());
                setGraphic(content);
            }
        }
    }
}
