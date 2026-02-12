package qupath.ext.dlclassifier.classifier.handlers;

import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.layout.GridPane;
import qupath.ext.dlclassifier.classifier.ClassifierHandler;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Handler for UNet architecture pixel classifiers.
 * <p>
 * UNet is a fully convolutional encoder-decoder network originally designed for
 * biomedical image segmentation. It uses skip connections between encoder and
 * decoder to preserve spatial information.
 *
 * <h3>Supported Backbones</h3>
 * <ul>
 *   <li>resnet18, resnet34, resnet50 (recommended)</li>
 *   <li>efficientnet-b0 through efficientnet-b4</li>
 *   <li>mobilenet_v2 (lightweight)</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class UNetHandler implements ClassifierHandler {

    /** Available backbone architectures for UNet encoder */
    public static final List<String> BACKBONES = List.of(
            "resnet18",
            "resnet34",
            "resnet50",
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "mobilenet_v2"
    );

    /** Supported tile sizes (must be divisible by 32 for UNet) */
    public static final List<Integer> TILE_SIZES = List.of(
            128, 256, 384, 512, 768, 1024
    );

    @Override
    public String getType() {
        return "unet";
    }

    @Override
    public String getDisplayName() {
        return "UNet";
    }

    @Override
    public String getDescription() {
        return "Encoder-decoder architecture with skip connections. " +
                "Excellent for semantic segmentation with strong boundary preservation. " +
                "Supports various pretrained backbones (ResNet, EfficientNet).";
    }

    @Override
    public TrainingConfig getDefaultTrainingConfig() {
        return TrainingConfig.builder()
                .modelType("unet")
                .backbone("resnet34")
                .epochs(50)
                .batchSize(8)
                .learningRate(0.001)
                .weightDecay(1e-4)
                .tileSize(512)
                .overlap(64)
                .validationSplit(0.2)
                .augmentation(true)
                .usePretrainedWeights(true)
                .freezeEncoderLayers(0)
                .build();
    }

    @Override
    public InferenceConfig getDefaultInferenceConfig() {
        return InferenceConfig.builder()
                .tileSize(512)
                .overlap(64)
                .blendMode(InferenceConfig.BlendMode.LINEAR)
                .outputType(InferenceConfig.OutputType.MEASUREMENTS)
                .minObjectSizeMicrons(10.0)
                .holeFillingMicrons(5.0)
                .boundarySmoothing(2.0)
                .maxTilesInMemory(50)
                .useGPU(true)
                .build();
    }

    @Override
    public boolean supportsVariableChannels() {
        return true;
    }

    @Override
    public int getMinChannels() {
        return 1;
    }

    @Override
    public int getMaxChannels() {
        return 64; // Practical upper limit
    }

    @Override
    public List<Integer> getSupportedTileSizes() {
        return TILE_SIZES;
    }

    @Override
    public Optional<String> validateChannelConfig(ChannelConfiguration channelConfig) {
        if (channelConfig == null) {
            return Optional.of("Channel configuration is required");
        }

        int numChannels = channelConfig.getNumChannels();
        if (numChannels < getMinChannels()) {
            return Optional.of(String.format(
                    "UNet requires at least %d channel(s), but %d selected",
                    getMinChannels(), numChannels));
        }
        if (numChannels > getMaxChannels()) {
            return Optional.of(String.format(
                    "UNet supports at most %d channels, but %d selected",
                    getMaxChannels(), numChannels));
        }

        return Optional.empty();
    }

    @Override
    public Map<String, Object> getArchitectureParams(TrainingConfig config) {
        Map<String, Object> params = new HashMap<>();
        params.put("architecture", "unet");
        params.put("available_backbones", BACKBONES);
        params.put("encoder_depth", 5);
        params.put("decoder_channels", List.of(256, 128, 64, 32, 16));

        if (config != null) {
            params.put("backbone", config.getBackbone());
            params.put("use_pretrained", config.isUsePretrainedWeights());
            params.put("freeze_encoder_layers", config.getFreezeEncoderLayers());
        } else {
            params.put("backbone", "resnet34");
            params.put("use_pretrained", true);
            params.put("freeze_encoder_layers", 0);
        }
        return params;
    }

    @Override
    public Optional<TrainingUI> createTrainingUI() {
        return Optional.of(new UNetTrainingUI());
    }

    @Override
    public ClassifierMetadata buildMetadata(TrainingConfig config,
                                            ChannelConfiguration channelConfig,
                                            List<String> classNames) {
        // Generate a unique ID
        String timestamp = LocalDateTime.now().format(
                DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String id = String.format("unet_%s_%s", config.getBackbone(), timestamp);

        ClassifierMetadata.Builder builder = ClassifierMetadata.builder()
                .id(id)
                .name(String.format("UNet %s Classifier", config.getBackbone()))
                .description(String.format("UNet with %s backbone, %d channels, %d classes",
                        config.getBackbone(),
                        channelConfig.getNumChannels(),
                        classNames.size()))
                .modelType("unet")
                .backbone(config.getBackbone())
                .inputSize(config.getTileSize(), config.getTileSize())
                .inputChannels(channelConfig.getNumChannels())
                .expectedChannelNames(channelConfig.getChannelNames())
                .normalizationStrategy(channelConfig.getNormalizationStrategy())
                .bitDepthTrained(channelConfig.getBitDepth())
                .trainingEpochs(config.getEpochs());

        // Add classes
        for (int i = 0; i < classNames.size(); i++) {
            String color = getDefaultColor(i);
            builder.addClass(i, classNames.get(i), color);
        }

        return builder.build();
    }

    /**
     * Returns a default color for a class index.
     */
    private String getDefaultColor(int index) {
        String[] colors = {
                "#808080", // Gray (background)
                "#FF0000", // Red
                "#00FF00", // Green
                "#0000FF", // Blue
                "#FFFF00", // Yellow
                "#FF00FF", // Magenta
                "#00FFFF", // Cyan
                "#FFA500"  // Orange
        };
        return colors[index % colors.length];
    }

    /**
     * UI component for UNet-specific training parameters.
     */
    private static class UNetTrainingUI implements TrainingUI {

        private final GridPane root;
        private final ComboBox<String> backboneCombo;
        private final Spinner<Integer> freezeLayersSpinner;

        public UNetTrainingUI() {
            root = new GridPane();
            root.setHgap(10);
            root.setVgap(10);
            root.setPadding(new Insets(10));

            // Backbone selection
            Label backboneLabel = new Label("Backbone:");
            backboneCombo = new ComboBox<>(FXCollections.observableArrayList(BACKBONES));
            backboneCombo.setValue("resnet34");
            backboneCombo.setMaxWidth(Double.MAX_VALUE);

            // Freeze layers
            Label freezeLabel = new Label("Freeze Encoder Layers:");
            freezeLayersSpinner = new Spinner<>();
            freezeLayersSpinner.setValueFactory(
                    new SpinnerValueFactory.IntegerSpinnerValueFactory(0, 5, 0));
            freezeLayersSpinner.setEditable(true);
            freezeLayersSpinner.setMaxWidth(100);

            root.add(backboneLabel, 0, 0);
            root.add(backboneCombo, 1, 0);
            root.add(freezeLabel, 0, 1);
            root.add(freezeLayersSpinner, 1, 1);
        }

        @Override
        public Node getNode() {
            return root;
        }

        @Override
        public Map<String, Object> getParameters() {
            Map<String, Object> params = new HashMap<>();
            params.put("backbone", backboneCombo.getValue());
            params.put("freeze_encoder_layers", freezeLayersSpinner.getValue());
            return params;
        }

        @Override
        public Optional<String> validate() {
            if (backboneCombo.getValue() == null) {
                return Optional.of("Please select a backbone architecture");
            }
            return Optional.empty();
        }
    }
}
