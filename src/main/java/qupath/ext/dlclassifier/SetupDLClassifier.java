package qupath.ext.dlclassifier;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.scene.control.CheckMenuItem;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;
import javafx.scene.control.SeparatorMenuItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.controller.DLClassifierController;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.ClassifierClient;
import qupath.ext.dlclassifier.service.DLPixelClassifier;
import qupath.ext.dlclassifier.service.ModelManager;
import qupath.ext.dlclassifier.service.OverlayService;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.ui.TooltipHelper;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.images.ImageData;

import java.awt.image.BufferedImage;
import java.util.List;

import java.util.ResourceBundle;

/**
 * Entry point for the Deep Learning Pixel Classifier extension.
 * <p>
 * This extension provides deep learning-based pixel classification capabilities for QuPath,
 * supporting both brightfield RGB and multi-channel fluorescence/spectral images.
 * <p>
 * Key features:
 * <ul>
 *   <li>Train custom pixel classifiers using sparse annotations</li>
 *   <li>Support for multi-channel images with per-channel normalization</li>
 *   <li>Pluggable model architecture system (UNet, SegFormer, etc.)</li>
 *   <li>REST API communication with Python deep learning server</li>
 *   <li>Output as measurements, objects, or classification overlays</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class SetupDLClassifier implements QuPathExtension, GitHubProject {

    private static final Logger logger = LoggerFactory.getLogger(SetupDLClassifier.class);

    // Load extension metadata
    private static final ResourceBundle res = ResourceBundle.getBundle("qupath.ext.dlclassifier.ui.strings");
    private static final String EXTENSION_NAME = res.getString("name");
    private static final String EXTENSION_DESCRIPTION = res.getString("description");
    private static final Version EXTENSION_QUPATH_VERSION = Version.parse("v0.6.0");
    private static final GitHubRepo EXTENSION_REPOSITORY =
            GitHubRepo.create(EXTENSION_NAME, "uw-loci", "qupath-extension-DL-pixel-classifier");

    /** True if the server connection passed validation. */
    private boolean serverAvailable;

    @Override
    public String getName() {
        return EXTENSION_NAME;
    }

    @Override
    public String getDescription() {
        return EXTENSION_DESCRIPTION;
    }

    @Override
    public Version getQuPathVersion() {
        return EXTENSION_QUPATH_VERSION;
    }

    @Override
    public GitHubRepo getRepository() {
        return EXTENSION_REPOSITORY;
    }

    @Override
    public void installExtension(QuPathGUI qupath) {
        logger.info("Installing extension: {}", EXTENSION_NAME);

        // Register persistent preferences
        DLClassifierPreferences.installPreferences(qupath);

        // Check server availability (non-blocking)
        checkServerAvailability();

        // Build menu on the FX thread
        Platform.runLater(() -> addMenuItem(qupath));
    }

    /**
     * Checks if the classification server is available.
     * This is done asynchronously to avoid blocking extension load.
     */
    private void checkServerAvailability() {
        // Initial state - will be updated by background check
        serverAvailable = false;

        // Perform async health check
        Thread healthThread = new Thread(() -> {
            try {
                serverAvailable = DLClassifierChecks.checkServerHealth();
                if (!serverAvailable) {
                    Platform.runLater(() ->
                            Dialogs.showWarningNotification(
                                    EXTENSION_NAME,
                                    "Classification server not available.\n" +
                                            "Start the Python server to enable classification features."
                            )
                    );
                }
            } catch (Exception e) {
                logger.debug("Server health check failed: {}", e.getMessage());
                serverAvailable = false;
            }
        }, "DLClassifier-HealthCheck");
        healthThread.setDaemon(true);
        healthThread.start();
    }

    private void addMenuItem(QuPathGUI qupath) {
        // Create the top level Extensions > DL Pixel Classifier menu
        var extensionMenu = qupath.getMenu("Extensions>" + EXTENSION_NAME, true);

        // === MAIN WORKFLOW MENU ITEMS ===

        // 1) Train Classifier - create a new classifier from annotations
        MenuItem trainOption = new MenuItem(res.getString("menu.training"));
        TooltipHelper.installOnMenuItem(trainOption,
                "Train a new deep learning pixel classifier from annotated regions.\n" +
                        "Requires at least 2 annotation classes (e.g. Foreground/Background).\n" +
                        "Supports single-image and multi-image training from project images.");
        trainOption.disableProperty().bind(
                Bindings.createBooleanBinding(
                        () -> qupath.getImageData() == null,
                        qupath.imageDataProperty()
                )
        );
        trainOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("training"));

        // 2) Apply Classifier - run inference on current image
        MenuItem inferenceOption = new MenuItem(res.getString("menu.inference"));
        TooltipHelper.installOnMenuItem(inferenceOption,
                "Apply a trained classifier to the current image or selected annotations.\n" +
                        "Results can be added as measurements, detection/annotation objects,\n" +
                        "or live classification overlays.");
        inferenceOption.disableProperty().bind(
                Bindings.createBooleanBinding(
                        () -> qupath.getImageData() == null,
                        qupath.imageDataProperty()
                )
        );
        inferenceOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("inference"));

        // 3) Live DL Prediction - toggle live tile classification on/off
        //    When checked and no overlay exists, prompts user to select a classifier
        CheckMenuItem livePredictionOption = new CheckMenuItem(res.getString("menu.toggleOverlay"));
        TooltipHelper.installOnMenuItem(livePredictionOption,
                "Toggle live DL classification overlay on the current viewer.\n" +
                        "If no overlay exists, you will be prompted to select a classifier.\n" +
                        "When unchecked, the overlay is removed and GPU memory is freed.");
        OverlayService overlayService = OverlayService.getInstance();
        // Sync CheckMenuItem state from the property (for programmatic changes)
        overlayService.livePredictionProperty().addListener((obs, wasLive, isLive) ->
                livePredictionOption.setSelected(isLive));
        // Trigger overlay creation or removal when user clicks the CheckMenuItem
        livePredictionOption.setOnAction(e -> {
            if (livePredictionOption.isSelected()) {
                if (overlayService.hasOverlay()) {
                    // Overlay exists - just re-enable live prediction
                    overlayService.setLivePrediction(true);
                } else {
                    // No overlay - create one by prompting for classifier selection
                    createOverlayFromClassifierSelection(qupath, overlayService, livePredictionOption);
                }
            } else {
                // Unchecked - remove the overlay entirely
                overlayService.removeOverlay();
            }
        });
        livePredictionOption.disableProperty().bind(
                Bindings.createBooleanBinding(
                        () -> qupath.getImageData() == null,
                        qupath.imageDataProperty()
                )
        );

        // 4) Remove Overlay - fully remove and clean up resources
        MenuItem removeOverlayOption = new MenuItem(res.getString("menu.removeOverlay"));
        TooltipHelper.installOnMenuItem(removeOverlayOption,
                "Permanently remove the DL classification overlay and free GPU/CPU resources.\n" +
                        "Use this to reclaim memory after you are done viewing the overlay.");
        removeOverlayOption.setOnAction(e -> {
            overlayService.removeOverlay();
            Dialogs.showInfoNotification(EXTENSION_NAME, "Classification overlay removed.");
        });

        // 5) Manage Models - browse and manage saved classifiers
        MenuItem modelsOption = new MenuItem(res.getString("menu.manageModels"));
        TooltipHelper.installOnMenuItem(modelsOption,
                "Browse, import, export, and delete saved classifiers.\n" +
                        "View model metadata, training configuration, and class mappings.");
        modelsOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("modelManagement"));

        // === UTILITIES SUBMENU ===
        Menu utilitiesMenu = new Menu("Utilities");

        // Server Settings
        MenuItem serverOption = new MenuItem(res.getString("menu.serverSettings"));
        TooltipHelper.installOnMenuItem(serverOption,
                "Configure the connection to the Python classification server.\n" +
                        "Test connectivity, view GPU availability, and check server version.");
        serverOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("serverSettings"));

        // Free GPU Memory
        MenuItem freeGpuOption = new MenuItem("Free GPU Memory");
        TooltipHelper.installOnMenuItem(freeGpuOption,
                "Force-clear all GPU memory held by the classification server.\n" +
                        "Cancels running training jobs, clears cached models, and\n" +
                        "frees GPU VRAM. Use after a crash or failed training.");
        freeGpuOption.setOnAction(e -> {
            freeGpuOption.setDisable(true);
            Thread clearThread = new Thread(() -> {
                try {
                    ClassifierClient client = new ClassifierClient(
                            DLClassifierPreferences.getServerHost(),
                            DLClassifierPreferences.getServerPort());
                    String result = client.clearGPUMemory();
                    Platform.runLater(() -> {
                        freeGpuOption.setDisable(false);
                        if (result != null) {
                            Dialogs.showInfoNotification(EXTENSION_NAME, result);
                        } else {
                            Dialogs.showErrorNotification(EXTENSION_NAME,
                                    "Failed to clear GPU memory. Is the server running?");
                        }
                    });
                } catch (Exception ex) {
                    logger.error("GPU memory clear failed", ex);
                    Platform.runLater(() -> {
                        freeGpuOption.setDisable(false);
                        Dialogs.showErrorNotification(EXTENSION_NAME,
                                "Error clearing GPU memory: " + ex.getMessage());
                    });
                }
            }, "DLClassifier-FreeGPU");
            clearThread.setDaemon(true);
            clearThread.start();
        });

        utilitiesMenu.getItems().addAll(serverOption, freeGpuOption);

        // === BUILD FINAL MENU ===
        extensionMenu.getItems().addAll(
                trainOption,
                inferenceOption,
                new SeparatorMenuItem(),
                livePredictionOption,
                removeOverlayOption,
                new SeparatorMenuItem(),
                modelsOption,
                new SeparatorMenuItem(),
                utilitiesMenu
        );

        logger.info("Menu items added for extension: {}", EXTENSION_NAME);
    }

    /**
     * Prompts the user to select a classifier and creates a live overlay.
     * Called when the user checks "Live DL Prediction" and no overlay exists.
     */
    private void createOverlayFromClassifierSelection(QuPathGUI qupath,
                                                       OverlayService overlayService,
                                                       CheckMenuItem livePredictionOption) {
        ImageData<BufferedImage> imageData = qupath.getImageData();
        if (imageData == null) {
            livePredictionOption.setSelected(false);
            Dialogs.showWarningNotification(EXTENSION_NAME, "No image is open.");
            return;
        }

        // List available classifiers
        ModelManager modelManager = new ModelManager();
        List<ClassifierMetadata> classifiers = modelManager.listClassifiers();
        if (classifiers.isEmpty()) {
            livePredictionOption.setSelected(false);
            Dialogs.showWarningNotification(EXTENSION_NAME,
                    "No classifiers available. Train a classifier first.");
            return;
        }

        // Show a choice dialog
        List<String> names = classifiers.stream()
                .map(c -> c.getName() + " (" + c.getId() + ")")
                .toList();
        String choice = Dialogs.showChoiceDialog("Select Classifier",
                "Choose a classifier for the live overlay:", names, names.get(0));
        if (choice == null) {
            livePredictionOption.setSelected(false);
            return;
        }

        // Find the selected classifier
        int selectedIdx = names.indexOf(choice);
        ClassifierMetadata metadata = classifiers.get(selectedIdx);

        // Build channel config from metadata
        List<String> expectedChannels = metadata.getExpectedChannelNames();
        List<Integer> selectedChannels = new java.util.ArrayList<>();
        for (int i = 0; i < Math.max(expectedChannels.size(), metadata.getInputChannels()); i++) {
            selectedChannels.add(i);
        }
        ChannelConfiguration channelConfig = ChannelConfiguration.builder()
                .selectedChannels(selectedChannels)
                .channelNames(expectedChannels.isEmpty()
                        ? List.of("Red", "Green", "Blue") : expectedChannels)
                .bitDepth(metadata.getBitDepthTrained())
                .normalizationStrategy(metadata.getNormalizationStrategy())
                .build();

        // Build a minimal inference config for overlay mode
        InferenceConfig inferenceConfig = InferenceConfig.builder()
                .tileSize(metadata.getInputWidth())
                .overlap(32)
                .outputType(InferenceConfig.OutputType.OVERLAY)
                .build();

        // Create the pixel classifier and apply overlay
        DLPixelClassifier pixelClassifier = new DLPixelClassifier(
                metadata, channelConfig, inferenceConfig, imageData);
        overlayService.applyClassifierOverlay(imageData, pixelClassifier);
        Dialogs.showInfoNotification(EXTENSION_NAME,
                "Live DL overlay applied: " + metadata.getName());
    }

}
