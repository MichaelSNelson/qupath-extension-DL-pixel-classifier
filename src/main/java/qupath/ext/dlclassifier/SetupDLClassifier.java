package qupath.ext.dlclassifier;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.scene.control.CheckMenuItem;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;
import javafx.scene.control.SeparatorMenuItem;
import javafx.scene.control.Tooltip;
import javafx.util.Duration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.controller.DLClassifierController;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.OverlayService;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;

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
        setMenuItemTooltip(trainOption,
                "Train a new deep learning pixel classifier from annotated regions. " +
                        "Requires foreground and background class annotations.");
        trainOption.disableProperty().bind(
                Bindings.createBooleanBinding(
                        () -> qupath.getImageData() == null,
                        qupath.imageDataProperty()
                )
        );
        trainOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("training"));

        // 2) Apply Classifier - run inference on current image
        MenuItem inferenceOption = new MenuItem(res.getString("menu.inference"));
        setMenuItemTooltip(inferenceOption,
                "Apply a trained classifier to the current image or selected annotations. " +
                        "Results can be added as measurements, objects, or overlays.");
        inferenceOption.disableProperty().bind(
                Bindings.createBooleanBinding(
                        () -> qupath.getImageData() == null,
                        qupath.imageDataProperty()
                )
        );
        inferenceOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("inference"));

        // 3) Live DL Prediction - toggle live tile classification on/off
        CheckMenuItem livePredictionOption = new CheckMenuItem(res.getString("menu.toggleOverlay"));
        setMenuItemTooltip(livePredictionOption,
                "Toggle live DL classification. When off, cached tiles remain " +
                        "visible but no new server requests are made. " +
                        "Works like QuPath's own 'Live prediction' toggle.");
        OverlayService overlayService = OverlayService.getInstance();
        // Sync CheckMenuItem state from the property (for programmatic changes)
        overlayService.livePredictionProperty().addListener((obs, wasLive, isLive) ->
                livePredictionOption.setSelected(isLive));
        // Trigger live prediction toggle when user clicks the CheckMenuItem
        livePredictionOption.setOnAction(e ->
                overlayService.setLivePrediction(livePredictionOption.isSelected()));
        livePredictionOption.disableProperty().bind(
                Bindings.createBooleanBinding(
                        () -> !overlayService.hasOverlay(),
                        overlayService.livePredictionProperty()
                )
        );

        // 4) Remove Overlay - fully remove and clean up resources
        MenuItem removeOverlayOption = new MenuItem(res.getString("menu.removeOverlay"));
        setMenuItemTooltip(removeOverlayOption,
                "Permanently remove the DL classification overlay and free resources.");
        removeOverlayOption.setOnAction(e -> {
            overlayService.removeOverlay();
            Dialogs.showInfoNotification(EXTENSION_NAME, "Classification overlay removed.");
        });

        // 5) Manage Models - browse and manage saved classifiers
        MenuItem modelsOption = new MenuItem(res.getString("menu.manageModels"));
        setMenuItemTooltip(modelsOption,
                "Browse, import, export, and delete saved classifiers. " +
                        "View model metadata and training configuration.");
        modelsOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("modelManagement"));

        // === UTILITIES SUBMENU ===
        Menu utilitiesMenu = new Menu("Utilities");

        // Server Settings
        MenuItem serverOption = new MenuItem(res.getString("menu.serverSettings"));
        setMenuItemTooltip(serverOption,
                "Configure the connection to the Python classification server. " +
                        "Test connectivity and view GPU availability.");
        serverOption.setOnAction(e -> DLClassifierController.getInstance().startWorkflow("serverSettings"));

        utilitiesMenu.getItems().addAll(serverOption);

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
     * Sets a tooltip on a MenuItem using the JavaFX tooltip mechanism.
     *
     * @param menuItem    the menu item to add tooltip to
     * @param tooltipText the tooltip text to display
     */
    private void setMenuItemTooltip(MenuItem menuItem, String tooltipText) {
        Tooltip tooltip = new Tooltip(tooltipText);
        tooltip.setShowDelay(Duration.millis(500));
        tooltip.setShowDuration(Duration.seconds(30));
        tooltip.setWrapText(true);
        tooltip.setMaxWidth(350);

        menuItem.parentPopupProperty().addListener((obs, oldPopup, newPopup) -> {
            if (newPopup != null) {
                newPopup.setOnShown(e -> {
                    var node = menuItem.getStyleableNode();
                    if (node != null) {
                        Tooltip.install(node, tooltip);
                    }
                });
            }
        });
    }
}
