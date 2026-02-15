package qupath.ext.dlclassifier.service;

import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.gui.viewer.overlays.PixelClassificationOverlay;
import qupath.lib.images.ImageData;

import java.awt.image.BufferedImage;

/**
 * Service for managing DL pixel classification overlays in QuPath viewers.
 * <p>
 * This service wraps a {@link DLPixelClassifier} in QuPath's native
 * {@link PixelClassificationOverlay} system, which handles on-demand tile
 * rendering, caching, and display as the user pans and zooms.
 * <p>
 * Live prediction can be toggled on/off without destroying the overlay.
 * When off, cached tiles remain visible but no new server requests are
 * made. This matches QuPath's own "Live prediction" toggle behavior.
 * <p>
 * Note: both this overlay and QuPath's built-in pixel classifier share
 * the same {@code customPixelLayerOverlay} viewer slot, so only one can
 * be active at a time.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class OverlayService {

    private static final Logger logger = LoggerFactory.getLogger(OverlayService.class);
    private static OverlayService instance;

    private PixelClassificationOverlay currentOverlay;
    private DLPixelClassifier currentClassifier;

    /** Observable property tracking whether live prediction is active. */
    private final BooleanProperty livePrediction = new SimpleBooleanProperty(false);

    private OverlayService() {}

    /**
     * Gets the singleton instance.
     *
     * @return the overlay service instance
     */
    public static synchronized OverlayService getInstance() {
        if (instance == null) {
            instance = new OverlayService();
        }
        return instance;
    }

    /**
     * Applies a DL pixel classifier as a native QuPath overlay.
     * <p>
     * This creates a {@link PixelClassificationOverlay} from the classifier
     * and sets it on all viewers displaying the given image. Tiles are
     * classified on demand as the user navigates.
     *
     * @param imageData  the image data to overlay
     * @param classifier the DL pixel classifier
     */
    public void applyClassifierOverlay(ImageData<BufferedImage> imageData,
                                        DLPixelClassifier classifier) {
        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null) {
            logger.warn("QuPath GUI not available - cannot apply overlay");
            return;
        }

        // Remove any existing DL overlay first
        removeOverlay();

        // Create the overlay using QuPath's native system
        var overlay = PixelClassificationOverlay.create(
                qupath.getOverlayOptions(),
                classifier,
                Runtime.getRuntime().availableProcessors());

        // Enable live prediction so tiles are classified as the user navigates
        overlay.setLivePrediction(true);

        // Apply to all viewers showing this image
        for (QuPathViewer viewer : qupath.getAllViewers()) {
            if (viewer.getImageData() == imageData) {
                Platform.runLater(() -> viewer.setCustomPixelLayerOverlay(overlay));
            }
        }

        this.currentOverlay = overlay;
        this.currentClassifier = classifier;
        livePrediction.set(true);
        logger.info("Applied DL pixel classifier overlay");
    }

    /**
     * Toggles live prediction on or off.
     * <p>
     * When off, the overlay remains in the viewer and cached tiles stay
     * visible, but no new tiles are requested from the server. When on,
     * new tiles are classified on demand as the user pans and zooms.
     * <p>
     * This matches QuPath's built-in "Live prediction" toggle behavior.
     *
     * @param live true to enable live prediction, false to pause
     */
    public void setLivePrediction(boolean live) {
        if (currentOverlay == null) {
            livePrediction.set(false);
            return;
        }

        currentOverlay.setLivePrediction(live);
        livePrediction.set(live);
        logger.info("DL overlay live prediction: {}", live ? "on" : "off");
    }

    /**
     * Removes the current DL classification overlay from all viewers
     * and cleans up all resources.
     */
    public void removeOverlay() {
        if (currentOverlay != null) {
            currentOverlay.stop();

            QuPathGUI qupath = QuPathGUI.getInstance();
            if (qupath != null) {
                for (QuPathViewer viewer : qupath.getAllViewers()) {
                    if (viewer.getCustomPixelLayerOverlay() == currentOverlay) {
                        Platform.runLater(viewer::resetCustomPixelLayerOverlay);
                    }
                }
            }

            currentOverlay = null;
            livePrediction.set(false);

            // Clean up classifier resources (shared temp directory)
            if (currentClassifier != null) {
                currentClassifier.cleanup();
                currentClassifier = null;
            }

            logger.info("Removed DL pixel classifier overlay");
        }
    }

    /**
     * Checks if an overlay exists (may have live prediction paused).
     *
     * @return true if an overlay has been applied
     */
    public boolean hasOverlay() {
        return currentOverlay != null;
    }

    /**
     * Observable property for live prediction state.
     * Bind to this for CheckMenuItem state, etc.
     *
     * @return the live prediction property
     */
    public BooleanProperty livePredictionProperty() {
        return livePrediction;
    }

    /**
     * Refreshes all viewers to update overlay display.
     */
    public void refreshViewers() {
        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath != null) {
            for (QuPathViewer viewer : qupath.getAllViewers()) {
                viewer.repaint();
            }
        }
    }
}
