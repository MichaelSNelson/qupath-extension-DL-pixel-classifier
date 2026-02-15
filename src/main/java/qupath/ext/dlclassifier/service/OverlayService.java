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
 * The overlay can be toggled on/off independently of QuPath's built-in
 * "Show pixel classification" (C key), allowing side-by-side comparison
 * with QuPath's own pixel classifiers. Toggling off stops server requests
 * but preserves cached tiles; toggling back on resumes live prediction.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class OverlayService {

    private static final Logger logger = LoggerFactory.getLogger(OverlayService.class);
    private static OverlayService instance;

    private PixelClassificationOverlay currentOverlay;
    private DLPixelClassifier currentClassifier;
    private ImageData<BufferedImage> currentImageData;

    /** Observable property tracking whether the DL overlay is visible. */
    private final BooleanProperty overlayVisible = new SimpleBooleanProperty(false);

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
        this.currentImageData = imageData;
        overlayVisible.set(true);
        logger.info("Applied DL pixel classifier overlay");
    }

    /**
     * Toggles the DL overlay on or off without destroying it.
     * <p>
     * When toggled off, live prediction is stopped and the overlay is
     * removed from the viewer, but the overlay object and cached tiles
     * are preserved. Toggling back on re-adds the overlay and resumes
     * live prediction.
     * <p>
     * This is independent of QuPath's "C" shortcut, which controls the
     * built-in pixel classification overlay visibility.
     *
     * @param visible true to show the overlay, false to hide it
     */
    public void setOverlayVisible(boolean visible) {
        if (currentOverlay == null) {
            overlayVisible.set(false);
            return;
        }

        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath == null) return;

        if (visible) {
            currentOverlay.setLivePrediction(true);
            for (QuPathViewer viewer : qupath.getAllViewers()) {
                if (viewer.getImageData() == currentImageData) {
                    Platform.runLater(() -> viewer.setCustomPixelLayerOverlay(currentOverlay));
                }
            }
            overlayVisible.set(true);
            logger.info("DL overlay shown");
        } else {
            currentOverlay.setLivePrediction(false);
            for (QuPathViewer viewer : qupath.getAllViewers()) {
                if (viewer.getCustomPixelLayerOverlay() == currentOverlay) {
                    Platform.runLater(viewer::resetCustomPixelLayerOverlay);
                }
            }
            overlayVisible.set(false);
            logger.info("DL overlay hidden (cached tiles preserved)");
        }
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
            currentImageData = null;
            overlayVisible.set(false);

            // Clean up classifier resources (shared temp directory)
            if (currentClassifier != null) {
                currentClassifier.cleanup();
                currentClassifier = null;
            }

            logger.info("Removed DL pixel classifier overlay");
        }
    }

    /**
     * Checks if an overlay exists (may be hidden).
     *
     * @return true if an overlay has been applied (visible or hidden)
     */
    public boolean hasOverlay() {
        return currentOverlay != null;
    }

    /**
     * Observable property for overlay visibility.
     * Bind to this for CheckMenuItem state, etc.
     *
     * @return the overlay visible property
     */
    public BooleanProperty overlayVisibleProperty() {
        return overlayVisible;
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
