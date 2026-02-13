package qupath.ext.dlclassifier.service;

import javafx.application.Platform;
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
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class OverlayService {

    private static final Logger logger = LoggerFactory.getLogger(OverlayService.class);
    private static OverlayService instance;

    private PixelClassificationOverlay currentOverlay;

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
        logger.info("Applied DL pixel classifier overlay");
    }

    /**
     * Removes the current DL classification overlay from all viewers.
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
            logger.info("Removed DL pixel classifier overlay");
        }
    }

    /**
     * Checks if an overlay is currently active.
     *
     * @return true if an overlay is applied
     */
    public boolean hasOverlay() {
        return currentOverlay != null;
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
