package qupath.ext.dlclassifier.service;

import javafx.scene.image.Image;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.viewer.QuPathViewer;

import java.awt.image.BufferedImage;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Service for managing classification overlays in QuPath viewers.
 * <p>
 * This service caches overlay images and provides methods for updating
 * and displaying classification results as semi-transparent overlays
 * on the viewer.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class OverlayService {

    private static final Logger logger = LoggerFactory.getLogger(OverlayService.class);
    private static OverlayService instance;

    private final Map<String, Image> overlayCache;
    private double overlayOpacity = 0.5;
    private boolean overlaysEnabled = true;

    private OverlayService() {
        this.overlayCache = new ConcurrentHashMap<>();
    }

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
     * Sets an overlay for a region.
     *
     * @param regionId  region identifier
     * @param overlay   overlay image
     * @param metadata  classifier metadata for coloring
     */
    public void setOverlay(String regionId, BufferedImage overlay, ClassifierMetadata metadata) {
        // Convert to JavaFX Image
        Image fxImage = convertToFXImage(overlay);
        overlayCache.put(regionId, fxImage);

        logger.debug("Set overlay for region: {}", regionId);
    }

    /**
     * Removes an overlay.
     *
     * @param regionId region identifier
     */
    public void removeOverlay(String regionId) {
        overlayCache.remove(regionId);
        logger.debug("Removed overlay for region: {}", regionId);
    }

    /**
     * Clears all overlays.
     */
    public void clearOverlays() {
        overlayCache.clear();
        logger.info("Cleared all overlays");
    }

    /**
     * Sets the overlay opacity.
     *
     * @param opacity opacity value (0.0 to 1.0)
     */
    public void setOverlayOpacity(double opacity) {
        this.overlayOpacity = Math.max(0, Math.min(1, opacity));
        refreshViewers();
    }

    /**
     * Gets the overlay opacity.
     *
     * @return current opacity
     */
    public double getOverlayOpacity() {
        return overlayOpacity;
    }

    /**
     * Enables or disables overlay display.
     *
     * @param enabled true to enable overlays
     */
    public void setOverlaysEnabled(boolean enabled) {
        this.overlaysEnabled = enabled;
        refreshViewers();
    }

    /**
     * Checks if overlays are enabled.
     *
     * @return true if enabled
     */
    public boolean isOverlaysEnabled() {
        return overlaysEnabled;
    }

    /**
     * Converts a BufferedImage to a JavaFX Image.
     */
    private Image convertToFXImage(BufferedImage bImage) {
        WritableImage fxImage = new WritableImage(bImage.getWidth(), bImage.getHeight());
        PixelWriter pw = fxImage.getPixelWriter();

        for (int y = 0; y < bImage.getHeight(); y++) {
            for (int x = 0; x < bImage.getWidth(); x++) {
                pw.setArgb(x, y, bImage.getRGB(x, y));
            }
        }

        return fxImage;
    }

    /**
     * Refreshes all viewers to update overlay display.
     */
    private void refreshViewers() {
        QuPathGUI qupath = QuPathGUI.getInstance();
        if (qupath != null) {
            for (QuPathViewer viewer : qupath.getAllViewers()) {
                viewer.repaint();
            }
        }
    }

    /**
     * Gets the number of cached overlays.
     *
     * @return cache size
     */
    public int getCacheSize() {
        return overlayCache.size();
    }
}
