package qupath.ext.dlclassifier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.BackendFactory;
import qupath.ext.dlclassifier.service.ClassifierBackend;

/**
 * Validation utilities for the DL Pixel Classifier extension.
 * <p>
 * This class provides methods to check system requirements, server availability,
 * and configuration validity before starting classification workflows.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public final class DLClassifierChecks {

    private static final Logger logger = LoggerFactory.getLogger(DLClassifierChecks.class);

    private DLClassifierChecks() {
        // Utility class - no instantiation
    }

    /**
     * Checks if the classification server is healthy and responding.
     *
     * @return true if server is available, false otherwise
     */
    public static boolean checkServerHealth() {
        try {
            ClassifierBackend backend = BackendFactory.getBackend();
            boolean healthy = backend.checkHealth();

            if (healthy) {
                logger.info("Classification backend is healthy");
            } else {
                logger.warn("Classification backend health check failed");
            }

            return healthy;
        } catch (Exception e) {
            logger.debug("Backend health check exception: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Validates that required configuration is present for training.
     *
     * @return true if configuration is valid for training
     */
    public static boolean validateTrainingConfig() {
        // Check server availability
        if (!checkServerHealth()) {
            logger.error("Training validation failed: server not available");
            return false;
        }

        // Check tile size is reasonable
        int tileSize = DLClassifierPreferences.getTileSize();
        if (tileSize < 64 || tileSize > 2048) {
            logger.error("Training validation failed: tile size {} is outside valid range [64, 2048]", tileSize);
            return false;
        }

        // Check overlap is valid
        int overlap = DLClassifierPreferences.getTileOverlap();
        if (overlap < 0 || overlap >= tileSize / 2) {
            logger.error("Training validation failed: overlap {} is invalid for tile size {}", overlap, tileSize);
            return false;
        }

        return true;
    }

    /**
     * Validates that required configuration is present for inference.
     *
     * @return true if configuration is valid for inference
     */
    public static boolean validateInferenceConfig() {
        // Check server availability
        if (!checkServerHealth()) {
            logger.error("Inference validation failed: server not available");
            return false;
        }

        return true;
    }

    /**
     * Gets information about the server's GPU capabilities.
     *
     * @return GPU info string, or "Unknown" if unavailable
     */
    public static String getGPUInfo() {
        try {
            ClassifierBackend backend = BackendFactory.getBackend();
            return backend.getGPUInfo();
        } catch (Exception e) {
            logger.debug("Failed to get GPU info: {}", e.getMessage());
            return "Unknown";
        }
    }
}
