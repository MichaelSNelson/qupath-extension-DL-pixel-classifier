package qupath.ext.dlclassifier.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Factory for obtaining the {@link ApposeClassifierBackend}.
 * <p>
 * Returns an Appose-based backend that runs Python inference
 * in an embedded environment via shared memory IPC.
 *
 * @author UW-LOCI
 * @since 0.2.0
 */
public final class BackendFactory {

    private static final Logger logger = LoggerFactory.getLogger(BackendFactory.class);

    private BackendFactory() {
        // Utility class
    }

    /**
     * Gets the Appose backend, initializing if necessary.
     *
     * @return the Appose backend
     * @throws IllegalStateException if Appose is not available
     */
    public static ClassifierBackend getBackend() {
        ApposeService appose = ApposeService.getInstance();
        if (!appose.isAvailable() && appose.getInitError() == null) {
            // Not yet initialized (background thread still running or hasn't started).
            // Try initializing here -- synchronized, so if the background thread is
            // already running initialize(), this blocks until it finishes.
            try {
                logger.info("Appose not yet available, waiting for initialization...");
                appose.initialize();
            } catch (Exception e) {
                logger.warn("Appose initialization failed: {}", e.getMessage());
            }
        }
        if (appose.isAvailable()) {
            return new ApposeClassifierBackend();
        }
        String error = appose.getInitError() != null
                ? appose.getInitError()
                : "Appose environment not initialized";
        throw new IllegalStateException(
                "DL classifier backend not available: " + error
                + ". Use Extensions > DL Pixel Classifier > Setup DL Environment to install.");
    }
}
