package qupath.ext.dlclassifier.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;

/**
 * Factory for obtaining the appropriate {@link ClassifierBackend}
 * based on user preferences and runtime availability.
 * <p>
 * When Appose is enabled (default), returns {@link ApposeClassifierBackend}
 * if the Appose environment is available. Falls back to
 * {@link HttpClassifierBackend} otherwise.
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
     * Gets the appropriate backend based on preferences and availability.
     * <p>
     * Priority:
     * <ol>
     *   <li>If useAppose preference is true and Appose is available, returns Appose backend</li>
     *   <li>If useAppose preference is true but Appose is unavailable, logs warning and falls back to HTTP</li>
     *   <li>If useAppose preference is false, returns HTTP backend</li>
     * </ol>
     *
     * @return the selected backend
     */
    public static ClassifierBackend getBackend() {
        if (DLClassifierPreferences.isUseAppose()) {
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
            logger.warn("Appose preference is enabled but service is not available"
                    + (appose.getInitError() != null
                    ? " (" + appose.getInitError() + ")" : "")
                    + ". Falling back to HTTP backend.");
        }
        return new HttpClassifierBackend(
                DLClassifierPreferences.getServerHost(),
                DLClassifierPreferences.getServerPort());
    }

    /**
     * Gets an HTTP backend regardless of preferences.
     * <p>
     * Use this when you specifically need the HTTP backend (e.g., for
     * server settings testing).
     *
     * @return an HTTP backend configured from current preferences
     */
    public static HttpClassifierBackend getHttpBackend() {
        return new HttpClassifierBackend(
                DLClassifierPreferences.getServerHost(),
                DLClassifierPreferences.getServerPort());
    }
}
