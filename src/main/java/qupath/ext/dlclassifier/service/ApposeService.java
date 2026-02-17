package qupath.ext.dlclassifier.service;

import org.apposed.appose.Appose;
import org.apposed.appose.Environment;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.ResponseType;
import org.apposed.appose.TaskException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Singleton managing the Appose Environment and Python Service lifecycle.
 * <p>
 * Provides an embedded Python runtime for DL inference and training via
 * Appose's shared-memory IPC. The Python worker is a single long-lived
 * subprocess -- globals set in {@code init()} persist across task calls,
 * enabling model caching without per-request overhead.
 * <p>
 * The Appose environment is built from a {@code pixi.toml} bundled in the
 * JAR resources. First-time setup downloads Python, PyTorch, and
 * dependencies (~2-4 GB). Subsequent launches reuse the cached environment.
 *
 * @author UW-LOCI
 * @since 0.2.0
 */
public class ApposeService {

    private static final Logger logger = LoggerFactory.getLogger(ApposeService.class);

    private static final String RESOURCE_BASE = "qupath/ext/dlclassifier/";
    private static final String PIXI_TOML_RESOURCE = RESOURCE_BASE + "pixi.toml";
    private static final String SCRIPTS_BASE = RESOURCE_BASE + "scripts/";
    private static final String ENV_NAME = "dl-pixel-classifier";

    private static ApposeService instance;

    private Environment environment;
    private Service pythonService;
    private boolean initialized;
    private String initError;

    private ApposeService() {
        // Private constructor for singleton
    }

    /**
     * Gets the singleton instance.
     *
     * @return the ApposeService instance
     */
    public static synchronized ApposeService getInstance() {
        if (instance == null) {
            instance = new ApposeService();
        }
        return instance;
    }

    /**
     * Builds the pixi environment and starts the Python service.
     * <p>
     * This is slow the first time (downloads ~2-4 GB of dependencies)
     * but instant on subsequent runs. Should be called from a background
     * thread with progress reporting.
     *
     * @throws IOException if resource loading or environment build fails
     */
    public synchronized void initialize() throws IOException {
        if (initialized) {
            return;
        }

        try {
            logger.info("Initializing Appose environment...");

            // Load pixi.toml from JAR resources
            String pixiToml = loadResource(PIXI_TOML_RESOURCE);

            // Build the pixi environment (downloads deps on first run)
            environment = Appose.pixi()
                    .content(pixiToml)
                    .name(ENV_NAME)
                    .logDebug()
                    .build();

            logger.info("Appose environment built successfully");

            // Create Python service (lazy - subprocess starts on first task)
            pythonService = environment.python();

            // Register debug output handler
            pythonService.debug(msg -> logger.debug("[Appose Python] {}", msg));

            // Pre-warm NumPy to avoid Windows hang (known Appose gotcha)
            pythonService.init("import numpy");

            // Initialize persistent services in the worker
            String initScript = loadScript("init_services.py");
            pythonService.init(initScript);

            initialized = true;
            initError = null;
            logger.info("Appose Python service initialized");

        } catch (Exception e) {
            initError = e.getMessage();
            initialized = false;
            logger.error("Failed to initialize Appose: {}", e.getMessage(), e);
            throw e instanceof IOException ? (IOException) e : new IOException(e);
        }
    }

    /**
     * Runs a named task script with the given inputs.
     * <p>
     * The script is loaded from JAR resources under
     * {@code scripts/<scriptName>.py}. The Python worker must already
     * be initialized via {@link #initialize()}.
     *
     * @param scriptName script name without .py extension (e.g. "inference_pixel")
     * @param inputs     map of input values passed to the script
     * @return the completed Task with outputs
     * @throws IOException if the service is not available or the task fails
     */
    public Task runTask(String scriptName, Map<String, Object> inputs) throws IOException {
        ensureInitialized();

        String script;
        try {
            script = loadScript(scriptName + ".py");
        } catch (IOException e) {
            throw new IOException("Failed to load task script: " + scriptName, e);
        }

        try {
            Task task = pythonService.task(script, inputs);
            task.listen(event -> {
                if (event.responseType == ResponseType.FAILURE
                        || event.responseType == ResponseType.CRASH) {
                    logger.error("Appose task '{}' {}: {}", scriptName,
                            event.responseType, task.error);
                }
            });
            task.waitFor();
            return task;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Appose task '" + scriptName + "' interrupted", e);
        } catch (TaskException e) {
            throw new IOException("Appose task '" + scriptName + "' failed: " + e.getMessage(), e);
        }
    }

    /**
     * Runs a task script with inputs and a custom event listener.
     * <p>
     * Use this for long-running tasks (training) where progress events
     * need to be forwarded to the UI.
     *
     * @param scriptName    script name without .py extension
     * @param inputs        input values
     * @param eventListener listener for task events (progress, completion, etc.)
     * @return the completed Task with outputs
     * @throws IOException if the service is not available or the task fails
     */
    public Task runTaskWithListener(String scriptName, Map<String, Object> inputs,
                                    java.util.function.Consumer<org.apposed.appose.TaskEvent> eventListener)
            throws IOException {
        ensureInitialized();

        String script;
        try {
            script = loadScript(scriptName + ".py");
        } catch (IOException e) {
            throw new IOException("Failed to load task script: " + scriptName, e);
        }

        try {
            Task task = pythonService.task(script, inputs);
            task.listen(eventListener::accept);
            task.waitFor();
            return task;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Appose task '" + scriptName + "' interrupted", e);
        } catch (TaskException e) {
            throw new IOException("Appose task '" + scriptName + "' failed: " + e.getMessage(), e);
        }
    }

    /**
     * Creates a task without waiting for it. Caller is responsible for
     * calling {@code task.waitFor()} and handling exceptions.
     * <p>
     * Use this for tasks that need cancellation support (e.g. training).
     *
     * @param scriptName script name without .py extension
     * @param inputs     input values
     * @return the Task (not yet started -- call {@code start()} or {@code waitFor()})
     * @throws IOException if the service is not available
     */
    public Task createTask(String scriptName, Map<String, Object> inputs) throws IOException {
        ensureInitialized();

        String script;
        try {
            script = loadScript(scriptName + ".py");
        } catch (IOException e) {
            throw new IOException("Failed to load task script: " + scriptName, e);
        }

        return pythonService.task(script, inputs);
    }

    /**
     * Gracefully shuts down the Python service and environment.
     */
    public synchronized void shutdown() {
        if (pythonService != null) {
            try {
                logger.info("Shutting down Appose Python service...");
                pythonService.close();
            } catch (Exception e) {
                logger.warn("Error closing Python service: {}", e.getMessage());
            }
            pythonService = null;
        }
        initialized = false;
        logger.info("Appose service shut down");
    }

    /**
     * Checks whether the Appose service is initialized and available.
     *
     * @return true if the service is ready for tasks
     */
    public boolean isAvailable() {
        return initialized && initError == null && pythonService != null;
    }

    /**
     * Gets the initialization error message, if any.
     *
     * @return error message, or null if no error
     */
    public String getInitError() {
        return initError;
    }

    // ==================== Resource Loading ====================

    private void ensureInitialized() throws IOException {
        if (!isAvailable()) {
            throw new IOException("Appose service is not available"
                    + (initError != null ? ": " + initError : ""));
        }
    }

    /**
     * Loads a Python script from JAR resources.
     *
     * @param scriptFileName script file name (e.g. "inference_pixel.py")
     * @return script content as string
     * @throws IOException if the script is not found
     */
    String loadScript(String scriptFileName) throws IOException {
        return loadResource(SCRIPTS_BASE + scriptFileName);
    }

    /**
     * Loads a text resource from the JAR.
     */
    private static String loadResource(String resourcePath) throws IOException {
        try (InputStream is = ApposeService.class.getClassLoader()
                .getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new IOException("Resource not found: " + resourcePath);
            }
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(is, StandardCharsets.UTF_8))) {
                return reader.lines().collect(Collectors.joining("\n"));
            }
        }
    }
}
