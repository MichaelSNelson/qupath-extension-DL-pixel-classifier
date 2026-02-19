package qupath.ext.dlclassifier.service;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * HTTP client for communicating with the Python deep learning classification server.
 * <p>
 * This client provides methods for training, inference, and model management via
 * a REST API. It handles serialization, error handling, and progress reporting.
 *
 * <h3>Server Endpoints</h3>
 * <ul>
 *   <li>GET /api/v1/health - Server health check</li>
 *   <li>GET /api/v1/gpu - GPU availability info</li>
 *   <li>GET /api/v1/models - List available models</li>
 *   <li>POST /api/v1/train - Start training job</li>
 *   <li>GET /api/v1/train/{job_id}/status - Training progress</li>
 *   <li>POST /api/v1/inference - Start inference job</li>
 *   <li>GET /api/v1/inference/{job_id}/result - Get inference results</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ClassifierClient {

    private static final Logger logger = LoggerFactory.getLogger(ClassifierClient.class);
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    private static final String API_VERSION = "v1";

    private final String baseUrl;
    private final OkHttpClient httpClient;
    private final Gson gson;

    /**
     * Creates a new classifier client.
     *
     * @param host server hostname
     * @param port server port
     */
    public ClassifierClient(String host, int port) {
        this.baseUrl = String.format("http://%s:%d/api/%s", host, port, API_VERSION);

        this.httpClient = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(300, TimeUnit.SECONDS) // Long timeout for training
                .writeTimeout(60, TimeUnit.SECONDS)
                .build();

        this.gson = new GsonBuilder()
                .setPrettyPrinting()
                .create();

        logger.info("ClassifierClient initialized with base URL: {}", baseUrl);
    }

    // ==================== Health & Status ====================

    /**
     * Checks if the server is healthy and responding.
     *
     * @return true if server is healthy
     */
    public boolean checkHealth() {
        try {
            Request request = new Request.Builder()
                    .url(baseUrl + "/health")
                    .get()
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (response.isSuccessful() && response.body() != null) {
                    JsonObject json = JsonParser.parseString(response.body().string()).getAsJsonObject();
                    return "healthy".equals(json.get("status").getAsString());
                }
            }
        } catch (Exception e) {
            logger.debug("Health check failed: {}", e.getMessage());
        }
        return false;
    }

    /**
     * Gets GPU availability information from the server.
     *
     * @return GPU info string
     */
    public String getGPUInfo() {
        try {
            Request request = new Request.Builder()
                    .url(baseUrl + "/gpu")
                    .get()
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (response.isSuccessful() && response.body() != null) {
                    JsonObject json = JsonParser.parseString(response.body().string()).getAsJsonObject();
                    boolean available = json.get("available").getAsBoolean();
                    if (available) {
                        String name = json.get("name").getAsString();
                        int memory = json.get("memory_mb").getAsInt();
                        return String.format("%s (%d MB)", name, memory);
                    } else {
                        return "No GPU available (CPU mode)";
                    }
                }
            }
        } catch (Exception e) {
            logger.debug("GPU info check failed: {}", e.getMessage());
        }
        return "Unknown";
    }

    /**
     * Forces the server to clear all GPU memory.
     * <p>
     * This cancels any running training jobs, clears the inference model cache,
     * runs garbage collection, and clears the GPU memory cache. Useful after a
     * crash or when GPU memory needs to be reclaimed.
     *
     * @return a summary string describing what was freed, or null on failure
     */
    public String clearGPUMemory() {
        try {
            Request request = new Request.Builder()
                    .url(baseUrl + "/gpu/clear")
                    .post(RequestBody.create("", JSON))
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (response.isSuccessful() && response.body() != null) {
                    JsonObject json = JsonParser.parseString(response.body().string()).getAsJsonObject();

                    StringBuilder summary = new StringBuilder("GPU memory cleared.");

                    // Report cancelled jobs
                    if (json.has("cancelled_jobs") && json.getAsJsonArray("cancelled_jobs").size() > 0) {
                        int count = json.getAsJsonArray("cancelled_jobs").size();
                        summary.append(String.format(" Cancelled %d training job(s).", count));
                    }

                    // Report freed memory
                    if (json.has("freed_mb")) {
                        double freedMb = json.get("freed_mb").getAsDouble();
                        summary.append(String.format(" Freed %.1f MB.", freedMb));
                    }

                    // Report current memory state
                    if (json.has("memory_after")) {
                        JsonObject memAfter = json.getAsJsonObject("memory_after");
                        if (memAfter.has("allocated_mb") && memAfter.has("total_mb")) {
                            double allocated = memAfter.get("allocated_mb").getAsDouble();
                            double total = memAfter.get("total_mb").getAsDouble();
                            summary.append(String.format(" Now using %.0f / %.0f MB.", allocated, total));
                        }
                    }

                    logger.info("GPU memory cleared: {}", summary);
                    return summary.toString();
                }
            }
        } catch (Exception e) {
            logger.error("Failed to clear GPU memory: {}", e.getMessage());
        }
        return null;
    }

    // ==================== Training ====================

    /**
     * Starts a training job on the server.
     *
     * @param trainingConfig    training configuration
     * @param channelConfig     channel configuration
     * @param classNames        list of class names
     * @param trainingDataPath  path to exported training data
     * @param progressCallback  callback for progress updates
     * @return training job result containing model path and metrics
     * @throws IOException if communication fails
     */
    public TrainingResult startTraining(TrainingConfig trainingConfig,
                                        ChannelConfiguration channelConfig,
                                        List<String> classNames,
                                        Path trainingDataPath,
                                        Consumer<TrainingProgress> progressCallback) throws IOException {
        return startTraining(trainingConfig, channelConfig, classNames,
                trainingDataPath, progressCallback, () -> false);
    }

    /**
     * Starts a training job on the server with cancellation support.
     *
     * @param trainingConfig    training configuration
     * @param channelConfig     channel configuration
     * @param classNames        list of class names
     * @param trainingDataPath  path to exported training data
     * @param progressCallback  callback for progress updates
     * @param cancelledCheck    supplier that returns true when cancelled
     * @return training job result containing model path and metrics
     * @throws IOException if communication fails
     */
    public TrainingResult startTraining(TrainingConfig trainingConfig,
                                        ChannelConfiguration channelConfig,
                                        List<String> classNames,
                                        Path trainingDataPath,
                                        Consumer<TrainingProgress> progressCallback,
                                        Supplier<Boolean> cancelledCheck) throws IOException {
        return startTraining(trainingConfig, channelConfig, classNames,
                trainingDataPath, progressCallback, cancelledCheck, null);
    }

    /**
     * Starts a training job on the server with cancellation support and job ID notification.
     *
     * @param trainingConfig    training configuration
     * @param channelConfig     channel configuration
     * @param classNames        list of class names
     * @param trainingDataPath  path to exported training data
     * @param progressCallback  callback for progress updates
     * @param cancelledCheck    supplier that returns true when cancelled
     * @param jobIdCallback     optional callback to receive the job ID once assigned
     * @return training job result containing model path and metrics
     * @throws IOException if communication fails
     */
    public TrainingResult startTraining(TrainingConfig trainingConfig,
                                        ChannelConfiguration channelConfig,
                                        List<String> classNames,
                                        Path trainingDataPath,
                                        Consumer<TrainingProgress> progressCallback,
                                        Supplier<Boolean> cancelledCheck,
                                        Consumer<String> jobIdCallback) throws IOException {
        // Build request body
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model_type", trainingConfig.getModelType());

        // Architecture params
        Map<String, Object> architecture = new HashMap<>();
        architecture.put("backbone", trainingConfig.getBackbone());
        architecture.put("input_size", List.of(trainingConfig.getTileSize(), trainingConfig.getTileSize()));
        architecture.put("downsample", trainingConfig.getDownsample());
        architecture.put("use_pretrained", trainingConfig.isUsePretrainedWeights());
        architecture.put("context_scale", trainingConfig.getContextScale());
        // Pass frozen layer names for transfer learning
        List<String> frozenLayers = trainingConfig.getFrozenLayers();
        if (frozenLayers != null && !frozenLayers.isEmpty()) {
            architecture.put("frozen_layers", frozenLayers);
        }
        requestBody.put("architecture", architecture);

        // Input config
        Map<String, Object> inputConfig = new HashMap<>();
        inputConfig.put("num_channels", channelConfig.getNumChannels());
        inputConfig.put("channel_names", channelConfig.getChannelNames());
        inputConfig.put("bit_depth", channelConfig.getBitDepth());

        Map<String, Object> normalization = new HashMap<>();
        normalization.put("strategy", channelConfig.getNormalizationStrategy().name().toLowerCase());
        normalization.put("per_channel", channelConfig.isPerChannelNormalization());
        normalization.put("clip_percentile", channelConfig.getClipPercentile());
        inputConfig.put("normalization", normalization);
        requestBody.put("input_config", inputConfig);

        // Training params
        Map<String, Object> trainingParams = new HashMap<>();
        trainingParams.put("epochs", trainingConfig.getEpochs());
        trainingParams.put("batch_size", trainingConfig.getBatchSize());
        trainingParams.put("learning_rate", trainingConfig.getLearningRate());
        trainingParams.put("weight_decay", trainingConfig.getWeightDecay());
        trainingParams.put("validation_split", trainingConfig.getValidationSplit());
        trainingParams.put("augmentation", trainingConfig.isAugmentation());
        trainingParams.put("scheduler", trainingConfig.getSchedulerType());
        trainingParams.put("loss_function", trainingConfig.getLossFunction());
        trainingParams.put("early_stopping", true);
        trainingParams.put("early_stopping_patience", trainingConfig.getEarlyStoppingPatience());
        trainingParams.put("early_stopping_metric", trainingConfig.getEarlyStoppingMetric());
        trainingParams.put("mixed_precision", trainingConfig.isMixedPrecision());
        requestBody.put("training", trainingParams);

        // Classes
        requestBody.put("classes", classNames);

        // Data path
        requestBody.put("data_path", trainingDataPath.toString());

        // Start the training request
        String json = gson.toJson(requestBody);
        logger.debug("Training request: {}", json);

        Request request = new Request.Builder()
                .url(baseUrl + "/train")
                .post(RequestBody.create(json, JSON))
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String error = response.body() != null ? response.body().string() : "Unknown error";
                throw new IOException("Training request failed: " + error);
            }

            JsonObject result = JsonParser.parseString(response.body().string()).getAsJsonObject();
            String jobId = result.get("job_id").getAsString();
            logger.info("Training job started: {}", jobId);

            if (jobIdCallback != null) {
                jobIdCallback.accept(jobId);
            }

            // Poll for progress with cancellation support
            return pollTrainingProgress(jobId, progressCallback, cancelledCheck);
        }
    }

    /**
     * Polls for training progress until completion, failure, cancellation, or pause.
     */
    private TrainingResult pollTrainingProgress(String jobId,
                                                Consumer<TrainingProgress> progressCallback,
                                                Supplier<Boolean> cancelledCheck)
            throws IOException {
        while (true) {
            // Check for cancellation before each poll
            if (cancelledCheck.get()) {
                logger.info("Training cancelled by user, sending cancel to server for job: {}", jobId);
                try {
                    cancelTraining(jobId);
                } catch (IOException e) {
                    logger.warn("Failed to send cancel to server: {}", e.getMessage());
                }
                return new TrainingResult(jobId, null, 0, 0);
            }

            Request request = new Request.Builder()
                    .url(baseUrl + "/train/" + jobId + "/status")
                    .get()
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new IOException("Failed to get training status");
                }

                JsonObject status = JsonParser.parseString(response.body().string()).getAsJsonObject();
                String state = status.get("status").getAsString();

                if ("training".equals(state)) {
                    int epoch = status.get("epoch").getAsInt();
                    int totalEpochs = status.get("total_epochs").getAsInt();

                    // Skip epoch 0 -- the server reports this during model setup/download
                    // before any actual training has occurred
                    if (epoch > 0) {
                        double valLoss = status.get("loss").getAsDouble();
                        double trainLoss = status.has("train_loss") ? status.get("train_loss").getAsDouble() : valLoss;
                        double accuracy = status.has("accuracy") ? status.get("accuracy").getAsDouble() : 0;
                        double meanIoU = status.has("mean_iou") ? status.get("mean_iou").getAsDouble() : 0;

                        Map<String, Double> perClassIoU = parseStringDoubleMap(status, "per_class_iou");
                        Map<String, Double> perClassLoss = parseStringDoubleMap(status, "per_class_loss");

                        String device = status.has("device") ? status.get("device").getAsString() : null;
                        String deviceInfo = status.has("device_info") ? status.get("device_info").getAsString() : null;
                        TrainingProgress progress = new TrainingProgress(
                                epoch, totalEpochs, trainLoss, valLoss, accuracy,
                                meanIoU, perClassIoU, perClassLoss, device, deviceInfo);
                        if (progressCallback != null) {
                            progressCallback.accept(progress);
                        }
                    }
                } else if ("completed".equals(state)) {
                    String modelPath = status.get("model_path").getAsString();
                    double finalLoss = status.get("final_loss").getAsDouble();
                    double finalAccuracy = status.get("final_accuracy").getAsDouble();

                    logger.info("Training completed. Model saved to: {}", modelPath);
                    return new TrainingResult(jobId, modelPath, finalLoss, finalAccuracy);
                } else if ("failed".equals(state)) {
                    String error = status.get("error").getAsString();
                    throw new IOException("Training failed: " + error);
                } else if ("cancelled".equals(state)) {
                    logger.info("Training job {} confirmed cancelled by server", jobId);
                    return new TrainingResult(jobId, null, 0, 0);
                } else if ("paused".equals(state)) {
                    int epoch = status.get("epoch").getAsInt();
                    int totalEpochs = status.get("total_epochs").getAsInt();
                    logger.info("Training job {} paused at epoch {}/{}", jobId, epoch, totalEpochs);
                    return new TrainingResult(jobId, null, 0, 0, 0, 0.0, true, epoch, totalEpochs);
                }
            }

            // Wait before polling again
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Training interrupted");
            }
        }
    }

    /**
     * Parses a JSON object field containing a string-to-double map.
     */
    private static Map<String, Double> parseStringDoubleMap(JsonObject parent, String fieldName) {
        Map<String, Double> result = new LinkedHashMap<>();
        if (parent.has(fieldName) && !parent.get(fieldName).isJsonNull()) {
            JsonObject obj = parent.getAsJsonObject(fieldName);
            for (String key : obj.keySet()) {
                result.put(key, obj.get(key).getAsDouble());
            }
        }
        return result;
    }

    /**
     * Cancels a running training job on the server.
     *
     * @param jobId the job ID to cancel
     * @throws IOException if communication fails
     */
    public void cancelTraining(String jobId) throws IOException {
        Request request = new Request.Builder()
                .url(baseUrl + "/train/" + jobId + "/cancel")
                .post(RequestBody.create("", JSON))
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String error = response.body() != null ? response.body().string() : "Unknown error";
                logger.warn("Failed to cancel training job {}: {}", jobId, error);
            } else {
                logger.info("Cancelled training job: {}", jobId);
            }
        }
    }

    /**
     * Pauses a running training job at the end of the current epoch.
     *
     * @param jobId the job ID to pause
     * @throws IOException if communication fails
     */
    public void pauseTraining(String jobId) throws IOException {
        Request request = new Request.Builder()
                .url(baseUrl + "/train/" + jobId + "/pause")
                .post(RequestBody.create("", JSON))
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String error = response.body() != null ? response.body().string() : "Unknown error";
                throw new IOException("Failed to pause training job " + jobId + ": " + error);
            }
            logger.info("Pause requested for training job: {}", jobId);
        }
    }

    /**
     * Resumes a paused training job.
     *
     * @param jobId            the job ID to resume
     * @param newDataPath      optional new data path for re-exported annotations
     * @param epochs           optional new total epochs
     * @param learningRate     optional new learning rate
     * @param batchSize        optional new batch size
     * @param progressCallback callback for progress updates
     * @param cancelledCheck   supplier that returns true when cancelled
     * @return training result (may be completed, cancelled, or paused again)
     * @throws IOException if communication fails
     */
    public TrainingResult resumeTraining(String jobId,
                                          Path newDataPath,
                                          Integer epochs,
                                          Double learningRate,
                                          Integer batchSize,
                                          Consumer<TrainingProgress> progressCallback,
                                          Supplier<Boolean> cancelledCheck) throws IOException {
        Map<String, Object> requestBody = new HashMap<>();
        if (newDataPath != null) {
            requestBody.put("data_path", newDataPath.toString());
        }
        if (epochs != null) {
            requestBody.put("epochs", epochs);
        }
        if (learningRate != null) {
            requestBody.put("learning_rate", learningRate);
        }
        if (batchSize != null) {
            requestBody.put("batch_size", batchSize);
        }

        String json = gson.toJson(requestBody);
        Request request = new Request.Builder()
                .url(baseUrl + "/train/" + jobId + "/resume")
                .post(RequestBody.create(json, JSON))
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String error = response.body() != null ? response.body().string() : "Unknown error";
                throw new IOException("Failed to resume training job " + jobId + ": " + error);
            }
            logger.info("Resumed training job: {}", jobId);
        }

        // Poll for progress until next terminal state
        return pollTrainingProgress(jobId, progressCallback, cancelledCheck);
    }

    // ==================== Inference ====================

    /**
     * Runs inference on a batch of tiles.
     *
     * @param modelPath       path to the trained model
     * @param tiles           list of tile data (base64 encoded or file paths)
     * @param channelConfig   channel configuration
     * @param inferenceConfig inference configuration
     * @return inference results for each tile
     * @throws IOException if communication fails
     */
    public InferenceResult runInference(String modelPath,
                                        List<TileData> tiles,
                                        ChannelConfiguration channelConfig,
                                        InferenceConfig inferenceConfig) throws IOException {
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model_path", modelPath);

        // Input config
        Map<String, Object> inputConfig = new HashMap<>();
        inputConfig.put("num_channels", channelConfig.getNumChannels());
        inputConfig.put("selected_channels", channelConfig.getSelectedChannels());

        Map<String, Object> normalization = new HashMap<>();
        normalization.put("strategy", channelConfig.getNormalizationStrategy().name().toLowerCase());
        normalization.put("per_channel", channelConfig.isPerChannelNormalization());
        normalization.put("clip_percentile", channelConfig.getClipPercentile());
        inputConfig.put("normalization", normalization);
        requestBody.put("input_config", inputConfig);

        // Tiles
        List<Map<String, Object>> tileList = tiles.stream()
                .map(t -> {
                    Map<String, Object> tileMap = new HashMap<>();
                    tileMap.put("id", t.id());
                    tileMap.put("data", t.data());
                    tileMap.put("x", t.x());
                    tileMap.put("y", t.y());
                    return tileMap;
                })
                .toList();
        requestBody.put("tiles", tileList);

        // Inference options
        Map<String, Object> options = new HashMap<>();
        options.put("use_gpu", inferenceConfig.isUseGPU());
        options.put("blend_mode", inferenceConfig.getBlendMode().name().toLowerCase());
        requestBody.put("options", options);

        String json = gson.toJson(requestBody);
        Request request = new Request.Builder()
                .url(baseUrl + "/inference")
                .post(RequestBody.create(json, JSON))
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String error = response.body() != null ? response.body().string() : "Unknown error";
                throw new IOException("Inference request failed: " + error);
            }

            JsonObject result = JsonParser.parseString(response.body().string()).getAsJsonObject();

            // Parse results
            Map<String, float[]> predictions = new HashMap<>();
            JsonObject predObj = result.getAsJsonObject("predictions");
            for (String tileId : predObj.keySet()) {
                var arr = predObj.getAsJsonArray(tileId);
                float[] probs = new float[arr.size()];
                for (int i = 0; i < arr.size(); i++) {
                    probs[i] = arr.get(i).getAsFloat();
                }
                predictions.put(tileId, probs);
            }

            return new InferenceResult(predictions);
        }
    }

    /**
     * Runs pixel-level inference, producing per-pixel probability maps for tile blending.
     *
     * <p>The server saves probability maps as raw float32 binary files in the specified
     * output directory. Each file contains C*H*W float32 values in CHW order, where C is
     * the number of classes and H/W are tile dimensions.</p>
     *
     * @param modelPath       path to the trained model
     * @param tiles           list of tile data (file paths)
     * @param channelConfig   channel configuration
     * @param inferenceConfig inference configuration
     * @param outputDir       directory for probability map files
     * @return pixel inference result with file paths and metadata
     * @throws IOException if communication fails
     */
    public PixelInferenceResult runPixelInference(String modelPath,
                                                   List<TileData> tiles,
                                                   ChannelConfiguration channelConfig,
                                                   InferenceConfig inferenceConfig,
                                                   Path outputDir,
                                                   int reflectionPadding) throws IOException {
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model_path", modelPath);
        requestBody.put("output_dir", outputDir.toString());

        // Input config
        Map<String, Object> inputConfig = new HashMap<>();
        inputConfig.put("num_channels", channelConfig.getNumChannels());
        inputConfig.put("selected_channels", channelConfig.getSelectedChannels());

        Map<String, Object> normalization = new HashMap<>();
        normalization.put("strategy", channelConfig.getNormalizationStrategy().name().toLowerCase());
        normalization.put("per_channel", channelConfig.isPerChannelNormalization());
        normalization.put("clip_percentile", channelConfig.getClipPercentile());
        inputConfig.put("normalization", normalization);
        requestBody.put("input_config", inputConfig);

        // Tiles
        List<Map<String, Object>> tileList = tiles.stream()
                .map(t -> {
                    Map<String, Object> tileMap = new HashMap<>();
                    tileMap.put("id", t.id());
                    tileMap.put("data", t.data());
                    tileMap.put("x", t.x());
                    tileMap.put("y", t.y());
                    return tileMap;
                })
                .toList();
        requestBody.put("tiles", tileList);

        // Options
        Map<String, Object> options = new HashMap<>();
        options.put("use_gpu", inferenceConfig.isUseGPU());
        options.put("blend_mode", inferenceConfig.getBlendMode().name().toLowerCase());
        options.put("reflection_padding", reflectionPadding);
        requestBody.put("options", options);

        String json = gson.toJson(requestBody);
        Request request = new Request.Builder()
                .url(baseUrl + "/inference/pixel")
                .post(RequestBody.create(json, JSON))
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String error = response.body() != null ? response.body().string() : "Unknown error";
                throw new IOException("Pixel inference request failed: " + error);
            }

            JsonObject result = JsonParser.parseString(response.body().string()).getAsJsonObject();

            Map<String, String> outputPaths = new HashMap<>();
            JsonObject pathsObj = result.getAsJsonObject("output_paths");
            for (String tileId : pathsObj.keySet()) {
                outputPaths.put(tileId, pathsObj.get(tileId).getAsString());
            }
            int numClasses = result.get("num_classes").getAsInt();

            return new PixelInferenceResult(outputPaths, numClasses);
        }
    }

    /**
     * Reads a probability map from a raw binary float32 file.
     *
     * @param filePath   path to the binary file
     * @param numClasses number of classes (C dimension)
     * @param height     tile height (H dimension)
     * @param width      tile width (W dimension)
     * @return probability map with shape [height][width][numClasses] (HWC order for TileProcessor)
     * @throws IOException if reading fails
     */
    public static float[][][] readProbabilityMap(Path filePath, int numClasses,
                                                  int height, int width) throws IOException {
        byte[] bytes = java.nio.file.Files.readAllBytes(filePath);

        // Validate file size matches expected dimensions
        long expectedSize = (long) numClasses * height * width * Float.BYTES;
        if (bytes.length != expectedSize) {
            throw new IOException(String.format(
                    "Probability map size mismatch for %s: expected %d bytes (C=%d, H=%d, W=%d) but got %d bytes",
                    filePath.getFileName(), expectedSize, numClasses, height, width, bytes.length));
        }

        java.nio.FloatBuffer buffer = java.nio.ByteBuffer.wrap(bytes)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer();

        // Data is in CHW order from Python, convert to HWC for TileProcessor
        float[][][] result = new float[height][width][numClasses];
        float[] classMin = new float[numClasses];
        float[] classMax = new float[numClasses];
        double[] classSum = new double[numClasses];
        java.util.Arrays.fill(classMin, Float.MAX_VALUE);
        java.util.Arrays.fill(classMax, -Float.MAX_VALUE);

        for (int c = 0; c < numClasses; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float val = buffer.get(c * height * width + h * width + w);
                    result[h][w][c] = val;
                    if (val < classMin[c]) classMin[c] = val;
                    if (val > classMax[c]) classMax[c] = val;
                    classSum[c] += val;
                }
            }
        }

        // Log probability distribution diagnostics
        if (logger.isDebugEnabled()) {
            int totalPixels = height * width;
            StringBuilder sb = new StringBuilder("Probability map stats for ");
            sb.append(filePath.getFileName()).append(": ");
            for (int c = 0; c < numClasses; c++) {
                double mean = classSum[c] / totalPixels;
                sb.append(String.format("C%d[min=%.3f, max=%.3f, mean=%.3f] ", c, classMin[c], classMax[c], mean));
            }
            logger.debug(sb.toString());
        }

        return result;
    }

    // ==================== Binary Inference ====================

    /**
     * Runs inference using binary tile transfer (multipart/form-data).
     * <p>
     * Tiles are sent as raw bytes in a single binary blob. For 8-bit RGB images
     * the bytes are uint8; for multi-channel or high-bit-depth images they are
     * little-endian float32.
     *
     * @param modelPath       path to the trained model
     * @param rawTileBytes    concatenated tile pixels (HWC order, uint8 or float32)
     * @param tileIds         ordered list of tile IDs matching byte order
     * @param tileHeight      height of each tile in pixels
     * @param tileWidth       width of each tile in pixels
     * @param numChannels     number of channels per tile
     * @param dtype           data type of raw bytes ("uint8" or "float32")
     * @param channelConfig   channel configuration
     * @param inferenceConfig inference configuration
     * @return inference results, or null if binary endpoint is unavailable
     * @throws IOException if communication fails for non-404 errors
     */
    public InferenceResult runInferenceBinary(String modelPath,
                                              byte[] rawTileBytes,
                                              List<String> tileIds,
                                              int tileHeight,
                                              int tileWidth,
                                              int numChannels,
                                              String dtype,
                                              ChannelConfiguration channelConfig,
                                              InferenceConfig inferenceConfig) throws IOException {
        // Build metadata JSON
        Map<String, Object> meta = new HashMap<>();
        meta.put("model_path", modelPath);
        meta.put("tile_ids", tileIds);
        meta.put("tile_height", tileHeight);
        meta.put("tile_width", tileWidth);
        meta.put("num_channels", numChannels);
        meta.put("dtype", dtype);

        Map<String, Object> inputConfig = new HashMap<>();
        inputConfig.put("num_channels", channelConfig.getNumChannels());
        inputConfig.put("selected_channels", channelConfig.getSelectedChannels());
        Map<String, Object> normalization = new HashMap<>();
        normalization.put("strategy", channelConfig.getNormalizationStrategy().name().toLowerCase());
        normalization.put("per_channel", channelConfig.isPerChannelNormalization());
        normalization.put("clip_percentile", channelConfig.getClipPercentile());
        inputConfig.put("normalization", normalization);
        meta.put("input_config", inputConfig);

        Map<String, Object> options = new HashMap<>();
        options.put("use_gpu", inferenceConfig.isUseGPU());
        options.put("blend_mode", inferenceConfig.getBlendMode().name().toLowerCase());
        meta.put("options", options);

        String metadataJson = gson.toJson(meta);

        RequestBody body = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("metadata", metadataJson)
                .addFormDataPart("tiles", "tiles.bin",
                        RequestBody.create(rawTileBytes, MediaType.get("application/octet-stream")))
                .build();

        Request request = new Request.Builder()
                .url(baseUrl + "/inference/binary")
                .post(body)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (response.code() == 404) {
                // Binary endpoint not available, caller should fall back
                logger.debug("Binary inference endpoint not available, will use JSON fallback");
                return null;
            }
            if (!response.isSuccessful()) {
                String error = response.body() != null ? response.body().string() : "Unknown error";
                throw new IOException("Binary inference request failed: " + error);
            }

            JsonObject result = JsonParser.parseString(response.body().string()).getAsJsonObject();
            Map<String, float[]> predictions = new HashMap<>();
            JsonObject predObj = result.getAsJsonObject("predictions");
            for (String tileId : predObj.keySet()) {
                var arr = predObj.getAsJsonArray(tileId);
                float[] probs = new float[arr.size()];
                for (int i = 0; i < arr.size(); i++) {
                    probs[i] = arr.get(i).getAsFloat();
                }
                predictions.put(tileId, probs);
            }
            return new InferenceResult(predictions);
        }
    }

    /**
     * Runs pixel-level inference using binary tile transfer.
     *
     * @param modelPath         path to the trained model
     * @param rawTileBytes      concatenated tile pixels (HWC order, uint8 or float32)
     * @param tileIds           ordered list of tile IDs
     * @param tileHeight        height of each tile in pixels
     * @param tileWidth         width of each tile in pixels
     * @param numChannels       number of channels per tile
     * @param dtype             data type of raw bytes ("uint8" or "float32")
     * @param channelConfig     channel configuration
     * @param inferenceConfig   inference configuration
     * @param outputDir         directory for probability map files
     * @param reflectionPadding reflection padding pixels
     * @return pixel inference result, or null if binary endpoint unavailable
     * @throws IOException if communication fails for non-404 errors
     */
    public PixelInferenceResult runPixelInferenceBinary(String modelPath,
                                                         byte[] rawTileBytes,
                                                         List<String> tileIds,
                                                         int tileHeight,
                                                         int tileWidth,
                                                         int numChannels,
                                                         String dtype,
                                                         ChannelConfiguration channelConfig,
                                                         InferenceConfig inferenceConfig,
                                                         Path outputDir,
                                                         int reflectionPadding) throws IOException {
        Map<String, Object> meta = new HashMap<>();
        meta.put("model_path", modelPath);
        meta.put("output_dir", outputDir.toString());
        meta.put("tile_ids", tileIds);
        meta.put("tile_height", tileHeight);
        meta.put("tile_width", tileWidth);
        meta.put("num_channels", numChannels);
        meta.put("dtype", dtype);

        Map<String, Object> inputConfig = new HashMap<>();
        inputConfig.put("num_channels", channelConfig.getNumChannels());
        inputConfig.put("selected_channels", channelConfig.getSelectedChannels());
        Map<String, Object> normalization = new HashMap<>();
        normalization.put("strategy", channelConfig.getNormalizationStrategy().name().toLowerCase());
        normalization.put("per_channel", channelConfig.isPerChannelNormalization());
        normalization.put("clip_percentile", channelConfig.getClipPercentile());
        inputConfig.put("normalization", normalization);
        meta.put("input_config", inputConfig);

        Map<String, Object> options = new HashMap<>();
        options.put("use_gpu", inferenceConfig.isUseGPU());
        options.put("blend_mode", inferenceConfig.getBlendMode().name().toLowerCase());
        options.put("reflection_padding", reflectionPadding);
        meta.put("options", options);

        String metadataJson = gson.toJson(meta);

        RequestBody body = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("metadata", metadataJson)
                .addFormDataPart("tiles", "tiles.bin",
                        RequestBody.create(rawTileBytes, MediaType.get("application/octet-stream")))
                .build();

        Request request = new Request.Builder()
                .url(baseUrl + "/inference/pixel/binary")
                .post(body)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (response.code() == 404) {
                logger.debug("Binary pixel inference endpoint not available, will use JSON fallback");
                return null;
            }
            if (!response.isSuccessful()) {
                String error = response.body() != null ? response.body().string() : "Unknown error";
                throw new IOException("Binary pixel inference request failed: " + error);
            }

            JsonObject result = JsonParser.parseString(response.body().string()).getAsJsonObject();
            Map<String, String> outputPaths = new HashMap<>();
            JsonObject pathsObj = result.getAsJsonObject("output_paths");
            for (String tileId : pathsObj.keySet()) {
                outputPaths.put(tileId, pathsObj.get(tileId).getAsString());
            }
            int numClasses = result.get("num_classes").getAsInt();
            return new PixelInferenceResult(outputPaths, numClasses);
        }
    }

    // ==================== Pretrained Models ====================

    /**
     * Gets available pretrained encoders from the server.
     *
     * @return list of encoder info objects
     * @throws IOException if communication fails
     */
    public List<EncoderInfo> listEncoders() throws IOException {
        Request request = new Request.Builder()
                .url(baseUrl + "/pretrained/encoders")
                .get()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Failed to list encoders");
            }

            var array = JsonParser.parseString(response.body().string()).getAsJsonArray();
            return array.asList().stream()
                    .map(e -> {
                        JsonObject enc = e.getAsJsonObject();
                        return new EncoderInfo(
                                enc.get("name").getAsString(),
                                enc.get("display_name").getAsString(),
                                enc.get("family").getAsString(),
                                enc.get("params_millions").getAsDouble(),
                                enc.get("license").getAsString()
                        );
                    })
                    .toList();
        }
    }

    /**
     * Gets available segmentation architectures from the server.
     *
     * @return list of architecture info objects
     * @throws IOException if communication fails
     */
    public List<ArchitectureInfo> listArchitectures() throws IOException {
        Request request = new Request.Builder()
                .url(baseUrl + "/pretrained/architectures")
                .get()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Failed to list architectures");
            }

            var array = JsonParser.parseString(response.body().string()).getAsJsonArray();
            return array.asList().stream()
                    .map(e -> {
                        JsonObject arch = e.getAsJsonObject();
                        return new ArchitectureInfo(
                                arch.get("name").getAsString(),
                                arch.get("display_name").getAsString(),
                                arch.get("description").getAsString()
                        );
                    })
                    .toList();
        }
    }

    /**
     * Gets the layer structure of a model for freeze/unfreeze configuration.
     *
     * @param architecture model architecture (e.g., "unet", "deeplabv3plus")
     * @param encoder      encoder name (e.g., "resnet34")
     * @param numChannels  number of input channels
     * @param numClasses   number of output classes
     * @return list of layer info objects
     * @throws IOException if communication fails
     */
    public List<LayerInfo> getModelLayers(String architecture, String encoder,
                                          int numChannels, int numClasses) throws IOException {
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("architecture", architecture);
        requestBody.put("encoder", encoder);
        requestBody.put("num_channels", numChannels);
        requestBody.put("num_classes", numClasses);

        String json = gson.toJson(requestBody);
        Request request = new Request.Builder()
                .url(baseUrl + "/pretrained/layers")
                .post(RequestBody.create(json, JSON))
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String error = response.body() != null ? response.body().string() : "Unknown error";
                throw new IOException("Failed to get model layers: " + error);
            }

            var array = JsonParser.parseString(response.body().string()).getAsJsonArray();
            return array.asList().stream()
                    .map(e -> {
                        JsonObject layer = e.getAsJsonObject();
                        return new LayerInfo(
                                layer.get("name").getAsString(),
                                layer.get("display_name").getAsString(),
                                layer.get("param_count").getAsInt(),
                                layer.get("is_encoder").getAsBoolean(),
                                layer.get("depth").getAsInt(),
                                layer.get("recommended_freeze").getAsBoolean(),
                                layer.get("description").getAsString()
                        );
                    })
                    .toList();
        }
    }

    /**
     * Gets recommended freeze settings for a dataset size.
     *
     * @param datasetSize "small", "medium", or "large"
     * @return map of depth to freeze recommendation
     * @throws IOException if communication fails
     */
    public Map<Integer, Boolean> getFreezeRecommendations(String datasetSize) throws IOException {
        return getFreezeRecommendations(datasetSize, null);
    }

    /**
     * Gets recommended freeze settings for a dataset size and specific encoder.
     * <p>
     * When a histology-pretrained encoder is specified, the server returns less
     * aggressive freeze recommendations since the features are already tissue-relevant.
     *
     * @param datasetSize "small", "medium", or "large"
     * @param encoder     optional encoder name for encoder-specific recommendations
     * @return map of depth to freeze recommendation
     * @throws IOException if communication fails
     */
    public Map<Integer, Boolean> getFreezeRecommendations(String datasetSize,
                                                           String encoder) throws IOException {
        String url = baseUrl + "/pretrained/freeze-recommendations/" + datasetSize;
        if (encoder != null && !encoder.isEmpty()) {
            url += "?encoder=" + encoder;
        }

        Request request = new Request.Builder()
                .url(url)
                .get()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Failed to get freeze recommendations");
            }

            JsonObject result = JsonParser.parseString(response.body().string()).getAsJsonObject();
            JsonObject recs = result.getAsJsonObject("recommendations");

            Map<Integer, Boolean> recommendations = new HashMap<>();
            for (String key : recs.keySet()) {
                recommendations.put(Integer.parseInt(key), recs.get(key).getAsBoolean());
            }
            return recommendations;
        }
    }

    // ==================== Model Management ====================

    /**
     * Lists available models on the server.
     *
     * @return list of model info objects
     * @throws IOException if communication fails
     */
    public List<ModelInfo> listModels() throws IOException {
        Request request = new Request.Builder()
                .url(baseUrl + "/models")
                .get()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Failed to list models");
            }

            JsonObject result = JsonParser.parseString(response.body().string()).getAsJsonObject();
            var modelsArray = result.getAsJsonArray("models");

            return modelsArray.asList().stream()
                    .map(e -> {
                        JsonObject m = e.getAsJsonObject();
                        return new ModelInfo(
                                m.get("id").getAsString(),
                                m.get("name").getAsString(),
                                m.get("type").getAsString(),
                                m.get("path").getAsString()
                        );
                    })
                    .toList();
        }
    }

    // ==================== Data Classes ====================

    /**
     * Training progress information.
     */
    public record TrainingProgress(
            int epoch,
            int totalEpochs,
            double loss,
            double valLoss,
            double accuracy,
            double meanIoU,
            Map<String, Double> perClassIoU,
            Map<String, Double> perClassLoss,
            String device,
            String deviceInfo
    ) {
        public double getProgress() {
            return (double) epoch / totalEpochs;
        }
    }

    /**
     * Training result information.
     * <p>
     * {@code finalLoss} and {@code finalAccuracy} reflect the <b>best</b> model
     * (the checkpoint that was actually saved), not the last training epoch.
     */
    public record TrainingResult(
            String jobId,
            String modelPath,
            double finalLoss,
            double finalAccuracy,
            int bestEpoch,
            double bestMeanIoU,
            boolean paused,
            int lastEpoch,
            int totalEpochs
    ) {
        /** Compact constructor for non-paused results. */
        public TrainingResult(String jobId, String modelPath, double finalLoss, double finalAccuracy,
                              int bestEpoch, double bestMeanIoU) {
            this(jobId, modelPath, finalLoss, finalAccuracy, bestEpoch, bestMeanIoU, false, 0, 0);
        }

        /** Compact constructor for cancelled results. */
        public TrainingResult(String jobId, String modelPath, double finalLoss, double finalAccuracy) {
            this(jobId, modelPath, finalLoss, finalAccuracy, 0, 0.0, false, 0, 0);
        }

        /** Returns true if training was cancelled (no model produced and not paused). */
        public boolean isCancelled() {
            return modelPath == null && !paused;
        }

        /** Returns true if training was paused. */
        public boolean isPaused() {
            return paused;
        }
    }

    /**
     * Tile data for inference.
     */
    public record TileData(String id, String data, int x, int y) {}

    /**
     * Inference result containing predictions for each tile.
     */
    public record InferenceResult(Map<String, float[]> predictions) {}

    /**
     * Pixel-level inference result with file paths to probability maps.
     */
    public record PixelInferenceResult(Map<String, String> outputPaths, int numClasses) {}

    /**
     * Model information.
     */
    public record ModelInfo(String id, String name, String type, String path) {}

    /**
     * Pretrained encoder information.
     */
    public record EncoderInfo(
            String name,
            String displayName,
            String family,
            double paramsMillion,
            String license
    ) {}

    /**
     * Segmentation architecture information.
     */
    public record ArchitectureInfo(
            String name,
            String displayName,
            String description
    ) {}

    /**
     * Model layer information for freeze/unfreeze configuration.
     */
    public record LayerInfo(
            String name,
            String displayName,
            int paramCount,
            boolean isEncoder,
            int depth,
            boolean recommendedFreeze,
            String description
    ) {}
}
