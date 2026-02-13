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
        // Build request body
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model_type", trainingConfig.getModelType());

        // Architecture params
        Map<String, Object> architecture = new HashMap<>();
        architecture.put("backbone", trainingConfig.getBackbone());
        architecture.put("input_size", List.of(trainingConfig.getTileSize(), trainingConfig.getTileSize()));
        architecture.put("use_pretrained", trainingConfig.isUsePretrainedWeights());
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

            // Poll for progress with cancellation support
            return pollTrainingProgress(jobId, progressCallback, cancelledCheck);
        }
    }

    /**
     * Polls for training progress until completion, failure, or cancellation.
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
                    double valLoss = status.get("loss").getAsDouble();
                    double trainLoss = status.has("train_loss") ? status.get("train_loss").getAsDouble() : valLoss;
                    double accuracy = status.has("accuracy") ? status.get("accuracy").getAsDouble() : 0;

                    TrainingProgress progress = new TrainingProgress(epoch, totalEpochs, trainLoss, valLoss, accuracy);
                    if (progressCallback != null) {
                        progressCallback.accept(progress);
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
                                                   Path outputDir) throws IOException {
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
        java.nio.FloatBuffer buffer = java.nio.ByteBuffer.wrap(bytes)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer();

        // Data is in CHW order from Python, convert to HWC for TileProcessor
        float[][][] result = new float[height][width][numClasses];
        for (int c = 0; c < numClasses; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result[h][w][c] = buffer.get(c * height * width + h * width + w);
                }
            }
        }
        return result;
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
        Request request = new Request.Builder()
                .url(baseUrl + "/pretrained/freeze-recommendations/" + datasetSize)
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
    public record TrainingProgress(int epoch, int totalEpochs, double loss, double valLoss, double accuracy) {
        public double getProgress() {
            return (double) epoch / totalEpochs;
        }
    }

    /**
     * Training result information.
     */
    public record TrainingResult(String jobId, String modelPath, double finalLoss, double finalAccuracy) {
        /** Returns true if training was cancelled (no model produced). */
        public boolean isCancelled() {
            return modelPath == null;
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
