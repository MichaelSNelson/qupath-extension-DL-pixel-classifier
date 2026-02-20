package qupath.ext.dlclassifier.service;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.apposed.appose.NDArray;
import org.apposed.appose.Service.ResponseType;
import org.apposed.appose.Service.Task;
import org.apposed.appose.TaskEvent;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Semaphore;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Backend implementation using Appose for embedded Python execution
 * with shared-memory tile transfer.
 * <p>
 * This backend eliminates the need for an external Python server by
 * managing a Python subprocess via Appose. Tile data is transferred
 * via shared memory (zero-copy), avoiding HTTP and file I/O overhead.
 *
 * @author UW-LOCI
 * @since 0.2.0
 */
public class ApposeClassifierBackend implements ClassifierBackend {

    private static final Logger logger = LoggerFactory.getLogger(ApposeClassifierBackend.class);
    private static final Gson gson = new Gson();

    // Limit concurrent Appose inference tasks. The Python side serializes GPU
    // access via inference_lock, but 16+ overlay threads submitting tasks
    // simultaneously creates 16 Python threads that mostly block and die.
    // Permits=2: one task running on GPU, one preparing (normalization, data copy).
    private static final Semaphore inferenceSemaphore = new Semaphore(2, true);

    // ==================== Health & Status ====================

    @Override
    public boolean checkHealth() {
        try {
            ApposeService appose = ApposeService.getInstance();
            if (!appose.isAvailable()) return false;

            Task task = appose.runTask("health_check", Map.of());
            Object healthy = task.outputs.get("healthy");
            return Boolean.TRUE.equals(healthy);
        } catch (Exception e) {
            logger.debug("Appose health check failed: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public String getGPUInfo() {
        try {
            Task task = ApposeService.getInstance().runTask("health_check", Map.of());
            boolean available = Boolean.TRUE.equals(task.outputs.get("gpu_available"));
            if (available) {
                String name = String.valueOf(task.outputs.get("gpu_name"));
                int memoryMb = ((Number) task.outputs.get("gpu_memory_mb")).intValue();
                return String.format("%s (%d MB) [Appose]", name, memoryMb);
            }
            return "No GPU available (CPU mode) [Appose]";
        } catch (Exception e) {
            logger.debug("Appose GPU info failed: {}", e.getMessage());
            return "Unknown [Appose]";
        }
    }

    @Override
    public String clearGPUMemory() {
        try {
            Task task = ApposeService.getInstance().runTask("clear_gpu", Map.of());
            boolean success = Boolean.TRUE.equals(task.outputs.get("success"));
            String message = String.valueOf(task.outputs.get("message"));
            return success ? message : null;
        } catch (Exception e) {
            logger.error("Appose GPU clear failed: {}", e.getMessage());
            return null;
        }
    }

    // ==================== Training ====================

    @Override
    public ClassifierClient.TrainingResult startTraining(
            TrainingConfig trainingConfig,
            ChannelConfiguration channelConfig,
            List<String> classNames,
            Path trainingDataPath,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck,
            Consumer<String> jobIdCallback) throws IOException {

        ApposeService appose = ApposeService.getInstance();

        // Build input maps matching the train.py script expectations
        Map<String, Object> architecture = new HashMap<>();
        architecture.put("backbone", trainingConfig.getBackbone());
        architecture.put("input_size", List.of(trainingConfig.getTileSize(), trainingConfig.getTileSize()));
        architecture.put("downsample", trainingConfig.getDownsample());
        architecture.put("use_pretrained", trainingConfig.isUsePretrainedWeights());
        List<String> frozenLayers = trainingConfig.getFrozenLayers();
        if (frozenLayers != null && !frozenLayers.isEmpty()) {
            architecture.put("frozen_layers", frozenLayers);
        }

        Map<String, Object> inputConfig = buildInputConfig(channelConfig);

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

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("model_type", trainingConfig.getModelType());
        inputs.put("architecture", architecture);
        inputs.put("input_config", inputConfig);
        inputs.put("training_params", trainingParams);
        inputs.put("classes", classNames);
        inputs.put("data_path", trainingDataPath.toString());

        // Generate a synthetic job ID for Appose-based training
        String jobId = "appose-" + System.currentTimeMillis();
        if (jobIdCallback != null) {
            jobIdCallback.accept(jobId);
        }

        // All Appose task operations need TCCL set for Groovy JSON serialization
        // (task.start/cancel/waitFor all go through Messages.encode -> Groovy).
        ClassLoader extensionCL = ApposeService.class.getClassLoader();
        ClassLoader originalCL = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(extensionCL);

        // Retry loop: "thread death" from concurrent overlay tile requests can
        // arrive as a stale error on the training task due to Appose message
        // ordering races. Retry once after a brief delay to let the queue settle.
        int maxAttempts = 2;
        try {
            for (int attempt = 1; attempt <= maxAttempts; attempt++) {
                try {
                    return executeTrainingTask(appose, inputs, jobId, extensionCL,
                            progressCallback, cancelledCheck);
                } catch (IOException e) {
                    String error = e.getMessage() != null ? e.getMessage() : "";
                    boolean isThreadDeath = error.toLowerCase().contains("thread death");
                    if (isThreadDeath && attempt < maxAttempts) {
                        logger.warn("Training task got transient 'thread death' error " +
                                "(attempt {}/{}), retrying after delay...", attempt, maxAttempts);
                        try {
                            Thread.sleep(3000);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            throw new IOException("Training interrupted during retry", ie);
                        }
                        continue;
                    }
                    throw e;
                }
            }
            // Unreachable, but compiler requires it
            throw new IOException("Training failed after " + maxAttempts + " attempts");
        } finally {
            Thread.currentThread().setContextClassLoader(originalCL);
        }
    }

    @Override
    public void pauseTraining(String jobId) throws IOException {
        // Appose does not support pause natively -- training can only be cancelled.
        // For pause/resume, fall back to HTTP backend.
        throw new IOException("Training pause is not supported via Appose. " +
                "Use the HTTP backend for pause/resume functionality.");
    }

    @Override
    public ClassifierClient.TrainingResult resumeTraining(
            String jobId,
            Path newDataPath,
            Integer epochs,
            Double learningRate,
            Integer batchSize,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck) throws IOException {
        // Appose does not support resume natively -- would need to re-start training.
        // For pause/resume, fall back to HTTP backend.
        throw new IOException("Training resume is not supported via Appose. " +
                "Use the HTTP backend for pause/resume functionality.");
    }

    /**
     * Executes a single training task attempt via Appose.
     */
    private ClassifierClient.TrainingResult executeTrainingTask(
            ApposeService appose,
            Map<String, Object> inputs,
            String jobId,
            ClassLoader extensionCL,
            Consumer<ClassifierClient.TrainingProgress> progressCallback,
            Supplier<Boolean> cancelledCheck) throws IOException {

        Task task = appose.createTask("train", inputs);

        // Listen for progress events
        task.listen(event -> {
            if (event.responseType == ResponseType.UPDATE && event.message != null) {
                try {
                    ClassifierClient.TrainingProgress progress = parseProgressJson(event.message);
                    if (progressCallback != null) {
                        progressCallback.accept(progress);
                    }
                } catch (Exception e) {
                    logger.debug("Failed to parse training progress: {}", e.getMessage());
                }
            }
        });

        // Start the task
        task.start();

        // Poll for cancellation in a background thread.
        // The cancel thread also needs TCCL because task.cancel()
        // sends a JSON message via Groovy serialization.
        Thread cancelThread = new Thread(() -> {
            Thread.currentThread().setContextClassLoader(extensionCL);
            while (!task.status.isFinished()) {
                if (cancelledCheck != null && cancelledCheck.get()) {
                    logger.info("Training cancel requested, sending to Appose task");
                    task.cancel();
                    break;
                }
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, "DLClassifier-ApposeTrainCancel");
        cancelThread.setDaemon(true);
        cancelThread.start();

        // Wait for completion
        try {
            task.waitFor();
        } catch (Exception e) {
            if (task.status == org.apposed.appose.Service.TaskStatus.CANCELED) {
                logger.info("Training cancelled via Appose");
                return new ClassifierClient.TrainingResult(jobId, null, 0, 0);
            }
            throw new IOException("Training failed: " + task.error, e);
        }

        String modelPath = String.valueOf(task.outputs.get("model_path"));
        double finalLoss = ((Number) task.outputs.getOrDefault("final_loss", 0.0)).doubleValue();
        double finalAccuracy = ((Number) task.outputs.getOrDefault("final_accuracy", 0.0)).doubleValue();
        int bestEpoch = ((Number) task.outputs.getOrDefault("best_epoch", 0)).intValue();
        double bestMeanIoU = ((Number) task.outputs.getOrDefault("best_mean_iou", 0.0)).doubleValue();

        return new ClassifierClient.TrainingResult(jobId, modelPath, finalLoss, finalAccuracy,
                bestEpoch, bestMeanIoU);
    }

    // ==================== Inference ====================

    @Override
    public ClassifierClient.PixelInferenceResult runPixelInferenceBinary(
            String modelPath,
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

        ApposeService appose = ApposeService.getInstance();
        int numTiles = tileIds.size();

        // For single-tile overlay inference, use shared-memory path
        if (numTiles == 1) {
            return runSingleTilePixelInference(
                    appose, modelPath, rawTileBytes, tileIds.get(0),
                    tileHeight, tileWidth, numChannels, dtype,
                    channelConfig, inferenceConfig, outputDir, reflectionPadding);
        }

        // Multi-tile: use file-based output (same as HTTP backend)
        return runMultiTilePixelInference(
                appose, modelPath, rawTileBytes, tileIds,
                tileHeight, tileWidth, numChannels, dtype,
                channelConfig, inferenceConfig, outputDir, reflectionPadding);
    }

    /**
     * Single-tile pixel inference with full shared-memory round-trip.
     * No file I/O -- probability map returned directly via shared memory.
     * <p>
     * Wrapped with {@link ApposeService#withExtensionClassLoader} because
     * NDArray allocation triggers ServiceLoader discovery of ShmFactory,
     * and QuPath's tile-rendering threads don't have the extension classloader
     * as their TCCL.
     */
    private ClassifierClient.PixelInferenceResult runSingleTilePixelInference(
            ApposeService appose,
            String modelPath,
            byte[] rawTileBytes,
            String tileId,
            int tileHeight,
            int tileWidth,
            int numChannels,
            String dtype,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig,
            Path outputDir,
            int reflectionPadding) throws IOException {

        // Throttle concurrent Appose task submissions. Without this, 16+
        // overlay threads each spawn a Python thread that blocks on
        // inference_lock, and many die with "thread death" before running.
        try {
            inferenceSemaphore.acquire();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Inference interrupted while waiting for semaphore", e);
        }

        try {
            return ApposeService.withExtensionClassLoader(() -> {
                // Create shared memory NDArray for input tile
                NDArray.Shape shape = new NDArray.Shape(
                        NDArray.Shape.Order.C_ORDER, tileHeight, tileWidth, numChannels);
                NDArray inputNd = new NDArray(NDArray.DType.FLOAT32, shape);

                try {
                    // Copy raw bytes into shared memory, converting dtype if needed
                    FloatBuffer fbuf = inputNd.buffer().order(ByteOrder.nativeOrder()).asFloatBuffer();
                    if ("uint8".equals(dtype)) {
                        for (byte b : rawTileBytes) {
                            fbuf.put((b & 0xFF) / 255.0f);
                        }
                    } else {
                        // float32: copy raw bytes directly
                        FloatBuffer srcBuf = ByteBuffer.wrap(rawTileBytes)
                                .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                        fbuf.put(srcBuf);
                    }

                    // Build input config dict
                    Map<String, Object> inputConfig = buildInputConfig(channelConfig);

                    Map<String, Object> inputs = new HashMap<>();
                    inputs.put("model_path", modelPath);
                    inputs.put("tile_data", inputNd);
                    inputs.put("tile_height", tileHeight);
                    inputs.put("tile_width", tileWidth);
                    inputs.put("num_channels", numChannels);
                    inputs.put("input_config", inputConfig);
                    inputs.put("reflection_padding", reflectionPadding);

                    Task task = appose.runTask("inference_pixel", inputs);

                    int numClasses = ((Number) task.outputs.get("num_classes")).intValue();
                    NDArray resultNd = (NDArray) task.outputs.get("probabilities");

                    try {
                        // Read probability map from shared memory and save as .bin file
                        // (matching the existing file-based contract for DLPixelClassifier)
                        Files.createDirectories(outputDir);
                        Path outputPath = outputDir.resolve(tileId + ".bin");

                        // Result is in CHW order (C classes, H height, W width) as float32
                        ByteBuffer resultBuf = resultNd.buffer().order(ByteOrder.nativeOrder());
                        byte[] resultBytes = new byte[resultBuf.remaining()];
                        resultBuf.get(resultBytes);
                        Files.write(outputPath, resultBytes);

                        Map<String, String> outputPaths = Map.of(tileId, outputPath.toString());
                        return new ClassifierClient.PixelInferenceResult(outputPaths, numClasses);
                    } finally {
                        resultNd.close();
                    }
                } finally {
                    inputNd.close();
                }
            });
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Single-tile pixel inference failed", e);
        } finally {
            inferenceSemaphore.release();
        }
    }

    /**
     * Multi-tile pixel inference with file-based output.
     * Wrapped with extension classloader for ShmFactory ServiceLoader discovery.
     */
    private ClassifierClient.PixelInferenceResult runMultiTilePixelInference(
            ApposeService appose,
            String modelPath,
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

        try {
            return ApposeService.withExtensionClassLoader(() -> {
                int numTiles = tileIds.size();
                int pixelsPerTile = tileHeight * tileWidth * numChannels;

                // Convert to float32 if needed
                byte[] float32Bytes;
                if ("uint8".equals(dtype)) {
                    float32Bytes = new byte[numTiles * pixelsPerTile * Float.BYTES];
                    FloatBuffer fbuf = ByteBuffer.wrap(float32Bytes)
                            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                    for (byte b : rawTileBytes) {
                        fbuf.put((b & 0xFF) / 255.0f);
                    }
                } else {
                    float32Bytes = rawTileBytes;
                }

                // Create shared memory NDArray for all tiles
                NDArray.Shape shape = new NDArray.Shape(
                        NDArray.Shape.Order.C_ORDER,
                        numTiles, tileHeight, tileWidth, numChannels);
                NDArray inputNd = new NDArray(NDArray.DType.FLOAT32, shape);

                try {
                    ByteBuffer buf = inputNd.buffer().order(ByteOrder.nativeOrder());
                    buf.put(float32Bytes);

                    Map<String, Object> inputConfig = buildInputConfig(channelConfig);

                    Map<String, Object> inputs = new HashMap<>();
                    inputs.put("model_path", modelPath);
                    inputs.put("tile_data", inputNd);
                    inputs.put("tile_ids", tileIds);
                    inputs.put("tile_height", tileHeight);
                    inputs.put("tile_width", tileWidth);
                    inputs.put("num_channels", numChannels);
                    inputs.put("input_config", inputConfig);
                    inputs.put("output_dir", outputDir.toString());
                    inputs.put("reflection_padding", reflectionPadding);

                    Task task = appose.runTask("inference_pixel_batch", inputs);

                    int numClasses = ((Number) task.outputs.get("num_classes")).intValue();
                    @SuppressWarnings("unchecked")
                    Map<String, String> outputPaths = (Map<String, String>) task.outputs.get("output_paths");
                    return new ClassifierClient.PixelInferenceResult(
                            new HashMap<>(outputPaths), numClasses);
                } finally {
                    inputNd.close();
                }
            });
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Multi-tile pixel inference failed", e);
        }
    }

    @Override
    public ClassifierClient.PixelInferenceResult runPixelInference(
            String modelPath,
            List<ClassifierClient.TileData> tiles,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig,
            Path outputDir,
            int reflectionPadding) throws IOException {
        // For the Appose backend, convert TileData to binary and delegate
        // to the binary path. This avoids needing a separate base64 script.
        // Note: This is a fallback path; the primary path uses runPixelInferenceBinary.
        logger.warn("Appose backend using base64 tile fallback -- this should not normally happen");
        throw new IOException("Appose backend does not support base64 tile transfer. " +
                "Use runPixelInferenceBinary instead.");
    }

    @Override
    public ClassifierClient.InferenceResult runInferenceBinary(
            String modelPath,
            byte[] rawTileBytes,
            List<String> tileIds,
            int tileHeight,
            int tileWidth,
            int numChannels,
            String dtype,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig) throws IOException {

        ApposeService appose = ApposeService.getInstance();

        try {
            return ApposeService.withExtensionClassLoader(() -> {
                int numTiles = tileIds.size();
                int pixelsPerTile = tileHeight * tileWidth * numChannels;

                // Convert to float32 if needed
                byte[] float32Bytes;
                if ("uint8".equals(dtype)) {
                    float32Bytes = new byte[numTiles * pixelsPerTile * Float.BYTES];
                    FloatBuffer fbuf = ByteBuffer.wrap(float32Bytes)
                            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                    for (byte b : rawTileBytes) {
                        fbuf.put((b & 0xFF) / 255.0f);
                    }
                } else {
                    float32Bytes = rawTileBytes;
                }

                NDArray.Shape shape = new NDArray.Shape(
                        NDArray.Shape.Order.C_ORDER,
                        numTiles, tileHeight, tileWidth, numChannels);
                NDArray inputNd = new NDArray(NDArray.DType.FLOAT32, shape);

                try {
                    ByteBuffer buf = inputNd.buffer().order(ByteOrder.nativeOrder());
                    buf.put(float32Bytes);

                    Map<String, Object> inputConfig = buildInputConfig(channelConfig);

                    Map<String, Object> inputs = new HashMap<>();
                    inputs.put("model_path", modelPath);
                    inputs.put("tile_data", inputNd);
                    inputs.put("tile_ids", tileIds);
                    inputs.put("tile_height", tileHeight);
                    inputs.put("tile_width", tileWidth);
                    inputs.put("num_channels", numChannels);
                    inputs.put("input_config", inputConfig);

                    Task task = appose.runTask("inference_batch", inputs);

                    @SuppressWarnings("unchecked")
                    Map<String, Object> rawPredictions =
                            (Map<String, Object>) task.outputs.get("predictions");

                    Map<String, float[]> predictions = new HashMap<>();
                    for (Map.Entry<String, Object> entry : rawPredictions.entrySet()) {
                        @SuppressWarnings("unchecked")
                        List<Number> probs = (List<Number>) entry.getValue();
                        float[] probArray = new float[probs.size()];
                        for (int i = 0; i < probs.size(); i++) {
                            probArray[i] = probs.get(i).floatValue();
                        }
                        predictions.put(entry.getKey(), probArray);
                    }

                    return new ClassifierClient.InferenceResult(predictions);
                } finally {
                    inputNd.close();
                }
            });
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Batch inference failed", e);
        }
    }

    @Override
    public ClassifierClient.InferenceResult runInference(
            String modelPath,
            List<ClassifierClient.TileData> tiles,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig) throws IOException {
        // For the Appose backend, the base64 path is not supported.
        // Callers should use runInferenceBinary.
        logger.warn("Appose backend does not support base64 tile transfer for batch inference");
        throw new IOException("Appose backend does not support base64 tile transfer. " +
                "Use runInferenceBinary instead.");
    }

    // ==================== Pretrained Model Info ====================

    @Override
    public List<ClassifierClient.LayerInfo> getModelLayers(
            String architecture, String encoder,
            int numChannels, int numClasses) throws IOException {

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("architecture", architecture);
        inputs.put("encoder", encoder);
        inputs.put("num_channels", numChannels);
        inputs.put("num_classes", numClasses);

        Task task = ApposeService.getInstance().runTask("get_model_layers", inputs);

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> layers = (List<Map<String, Object>>) task.outputs.get("layers");

        List<ClassifierClient.LayerInfo> result = new ArrayList<>();
        if (layers != null) {
            for (Map<String, Object> layer : layers) {
                result.add(new ClassifierClient.LayerInfo(
                        String.valueOf(layer.get("name")),
                        String.valueOf(layer.get("display_name")),
                        ((Number) layer.getOrDefault("param_count", 0)).intValue(),
                        Boolean.TRUE.equals(layer.get("is_encoder")),
                        ((Number) layer.getOrDefault("depth", 0)).intValue(),
                        Boolean.TRUE.equals(layer.get("recommended_freeze")),
                        String.valueOf(layer.getOrDefault("description", ""))
                ));
            }
        }
        return result;
    }

    @Override
    public Map<Integer, Boolean> getFreezeRecommendations(
            String datasetSize, String encoder) throws IOException {

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("dataset_size", datasetSize);
        if (encoder != null) {
            inputs.put("encoder", encoder);
        }

        Task task = ApposeService.getInstance().runTask("get_freeze_recommendations", inputs);

        @SuppressWarnings("unchecked")
        Map<String, Object> recs = (Map<String, Object>) task.outputs.get("recommendations");

        Map<Integer, Boolean> result = new HashMap<>();
        if (recs != null) {
            for (Map.Entry<String, Object> entry : recs.entrySet()) {
                try {
                    result.put(Integer.parseInt(entry.getKey()),
                            Boolean.TRUE.equals(entry.getValue()));
                } catch (NumberFormatException e) {
                    logger.debug("Skipping non-integer freeze key: {}", entry.getKey());
                }
            }
        }
        return result;
    }

    // ==================== Helpers ====================

    /**
     * Builds the input configuration dict from a ChannelConfiguration.
     * <p>
     * Includes normalization strategy, per-channel flag, clip percentile,
     * fixed min/max range, and precomputed image-level stats when available.
     */
    private static Map<String, Object> buildInputConfig(ChannelConfiguration channelConfig) {
        Map<String, Object> inputConfig = new HashMap<>();
        inputConfig.put("num_channels", channelConfig.getNumChannels());
        inputConfig.put("selected_channels", channelConfig.getSelectedChannels());

        Map<String, Object> normalization = new HashMap<>();
        normalization.put("strategy", channelConfig.getNormalizationStrategy().name().toLowerCase());
        normalization.put("per_channel", channelConfig.isPerChannelNormalization());
        normalization.put("clip_percentile", channelConfig.getClipPercentile());
        // Include fixed range values (previously missing for FIXED_RANGE strategy)
        normalization.put("min", channelConfig.getFixedMin());
        normalization.put("max", channelConfig.getFixedMax());

        // Include precomputed image-level normalization stats when available
        List<Map<String, Double>> precomputedStats = channelConfig.getPrecomputedChannelStats();
        if (precomputedStats != null && !precomputedStats.isEmpty()) {
            normalization.put("precomputed", true);
            normalization.put("channel_stats", precomputedStats);
        }

        inputConfig.put("normalization", normalization);

        return inputConfig;
    }

    /**
     * Parses a JSON training progress message from a task update event.
     */
    private static ClassifierClient.TrainingProgress parseProgressJson(String json) {
        JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
        int epoch = obj.get("epoch").getAsInt();
        int totalEpochs = obj.get("total_epochs").getAsInt();
        double trainLoss = obj.has("train_loss") ? obj.get("train_loss").getAsDouble() : 0;
        double valLoss = obj.has("val_loss") ? obj.get("val_loss").getAsDouble() : 0;
        double accuracy = obj.has("accuracy") ? obj.get("accuracy").getAsDouble() : 0;
        double meanIoU = obj.has("mean_iou") ? obj.get("mean_iou").getAsDouble() : 0;

        Map<String, Double> perClassIoU = parseStringDoubleMap(obj, "per_class_iou");
        Map<String, Double> perClassLoss = parseStringDoubleMap(obj, "per_class_loss");

        // Device info (present in epoch-0 pre-training update, absent in later epochs)
        String device = obj.has("device") ? obj.get("device").getAsString() : null;
        String deviceInfo = obj.has("device_info") ? obj.get("device_info").getAsString() : null;

        return new ClassifierClient.TrainingProgress(
                epoch, totalEpochs, trainLoss, valLoss, accuracy,
                meanIoU, perClassIoU, perClassLoss, device, deviceInfo);
    }

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
}
