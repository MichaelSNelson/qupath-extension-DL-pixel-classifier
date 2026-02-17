package qupath.ext.dlclassifier.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ChannelConfiguration;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.TrainingConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Backend implementation that delegates to the HTTP-based {@link ClassifierClient}.
 * <p>
 * This is a thin adapter that wraps the existing REST API client to conform
 * to the {@link ClassifierBackend} interface.
 *
 * @author UW-LOCI
 * @since 0.2.0
 */
public class HttpClassifierBackend implements ClassifierBackend {

    private static final Logger logger = LoggerFactory.getLogger(HttpClassifierBackend.class);

    private final String host;
    private final int port;

    /**
     * Creates a new HTTP backend.
     *
     * @param host server hostname
     * @param port server port
     */
    public HttpClassifierBackend(String host, int port) {
        this.host = host;
        this.port = port;
    }

    private ClassifierClient createClient() {
        return new ClassifierClient(host, port);
    }

    // ==================== Health & Status ====================

    @Override
    public boolean checkHealth() {
        try {
            return createClient().checkHealth();
        } catch (Exception e) {
            logger.debug("HTTP health check failed: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public String getGPUInfo() {
        try {
            return createClient().getGPUInfo();
        } catch (Exception e) {
            logger.debug("HTTP GPU info failed: {}", e.getMessage());
            return "Unknown";
        }
    }

    @Override
    public String clearGPUMemory() {
        return createClient().clearGPUMemory();
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
        return createClient().startTraining(
                trainingConfig, channelConfig, classNames, trainingDataPath,
                progressCallback, cancelledCheck, jobIdCallback);
    }

    @Override
    public void pauseTraining(String jobId) throws IOException {
        createClient().pauseTraining(jobId);
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
        return createClient().resumeTraining(
                jobId, newDataPath, epochs, learningRate, batchSize,
                progressCallback, cancelledCheck);
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
        return createClient().runPixelInferenceBinary(
                modelPath, rawTileBytes, tileIds, tileHeight, tileWidth,
                numChannels, dtype, channelConfig, inferenceConfig,
                outputDir, reflectionPadding);
    }

    @Override
    public ClassifierClient.PixelInferenceResult runPixelInference(
            String modelPath,
            List<ClassifierClient.TileData> tiles,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig,
            Path outputDir,
            int reflectionPadding) throws IOException {
        return createClient().runPixelInference(
                modelPath, tiles, channelConfig, inferenceConfig,
                outputDir, reflectionPadding);
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
        return createClient().runInferenceBinary(
                modelPath, rawTileBytes, tileIds, tileHeight, tileWidth,
                numChannels, dtype, channelConfig, inferenceConfig);
    }

    @Override
    public ClassifierClient.InferenceResult runInference(
            String modelPath,
            List<ClassifierClient.TileData> tiles,
            ChannelConfiguration channelConfig,
            InferenceConfig inferenceConfig) throws IOException {
        return createClient().runInference(
                modelPath, tiles, channelConfig, inferenceConfig);
    }

    // ==================== Pretrained Model Info ====================

    @Override
    public List<ClassifierClient.LayerInfo> getModelLayers(
            String architecture, String encoder,
            int numChannels, int numClasses) throws IOException {
        return createClient().getModelLayers(architecture, encoder, numChannels, numClasses);
    }

    @Override
    public Map<Integer, Boolean> getFreezeRecommendations(
            String datasetSize, String encoder) throws IOException {
        return createClient().getFreezeRecommendations(datasetSize, encoder);
    }
}
