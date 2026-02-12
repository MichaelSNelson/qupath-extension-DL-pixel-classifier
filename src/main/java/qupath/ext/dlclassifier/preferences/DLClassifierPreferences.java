package qupath.ext.dlclassifier.preferences;

import javafx.beans.property.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.prefs.PathPrefs;

/**
 * Persistent preferences for the DL Pixel Classifier extension.
 * <p>
 * All preferences are stored using QuPath's preference system and persist
 * across sessions.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public final class DLClassifierPreferences {

    private static final Logger logger = LoggerFactory.getLogger(DLClassifierPreferences.class);

    // Server settings
    private static final StringProperty serverHost = PathPrefs.createPersistentPreference(
            "dlclassifier.serverHost", "localhost");

    private static final IntegerProperty serverPort = PathPrefs.createPersistentPreference(
            "dlclassifier.serverPort", 8765);

    // Tile settings
    private static final IntegerProperty tileSize = PathPrefs.createPersistentPreference(
            "dlclassifier.tileSize", 512);

    private static final IntegerProperty tileOverlap = PathPrefs.createPersistentPreference(
            "dlclassifier.tileOverlap", 64);

    private static final DoubleProperty tileOverlapPercent = PathPrefs.createPersistentPreference(
            "dlclassifier.tileOverlapPercent", 12.5);

    // Object output settings
    private static final StringProperty defaultObjectType = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultObjectType", "DETECTION");

    // Training defaults
    private static final IntegerProperty defaultEpochs = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultEpochs", 50);

    private static final IntegerProperty defaultBatchSize = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultBatchSize", 8);

    private static final DoubleProperty defaultLearningRate = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultLearningRate", 0.001);

    private static final BooleanProperty useAugmentation = PathPrefs.createPersistentPreference(
            "dlclassifier.useAugmentation", true);

    private static final BooleanProperty usePretrainedWeights = PathPrefs.createPersistentPreference(
            "dlclassifier.usePretrainedWeights", true);

    // Inference defaults
    private static final BooleanProperty useGPU = PathPrefs.createPersistentPreference(
            "dlclassifier.useGPU", true);

    private static final DoubleProperty minObjectSizeMicrons = PathPrefs.createPersistentPreference(
            "dlclassifier.minObjectSizeMicrons", 10.0);

    private static final DoubleProperty holeFillingMicrons = PathPrefs.createPersistentPreference(
            "dlclassifier.holeFillingMicrons", 5.0);

    // Normalization
    private static final StringProperty defaultNormalization = PathPrefs.createPersistentPreference(
            "dlclassifier.defaultNormalization", "PERCENTILE_99");

    private DLClassifierPreferences() {
        // Utility class - no instantiation
    }

    /**
     * Registers preferences with QuPath's preference panel.
     *
     * @param qupath the QuPath GUI instance
     */
    public static void installPreferences(QuPathGUI qupath) {
        logger.info("Installing DL Pixel Classifier preferences");

        // Preferences will be automatically available through QuPath's preference system
        // Custom preference dialog can be added later if needed
    }

    // ==================== Server Settings ====================

    public static String getServerHost() {
        return serverHost.get();
    }

    public static void setServerHost(String host) {
        serverHost.set(host);
    }

    public static StringProperty serverHostProperty() {
        return serverHost;
    }

    public static int getServerPort() {
        return serverPort.get();
    }

    public static void setServerPort(int port) {
        serverPort.set(port);
    }

    public static IntegerProperty serverPortProperty() {
        return serverPort;
    }

    // ==================== Tile Settings ====================

    public static int getTileSize() {
        return tileSize.get();
    }

    public static void setTileSize(int size) {
        tileSize.set(size);
    }

    public static IntegerProperty tileSizeProperty() {
        return tileSize;
    }

    public static int getTileOverlap() {
        return tileOverlap.get();
    }

    public static void setTileOverlap(int overlap) {
        tileOverlap.set(overlap);
    }

    public static IntegerProperty tileOverlapProperty() {
        return tileOverlap;
    }

    public static double getTileOverlapPercent() {
        return tileOverlapPercent.get();
    }

    public static void setTileOverlapPercent(double percent) {
        tileOverlapPercent.set(percent);
    }

    public static DoubleProperty tileOverlapPercentProperty() {
        return tileOverlapPercent;
    }

    // ==================== Object Output Settings ====================

    public static String getDefaultObjectType() {
        return defaultObjectType.get();
    }

    public static void setDefaultObjectType(String type) {
        defaultObjectType.set(type);
    }

    public static StringProperty defaultObjectTypeProperty() {
        return defaultObjectType;
    }

    // ==================== Training Defaults ====================

    public static int getDefaultEpochs() {
        return defaultEpochs.get();
    }

    public static void setDefaultEpochs(int epochs) {
        defaultEpochs.set(epochs);
    }

    public static IntegerProperty defaultEpochsProperty() {
        return defaultEpochs;
    }

    public static int getDefaultBatchSize() {
        return defaultBatchSize.get();
    }

    public static void setDefaultBatchSize(int batchSize) {
        defaultBatchSize.set(batchSize);
    }

    public static IntegerProperty defaultBatchSizeProperty() {
        return defaultBatchSize;
    }

    public static double getDefaultLearningRate() {
        return defaultLearningRate.get();
    }

    public static void setDefaultLearningRate(double lr) {
        defaultLearningRate.set(lr);
    }

    public static DoubleProperty defaultLearningRateProperty() {
        return defaultLearningRate;
    }

    public static boolean isUseAugmentation() {
        return useAugmentation.get();
    }

    public static void setUseAugmentation(boolean use) {
        useAugmentation.set(use);
    }

    public static BooleanProperty useAugmentationProperty() {
        return useAugmentation;
    }

    public static boolean isUsePretrainedWeights() {
        return usePretrainedWeights.get();
    }

    public static void setUsePretrainedWeights(boolean use) {
        usePretrainedWeights.set(use);
    }

    public static BooleanProperty usePretrainedWeightsProperty() {
        return usePretrainedWeights;
    }

    // ==================== Inference Defaults ====================

    public static boolean isUseGPU() {
        return useGPU.get();
    }

    public static void setUseGPU(boolean use) {
        useGPU.set(use);
    }

    public static BooleanProperty useGPUProperty() {
        return useGPU;
    }

    public static double getMinObjectSizeMicrons() {
        return minObjectSizeMicrons.get();
    }

    public static void setMinObjectSizeMicrons(double size) {
        minObjectSizeMicrons.set(size);
    }

    public static DoubleProperty minObjectSizeMicronsProperty() {
        return minObjectSizeMicrons;
    }

    public static double getHoleFillingMicrons() {
        return holeFillingMicrons.get();
    }

    public static void setHoleFillingMicrons(double size) {
        holeFillingMicrons.set(size);
    }

    public static DoubleProperty holeFillingMicronsProperty() {
        return holeFillingMicrons;
    }

    // ==================== Normalization ====================

    public static String getDefaultNormalization() {
        return defaultNormalization.get();
    }

    public static void setDefaultNormalization(String normalization) {
        defaultNormalization.set(normalization);
    }

    public static StringProperty defaultNormalizationProperty() {
        return defaultNormalization;
    }
}
