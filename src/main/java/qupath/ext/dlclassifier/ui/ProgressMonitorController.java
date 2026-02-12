package qupath.ext.dlclassifier.ui;

import javafx.application.Platform;
import javafx.beans.property.*;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * Controller for monitoring training and inference progress.
 * <p>
 * Provides real-time feedback including:
 * <ul>
 *   <li>Progress bars for overall and current task</li>
 *   <li>Training metrics visualization (loss curves)</li>
 *   <li>Time estimation for remaining work</li>
 *   <li>Cancel functionality</li>
 *   <li>Log message display</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class ProgressMonitorController {

    private static final Logger logger = LoggerFactory.getLogger(ProgressMonitorController.class);

    private final Stage stage;
    private final ProgressBar overallProgressBar;
    private final ProgressBar currentProgressBar;
    private final Label statusLabel;
    private final Label timeLabel;
    private final Label detailLabel;
    private final TextArea logArea;
    private final Button cancelButton;
    private final LineChart<Number, Number> lossChart;
    private final XYChart.Series<Number, Number> trainLossSeries;
    private final XYChart.Series<Number, Number> valLossSeries;

    private final DoubleProperty overallProgress = new SimpleDoubleProperty(0);
    private final DoubleProperty currentProgress = new SimpleDoubleProperty(0);
    private final StringProperty status = new SimpleStringProperty("Initializing...");
    private final StringProperty detail = new SimpleStringProperty("");
    private final BooleanProperty cancelled = new SimpleBooleanProperty(false);

    private final AtomicLong startTime = new AtomicLong(0);
    private final AtomicBoolean isRunning = new AtomicBoolean(false);

    private Consumer<Void> onCancelCallback;

    /**
     * Creates a new progress monitor for training.
     *
     * @param title the window title
     * @param showLossChart whether to show the loss chart (for training)
     */
    public ProgressMonitorController(String title, boolean showLossChart) {
        stage = new Stage();
        stage.initModality(Modality.NONE);
        stage.initStyle(StageStyle.DECORATED);
        stage.setTitle(title);
        stage.setResizable(true);

        // Create components
        overallProgressBar = new ProgressBar(0);
        overallProgressBar.setPrefWidth(400);
        overallProgressBar.progressProperty().bind(overallProgress);

        currentProgressBar = new ProgressBar(0);
        currentProgressBar.setPrefWidth(400);
        currentProgressBar.progressProperty().bind(currentProgress);

        statusLabel = new Label();
        statusLabel.textProperty().bind(status);
        statusLabel.setStyle("-fx-font-weight: bold;");

        timeLabel = new Label("Elapsed: 00:00:00");
        timeLabel.setStyle("-fx-text-fill: #666;");

        detailLabel = new Label();
        detailLabel.textProperty().bind(detail);
        detailLabel.setStyle("-fx-text-fill: #666;");
        detailLabel.setWrapText(true);

        logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setPrefRowCount(6);
        logArea.setWrapText(true);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");

        cancelButton = new Button("Cancel");
        cancelButton.setOnAction(e -> handleCancel());

        // Create loss chart
        NumberAxis xAxis = new NumberAxis();
        xAxis.setLabel("Epoch");
        xAxis.setAutoRanging(true);

        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Loss");
        yAxis.setAutoRanging(true);

        lossChart = new LineChart<>(xAxis, yAxis);
        lossChart.setTitle("Training Progress");
        lossChart.setCreateSymbols(false);
        lossChart.setAnimated(false);
        lossChart.setPrefHeight(200);

        trainLossSeries = new XYChart.Series<>();
        trainLossSeries.setName("Train Loss");

        valLossSeries = new XYChart.Series<>();
        valLossSeries.setName("Val Loss");

        lossChart.getData().addAll(trainLossSeries, valLossSeries);

        // Build layout
        VBox root = new VBox(10);
        root.setPadding(new Insets(15));
        root.setAlignment(Pos.CENTER);

        // Status section
        VBox statusBox = new VBox(5);
        statusBox.setAlignment(Pos.CENTER_LEFT);
        statusBox.getChildren().addAll(
                statusLabel,
                new HBox(10, new Label("Overall:"), overallProgressBar),
                new HBox(10, new Label("Current:"), currentProgressBar),
                new HBox(20, timeLabel, detailLabel)
        );

        root.getChildren().add(statusBox);

        // Loss chart (if enabled)
        if (showLossChart) {
            TitledPane chartPane = new TitledPane("Training Metrics", lossChart);
            chartPane.setExpanded(true);
            root.getChildren().add(chartPane);
        }

        // Log section
        TitledPane logPane = new TitledPane("Log", logArea);
        logPane.setExpanded(false);
        VBox.setVgrow(logPane, Priority.ALWAYS);
        root.getChildren().add(logPane);

        // Buttons
        HBox buttonBox = new HBox(10);
        buttonBox.setAlignment(Pos.CENTER_RIGHT);
        buttonBox.getChildren().add(cancelButton);
        root.getChildren().add(buttonBox);

        Scene scene = new Scene(root, showLossChart ? 500 : 450, showLossChart ? 500 : 300);
        stage.setScene(scene);

        // Handle window close
        stage.setOnCloseRequest(e -> {
            if (isRunning.get()) {
                e.consume();
                handleCancel();
            }
        });

        // Start time updater
        startTimeUpdater();
    }

    /**
     * Shows the progress monitor.
     */
    public void show() {
        Platform.runLater(() -> {
            startTime.set(System.currentTimeMillis());
            isRunning.set(true);
            stage.show();
        });
    }

    /**
     * Hides the progress monitor.
     */
    public void hide() {
        Platform.runLater(() -> {
            isRunning.set(false);
            stage.hide();
        });
    }

    /**
     * Closes the progress monitor.
     */
    public void close() {
        Platform.runLater(() -> {
            isRunning.set(false);
            stage.close();
        });
    }

    /**
     * Sets the overall progress (0.0 to 1.0).
     *
     * @param progress progress value
     */
    public void setOverallProgress(double progress) {
        Platform.runLater(() -> overallProgress.set(Math.max(0, Math.min(1, progress))));
    }

    /**
     * Sets the current task progress (0.0 to 1.0).
     *
     * @param progress progress value
     */
    public void setCurrentProgress(double progress) {
        Platform.runLater(() -> currentProgress.set(Math.max(0, Math.min(1, progress))));
    }

    /**
     * Sets the status message.
     *
     * @param message status message
     */
    public void setStatus(String message) {
        Platform.runLater(() -> status.set(message));
    }

    /**
     * Sets the detail message.
     *
     * @param message detail message
     */
    public void setDetail(String message) {
        Platform.runLater(() -> detail.set(message));
    }

    /**
     * Adds a log message.
     *
     * @param message log message
     */
    public void log(String message) {
        Platform.runLater(() -> {
            logArea.appendText(message + "\n");
            logArea.setScrollTop(Double.MAX_VALUE);
        });
    }

    /**
     * Updates training metrics.
     *
     * @param epoch current epoch
     * @param trainLoss training loss
     * @param valLoss validation loss (or NaN if not available)
     */
    public void updateTrainingMetrics(int epoch, double trainLoss, double valLoss) {
        Platform.runLater(() -> {
            trainLossSeries.getData().add(new XYChart.Data<>(epoch, trainLoss));
            if (!Double.isNaN(valLoss)) {
                valLossSeries.getData().add(new XYChart.Data<>(epoch, valLoss));
            }
        });
    }

    /**
     * Sets the cancel callback.
     *
     * @param callback callback to invoke when cancel is clicked
     */
    public void setOnCancel(Consumer<Void> callback) {
        this.onCancelCallback = callback;
    }

    /**
     * Checks if the operation was cancelled.
     *
     * @return true if cancelled
     */
    public boolean isCancelled() {
        return cancelled.get();
    }

    /**
     * Gets the cancelled property for binding.
     *
     * @return cancelled property
     */
    public BooleanProperty cancelledProperty() {
        return cancelled;
    }

    /**
     * Marks the operation as complete.
     *
     * @param success whether the operation succeeded
     * @param message completion message
     */
    public void complete(boolean success, String message) {
        Platform.runLater(() -> {
            isRunning.set(false);
            cancelButton.setText("Close");
            cancelButton.setOnAction(e -> close());

            if (success) {
                status.set("Complete");
                statusLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: green;");
            } else {
                status.set("Failed");
                statusLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: red;");
            }

            detail.set(message);
            log(message);
        });
    }

    private void handleCancel() {
        if (!isRunning.get()) {
            close();
            return;
        }

        Alert confirm = new Alert(Alert.AlertType.CONFIRMATION);
        confirm.setTitle("Cancel Operation");
        confirm.setHeaderText("Are you sure you want to cancel?");
        confirm.setContentText("The current operation will be stopped.");

        confirm.showAndWait().ifPresent(response -> {
            if (response == ButtonType.OK) {
                cancelled.set(true);
                status.set("Cancelling...");
                cancelButton.setDisable(true);

                if (onCancelCallback != null) {
                    onCancelCallback.accept(null);
                }

                log("Cancellation requested by user");
            }
        });
    }

    private void startTimeUpdater() {
        Thread updater = new Thread(() -> {
            while (!Thread.interrupted()) {
                if (isRunning.get() && startTime.get() > 0) {
                    long elapsed = System.currentTimeMillis() - startTime.get();
                    String timeStr = formatDuration(elapsed);
                    Platform.runLater(() -> timeLabel.setText("Elapsed: " + timeStr));
                }

                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        });
        updater.setDaemon(true);
        updater.setName("ProgressMonitor-TimeUpdater");
        updater.start();
    }

    private String formatDuration(long millis) {
        long seconds = millis / 1000;
        long hours = seconds / 3600;
        long minutes = (seconds % 3600) / 60;
        long secs = seconds % 60;
        return String.format("%02d:%02d:%02d", hours, minutes, secs);
    }

    /**
     * Creates a progress monitor for training.
     *
     * @return new progress monitor configured for training
     */
    public static ProgressMonitorController forTraining() {
        return new ProgressMonitorController("Training Classifier", true);
    }

    /**
     * Creates a progress monitor for inference.
     *
     * @return new progress monitor configured for inference
     */
    public static ProgressMonitorController forInference() {
        return new ProgressMonitorController("Applying Classifier", false);
    }
}
