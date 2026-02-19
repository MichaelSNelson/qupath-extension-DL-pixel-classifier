package qupath.ext.dlclassifier.controller;

import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.DLClassifierChecks;
import qupath.ext.dlclassifier.preferences.DLClassifierPreferences;
import qupath.ext.dlclassifier.service.BackendFactory;
import qupath.ext.dlclassifier.service.ClassifierBackend;
import qupath.ext.dlclassifier.service.HttpClassifierBackend;
import qupath.lib.gui.QuPathGUI;

/**
 * Main controller for the DL Pixel Classifier extension.
 * <p>
 * This controller routes menu selections to the appropriate workflow
 * handlers for training, inference, and model management.
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class DLClassifierController {

    private static final Logger logger = LoggerFactory.getLogger(DLClassifierController.class);
    private static DLClassifierController instance;

    private final QuPathGUI qupath;
    private final TrainingWorkflow trainingWorkflow;
    private final InferenceWorkflow inferenceWorkflow;
    private final ModelManagementWorkflow modelManagementWorkflow;

    private DLClassifierController() {
        this.qupath = QuPathGUI.getInstance();
        this.trainingWorkflow = new TrainingWorkflow();
        this.inferenceWorkflow = new InferenceWorkflow();
        this.modelManagementWorkflow = new ModelManagementWorkflow();

        logger.info("DLClassifierController initialized");
    }

    /**
     * Gets the singleton instance of the controller.
     *
     * @return the controller instance
     */
    public static synchronized DLClassifierController getInstance() {
        if (instance == null) {
            instance = new DLClassifierController();
        }
        return instance;
    }

    /**
     * Starts the specified workflow.
     *
     * @param workflowName the workflow to start
     */
    public void startWorkflow(String workflowName) {
        logger.info("Starting workflow: {}", workflowName);

        switch (workflowName) {
            case "training" -> trainingWorkflow.start();
            case "inference" -> inferenceWorkflow.start();
            case "modelManagement" -> modelManagementWorkflow.start();
            case "serverSettings" -> showServerSettings();
            default -> logger.warn("Unknown workflow: {}", workflowName);
        }
    }

    /**
     * Shows the server settings dialog.
     */
    private void showServerSettings() {
        logger.info("Showing server settings dialog");

        Dialog<ButtonType> dialog = new Dialog<>();
        dialog.initOwner(qupath.getStage());
        dialog.setTitle("DL Pixel Classifier - Server Settings");
        dialog.setHeaderText("Configure the Python classification server connection");
        dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);

        // Host and port fields
        TextField hostField = new TextField(DLClassifierPreferences.getServerHost());
        hostField.setPromptText("localhost");

        Spinner<Integer> portSpinner = new Spinner<>(1024, 65535,
                DLClassifierPreferences.getServerPort());
        portSpinner.setEditable(true);
        portSpinner.setPrefWidth(100);

        // Status indicators
        Label statusLabel = new Label("Not checked");
        statusLabel.setStyle("-fx-text-fill: #888;");

        Label gpuLabel = new Label("");

        // Test connection button
        Button testButton = new Button("Test Connection");
        testButton.setOnAction(e -> {
            statusLabel.setText("Checking...");
            statusLabel.setStyle("-fx-text-fill: #888;");
            gpuLabel.setText("");

            Thread checkThread = new Thread(() -> {
                try {
                    // Test connection to the specified host/port
                    String host = hostField.getText().trim();
                    int port = portSpinner.getValue();
                    HttpClassifierBackend testBackend = new HttpClassifierBackend(host, port);
                    boolean healthy = testBackend.checkHealth();

                    String gpuInfo = "";
                    if (healthy) {
                        try {
                            gpuInfo = testBackend.getGPUInfo();
                        } catch (Exception ex) {
                            gpuInfo = "Could not retrieve GPU info";
                        }
                    }

                    final boolean isHealthy = healthy;
                    final String finalGpuInfo = gpuInfo;
                    Platform.runLater(() -> {
                        if (isHealthy) {
                            statusLabel.setText("Connected");
                            statusLabel.setStyle("-fx-text-fill: green; -fx-font-weight: bold;");
                            gpuLabel.setText("GPU: " + finalGpuInfo);
                        } else {
                            statusLabel.setText("Server not responding");
                            statusLabel.setStyle("-fx-text-fill: red;");
                            gpuLabel.setText("");
                        }
                    });
                } catch (Exception ex) {
                    Platform.runLater(() -> {
                        statusLabel.setText("Connection failed: " + ex.getMessage());
                        statusLabel.setStyle("-fx-text-fill: red;");
                        gpuLabel.setText("");
                    });
                }
            }, "DLClassifier-ConnectionTest");
            checkThread.setDaemon(true);
            checkThread.start();
        });

        // Layout
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new Insets(15));

        grid.add(new Label("Server Host:"), 0, 0);
        grid.add(hostField, 1, 0);
        GridPane.setHgrow(hostField, Priority.ALWAYS);

        grid.add(new Label("Server Port:"), 0, 1);
        grid.add(portSpinner, 1, 1);

        grid.add(new Separator(), 0, 2, 2, 1);

        HBox testBox = new HBox(10, testButton, statusLabel);
        testBox.setPadding(new Insets(5, 0, 0, 0));
        grid.add(testBox, 0, 3, 2, 1);
        grid.add(gpuLabel, 0, 4, 2, 1);

        dialog.getDialogPane().setContent(grid);
        dialog.getDialogPane().setPrefWidth(400);

        // Handle result
        dialog.showAndWait().ifPresent(result -> {
            if (result == ButtonType.OK) {
                String newHost = hostField.getText().trim();
                int newPort = portSpinner.getValue();

                if (!newHost.isEmpty()) {
                    DLClassifierPreferences.setServerHost(newHost);
                    DLClassifierPreferences.setServerPort(newPort);
                    logger.info("Server settings updated: {}:{}", newHost, newPort);
                }
            }
        });
    }

    /**
     * Gets the QuPath GUI instance.
     *
     * @return the QuPath GUI
     */
    public QuPathGUI getQuPath() {
        return qupath;
    }
}
