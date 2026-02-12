package qupath.ext.dlclassifier.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
        // Server settings dialog implementation
        logger.info("Showing server settings dialog");
        // TODO: Implement server settings dialog
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
