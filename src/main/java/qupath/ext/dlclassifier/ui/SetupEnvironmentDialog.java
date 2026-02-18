package qupath.ext.dlclassifier.ui;

import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.Window;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.service.ApposeService;

import java.util.ResourceBundle;

/**
 * Setup wizard dialog for first-time DL environment installation.
 * <p>
 * Guides the user through downloading and configuring the Python
 * environment with PyTorch and related dependencies (~2-4 GB).
 * Shows download warnings, optional component checkboxes, and
 * progress during installation.
 *
 * @author UW-LOCI
 * @since 0.2.0
 */
public class SetupEnvironmentDialog {

    private static final Logger logger = LoggerFactory.getLogger(SetupEnvironmentDialog.class);
    private static final ResourceBundle res = ResourceBundle.getBundle("qupath.ext.dlclassifier.ui.strings");

    private final Stage stage;
    private final Runnable onComplete;

    // UI components shared across states
    private VBox contentBox;
    private Label statusLabel;
    private ProgressBar progressBar;
    private CheckBox onnxCheckBox;
    private Button beginButton;
    private Button cancelButton;
    private Button retryButton;

    /**
     * Creates a new setup dialog.
     *
     * @param owner      the owner window for modality (typically QuPath's primary stage)
     * @param onComplete callback invoked on successful setup completion
     */
    public SetupEnvironmentDialog(Window owner, Runnable onComplete) {
        this.onComplete = onComplete;
        this.stage = new Stage();
        stage.setTitle(res.getString("setup.title"));
        stage.initModality(Modality.APPLICATION_MODAL);
        if (owner != null) {
            stage.initOwner(owner);
        }
        stage.setResizable(false);

        buildPreSetupView();

        Scene scene = new Scene(contentBox);
        stage.setScene(scene);
    }

    /**
     * Shows the dialog.
     */
    public void show() {
        stage.show();
    }

    // ==================== View States ====================

    private void buildPreSetupView() {
        contentBox = new VBox(12);
        contentBox.setPadding(new Insets(20));
        contentBox.setPrefWidth(500);

        // Title
        Label titleLabel = new Label(res.getString("setup.title"));
        titleLabel.setFont(Font.font(null, FontWeight.BOLD, 14));

        // Description
        Label descLabel = new Label(res.getString("setup.description"));
        descLabel.setWrapText(true);

        // Download warning
        Label downloadLabel = new Label(res.getString("setup.downloadWarning"));
        downloadLabel.setWrapText(true);

        // Metered connection warning
        Label meteredLabel = new Label("[!] " + res.getString("setup.meteredWarning"));
        meteredLabel.setWrapText(true);
        meteredLabel.setStyle("-fx-font-style: italic;");

        // Optional components section
        Label optionalLabel = new Label("Optional components:");
        optionalLabel.setFont(Font.font(null, FontWeight.BOLD, 12));

        onnxCheckBox = new CheckBox(res.getString("setup.onnxOption"));
        onnxCheckBox.setSelected(true);

        Label onnxDescLabel = new Label(res.getString("setup.onnxDescription"));
        onnxDescLabel.setWrapText(true);
        onnxDescLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666666;");
        onnxDescLabel.setPadding(new Insets(0, 0, 0, 24));

        // Environment location
        Label envLocLabel = new Label(res.getString("setup.envLocation"));
        envLocLabel.setFont(Font.font(null, FontWeight.BOLD, 12));

        Label envPathLabel = new Label(ApposeService.getEnvironmentPath().toString());
        envPathLabel.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
        envPathLabel.setPadding(new Insets(0, 0, 0, 8));

        // Buttons
        beginButton = new Button(res.getString("setup.beginSetup"));
        beginButton.setDefaultButton(true);
        beginButton.setOnAction(e -> startSetup());

        cancelButton = new Button("Cancel");
        cancelButton.setCancelButton(true);
        cancelButton.setOnAction(e -> stage.close());

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        HBox buttonBox = new HBox(8, spacer, beginButton, cancelButton);
        buttonBox.setAlignment(Pos.CENTER_RIGHT);

        contentBox.getChildren().addAll(
                titleLabel,
                descLabel,
                downloadLabel,
                meteredLabel,
                optionalLabel,
                onnxCheckBox,
                onnxDescLabel,
                envLocLabel,
                envPathLabel,
                buttonBox
        );
    }

    private void showInProgressView() {
        contentBox.getChildren().clear();

        Label titleLabel = new Label(res.getString("setup.title"));
        titleLabel.setFont(Font.font(null, FontWeight.BOLD, 14));

        Label inProgressLabel = new Label(res.getString("setup.inProgress"));

        statusLabel = new Label("Preparing...");
        statusLabel.setWrapText(true);

        progressBar = new ProgressBar(-1); // indeterminate
        progressBar.setMaxWidth(Double.MAX_VALUE);

        cancelButton = new Button("Cancel");
        cancelButton.setCancelButton(true);
        cancelButton.setOnAction(e -> stage.close());

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        HBox buttonBox = new HBox(8, spacer, cancelButton);
        buttonBox.setAlignment(Pos.CENTER_RIGHT);

        contentBox.getChildren().addAll(
                titleLabel,
                inProgressLabel,
                statusLabel,
                progressBar,
                buttonBox
        );
    }

    private void showCompleteView() {
        contentBox.getChildren().clear();

        Label titleLabel = new Label(res.getString("setup.title"));
        titleLabel.setFont(Font.font(null, FontWeight.BOLD, 14));

        Label completeLabel = new Label("[OK] " + res.getString("setup.complete"));
        completeLabel.setFont(Font.font(null, FontWeight.BOLD, 13));
        completeLabel.setStyle("-fx-text-fill: #2e7d32;");

        // Split on literal \n from properties file
        String detail = res.getString("setup.completeDetail").replace("\\n", "\n");
        Label detailLabel = new Label(detail);
        detailLabel.setWrapText(true);

        Button closeButton = new Button("Close");
        closeButton.setDefaultButton(true);
        closeButton.setOnAction(e -> stage.close());

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        HBox buttonBox = new HBox(8, spacer, closeButton);
        buttonBox.setAlignment(Pos.CENTER_RIGHT);

        // Check GPU status and show platform-appropriate guidance
        ApposeService appose = ApposeService.getInstance();
        String gpuType = appose.getGpuType();

        if ("cpu".equals(gpuType)) {
            // No GPU detected -- show warning with platform-appropriate instructions
            String warningText;
            boolean isMac = System.getProperty("os.name", "").toLowerCase().contains("mac");
            if (isMac) {
                warningText = "[!] No GPU acceleration detected. Training and inference will be slow on CPU.\n\n"
                        + "Apple MPS (Metal) was not found. If you have Apple Silicon (M1/M2/M3),\n"
                        + "ensure you are using a compatible PyTorch version.\n\n"
                        + "Try: Extensions > DL Pixel Classifier > Utilities > Rebuild DL Environment";
            } else {
                warningText = "[!] No GPU (CUDA) detected. Training and inference will be very slow on CPU.\n\n"
                        + "To enable GPU acceleration:\n"
                        + "  1. Install or update your NVIDIA GPU drivers\n"
                        + "  2. Use Extensions > DL Pixel Classifier > Utilities >\n"
                        + "     Rebuild DL Environment to reinstall with GPU support\n\n"
                        + "If you do not have an NVIDIA GPU, CPU mode will still work\n"
                        + "but training will take significantly longer.";
            }
            Label gpuWarning = new Label(warningText);
            gpuWarning.setWrapText(true);
            gpuWarning.setStyle("-fx-text-fill: #e65100; -fx-font-size: 11px; "
                    + "-fx-border-color: #ffcc80; -fx-border-width: 1; "
                    + "-fx-background-color: #fff3e0; -fx-padding: 8;");

            contentBox.getChildren().addAll(
                    titleLabel,
                    completeLabel,
                    detailLabel,
                    gpuWarning,
                    buttonBox
            );
        } else {
            contentBox.getChildren().addAll(
                    titleLabel,
                    completeLabel,
                    detailLabel,
                    buttonBox
            );
        }
    }

    private void showErrorView(String errorMessage) {
        contentBox.getChildren().clear();

        Label titleLabel = new Label(res.getString("setup.title"));
        titleLabel.setFont(Font.font(null, FontWeight.BOLD, 14));

        Label failedLabel = new Label(res.getString("setup.failed"));
        failedLabel.setFont(Font.font(null, FontWeight.BOLD, 13));
        failedLabel.setStyle("-fx-text-fill: #c62828;");

        Label errorLabel = new Label(errorMessage);
        errorLabel.setWrapText(true);
        errorLabel.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");

        retryButton = new Button(res.getString("setup.retry"));
        retryButton.setDefaultButton(true);
        retryButton.setOnAction(e -> startSetup());

        cancelButton = new Button("Cancel");
        cancelButton.setCancelButton(true);
        cancelButton.setOnAction(e -> stage.close());

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        HBox buttonBox = new HBox(8, spacer, retryButton, cancelButton);
        buttonBox.setAlignment(Pos.CENTER_RIGHT);

        contentBox.getChildren().addAll(
                titleLabel,
                failedLabel,
                errorLabel,
                buttonBox
        );
    }

    // ==================== Setup Execution ====================

    private void startSetup() {
        boolean includeOnnx = onnxCheckBox != null && onnxCheckBox.isSelected();

        showInProgressView();

        Thread setupThread = new Thread(() -> {
            try {
                ApposeService.getInstance().initialize(
                        status -> Platform.runLater(() -> {
                            if (statusLabel != null) {
                                statusLabel.setText(status);
                            }
                        }),
                        includeOnnx
                );

                Platform.runLater(() -> {
                    showCompleteView();
                    if (onComplete != null) {
                        onComplete.run();
                    }
                });

            } catch (Exception e) {
                logger.error("Environment setup failed", e);
                Platform.runLater(() -> showErrorView(e.getMessage()));
            }
        }, "DLClassifier-EnvironmentSetup");
        setupThread.setDaemon(true);
        setupThread.start();
    }
}
