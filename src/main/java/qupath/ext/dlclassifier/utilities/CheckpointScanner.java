package qupath.ext.dlclassifier.utilities;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Scans the central checkpoint registry (~/.dlclassifier/checkpoints/) for
 * orphaned best-in-progress training checkpoint files left behind by
 * interrupted training runs (crashes, power loss, force kill).
 * <p>
 * Training runs that complete or cancel normally clean up their own files
 * via _cleanup_best_in_progress on the Python side. Files that remain in
 * the registry belong to runs that died without a chance to clean up.
 * <p>
 * Used by the project-open toast in SetupDLClassifier and the banner in
 * TrainingDialog to surface recovery options to the user.
 */
public final class CheckpointScanner {

    private static final Logger logger = LoggerFactory.getLogger(CheckpointScanner.class);

    private static final Pattern BEST_IN_PROGRESS_PATTERN =
            Pattern.compile("^best_in_progress_(.+)\\.pt$");

    // Files younger than this are likely from a currently-running training
    // (the Python side rewrites the file on every new-best epoch). Excluding
    // them avoids offering to "recover" a live run.
    private static final long MIN_AGE_SECONDS = 120;

    private CheckpointScanner() {}

    /**
     * Record representing a single orphaned checkpoint file.
     *
     * @param file           absolute path to the checkpoint file
     * @param classifierName parsed classifier/model name from the filename
     * @param modified       last-modified time of the file
     * @param sizeBytes      file size in bytes
     */
    public record OrphanedCheckpoint(Path file, String classifierName,
                                      Instant modified, long sizeBytes) {}

    /**
     * Returns the path to the central checkpoint registry directory.
     * Corresponds to TrainingService._save_best_in_progress on the Python side.
     */
    public static Path getRegistryDir() {
        return Path.of(System.getProperty("user.home"), ".dlclassifier", "checkpoints");
    }

    /**
     * Scan the central checkpoint registry for orphaned best-in-progress files.
     *
     * @param activeClassifierNames classifier names currently being trained.
     *                              Files matching these names are excluded
     *                              from the result so a live run is not flagged.
     *                              Pass an empty set if no training is active.
     * @return list of orphaned checkpoints, sorted by modified time descending
     *         (most recent first). Empty list if the registry does not exist.
     */
    public static List<OrphanedCheckpoint> scanCentralRegistry(
            Set<String> activeClassifierNames) {
        Path registry = getRegistryDir();
        if (!Files.isDirectory(registry)) {
            return Collections.emptyList();
        }

        List<OrphanedCheckpoint> orphans = new ArrayList<>();
        Instant cutoff = Instant.now().minusSeconds(MIN_AGE_SECONDS);

        try (Stream<Path> files = Files.list(registry)) {
            files.forEach(file -> {
                String name = file.getFileName().toString();
                Matcher m = BEST_IN_PROGRESS_PATTERN.matcher(name);
                if (!m.matches()) {
                    return;
                }
                String classifierName = m.group(1);
                if (activeClassifierNames != null
                        && activeClassifierNames.contains(classifierName)) {
                    return;
                }
                try {
                    FileTime mtime = Files.getLastModifiedTime(file);
                    Instant modified = mtime.toInstant();
                    if (modified.isAfter(cutoff)) {
                        return; // too fresh, likely a live training
                    }
                    long size = Files.size(file);
                    orphans.add(new OrphanedCheckpoint(
                            file, classifierName, modified, size));
                } catch (IOException e) {
                    logger.debug("Could not stat checkpoint {}: {}", file, e.getMessage());
                }
            });
        } catch (IOException e) {
            logger.warn("Failed to scan checkpoint registry {}: {}", registry, e.getMessage());
            return Collections.emptyList();
        }

        orphans.sort(Comparator.comparing(OrphanedCheckpoint::modified).reversed());
        return orphans;
    }
}
