package qupath.ext.dlclassifier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.util.Properties;

/**
 * Provides build-time information (version, git hash, build timestamp)
 * baked into the JAR by the Gradle build.
 * <p>
 * This allows verifying exactly which code version is running,
 * which is critical for debugging issues after deployment.
 */
public final class BuildInfo {

    private static final Logger logger = LoggerFactory.getLogger(BuildInfo.class);

    private static final String PROPS_PATH = "/qupath/ext/dlclassifier/build-info.properties";

    private static final String version;
    private static final String gitHash;
    private static final String buildTimestamp;

    static {
        Properties props = new Properties();
        try (InputStream is = BuildInfo.class.getResourceAsStream(PROPS_PATH)) {
            if (is != null) {
                props.load(is);
            } else {
                logger.debug("build-info.properties not found (development mode?)");
            }
        } catch (Exception e) {
            logger.debug("Could not load build info: {}", e.getMessage());
        }
        version = props.getProperty("version", "dev");
        gitHash = props.getProperty("git.hash", "unknown");
        buildTimestamp = props.getProperty("build.timestamp", "unknown");
    }

    private BuildInfo() {}

    /** Extension version from build.gradle.kts (e.g., "0.5.1-dev"). */
    public static String getVersion() {
        return version;
    }

    /** Short git commit hash, with "-dirty" suffix if uncommitted changes existed at build time. */
    public static String getGitHash() {
        return gitHash;
    }

    /** ISO-8601 timestamp of when the JAR was built. */
    public static String getBuildTimestamp() {
        return buildTimestamp;
    }

    /** One-line summary suitable for logging: "v0.5.1-dev (abc1234, 2026-04-03T12:00:00Z)". */
    public static String getSummary() {
        return String.format("v%s (%s, built %s)", version, gitHash, buildTimestamp);
    }
}
