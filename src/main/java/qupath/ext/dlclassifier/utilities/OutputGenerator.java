package qupath.ext.dlclassifier.utilities;

import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.operation.union.UnaryUnionOp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.dlclassifier.model.ClassifierMetadata;
import qupath.ext.dlclassifier.model.InferenceConfig;
import qupath.ext.dlclassifier.model.InferenceConfig.OutputObjectType;
import qupath.lib.common.GeneralTools;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.GeometryTools;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;

import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

/**
 * Generates output from classification results.
 * <p>
 * This class converts raw classification probabilities into QuPath outputs:
 * <ul>
 *   <li>Measurements: Area and percentage per class</li>
 *   <li>Objects: Detection objects from connected components</li>
 *   <li>Overlay: Classification overlay for visualization</li>
 * </ul>
 *
 * @author UW-LOCI
 * @since 0.1.0
 */
public class OutputGenerator {

    private static final Logger logger = LoggerFactory.getLogger(OutputGenerator.class);

    private final ImageData<BufferedImage> imageData;
    private final ClassifierMetadata metadata;
    private final InferenceConfig config;
    private final double pixelSizeMicrons;

    /**
     * Creates a new output generator.
     *
     * @param imageData image data
     * @param metadata  classifier metadata
     * @param config    inference configuration
     */
    public OutputGenerator(ImageData<BufferedImage> imageData,
                           ClassifierMetadata metadata,
                           InferenceConfig config) {
        this.imageData = imageData;
        this.metadata = metadata;
        this.config = config;
        this.pixelSizeMicrons = imageData.getServer().getPixelCalibration().getAveragedPixelSizeMicrons();
    }

    /**
     * Generates measurements output for a parent annotation.
     *
     * @param parent      the parent annotation
     * @param predictions probability map [height][width][numClasses]
     */
    public void addMeasurements(PathObject parent, float[][][] predictions) {
        logger.info("Adding measurements to: {}", parent.getName());

        int height = predictions.length;
        int width = predictions[0].length;
        int numClasses = predictions[0][0].length;
        int totalPixels = height * width;

        // Calculate area per class
        double[] classAreas = new double[numClasses];
        int[] classCounts = new int[numClasses];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Find winning class
                int maxClass = 0;
                float maxProb = predictions[y][x][0];
                for (int c = 1; c < numClasses; c++) {
                    if (predictions[y][x][c] > maxProb) {
                        maxProb = predictions[y][x][c];
                        maxClass = c;
                    }
                }
                classCounts[maxClass]++;
            }
        }

        // Convert to area
        double pixelArea = pixelSizeMicrons * pixelSizeMicrons;
        for (int c = 0; c < numClasses; c++) {
            classAreas[c] = classCounts[c] * pixelArea;
        }

        // Add measurements
        var ml = parent.getMeasurementList();
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();

        for (int c = 0; c < numClasses; c++) {
            String className = c < classes.size() ? classes.get(c).name() : "Class " + c;
            double percentage = 100.0 * classCounts[c] / totalPixels;

            ml.put("DL: " + className + " area (um^2)", classAreas[c]);
            ml.put("DL: " + className + " %", percentage);
        }

        logger.info("Added {} measurements", numClasses * 2);
    }

    /**
     * Creates detection objects from classification results.
     *
     * @param predictions probability map [height][width][numClasses]
     * @param offsetX     X offset in image coordinates
     * @param offsetY     Y offset in image coordinates
     * @return list of created detection objects
     */
    public List<PathObject> createObjects(float[][][] predictions, int offsetX, int offsetY) {
        logger.info("Creating objects from predictions");

        List<PathObject> detections = new ArrayList<>();
        int height = predictions.length;
        int width = predictions[0].length;
        int numClasses = predictions[0][0].length;

        // Create classification map
        int[][] classMap = new int[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int maxClass = 0;
                float maxProb = predictions[y][x][0];
                for (int c = 1; c < numClasses; c++) {
                    if (predictions[y][x][c] > maxProb) {
                        maxProb = predictions[y][x][c];
                        maxClass = c;
                    }
                }
                classMap[y][x] = maxClass;
            }
        }

        // Connected component labeling for each class (skip background = class 0)
        for (int classIdx = 1; classIdx < numClasses; classIdx++) {
            List<ROI> rois = extractConnectedComponents(classMap, classIdx, offsetX, offsetY);

            // Get class info
            List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();
            String className = classIdx < classes.size() ? classes.get(classIdx).name() : "Class " + classIdx;
            PathClass pathClass = PathClass.fromString(className);

            // Create detection objects
            for (ROI roi : rois) {
                // Apply minimum size filter
                double areaMicrons = roi.getArea() * pixelSizeMicrons * pixelSizeMicrons;
                if (areaMicrons >= config.getMinObjectSizeMicrons()) {
                    PathObject detection = PathObjects.createDetectionObject(roi, pathClass);
                    detections.add(detection);
                }
            }
        }

        logger.info("Created {} detection objects", detections.size());
        return detections;
    }

    /**
     * Extracts connected components for a class.
     */
    private List<ROI> extractConnectedComponents(int[][] classMap, int targetClass,
                                                 int offsetX, int offsetY) {
        int height = classMap.length;
        int width = classMap[0].length;
        boolean[][] visited = new boolean[height][width];
        List<ROI> rois = new ArrayList<>();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (classMap[y][x] == targetClass && !visited[y][x]) {
                    // Found start of new component - BFS
                    List<int[]> component = new ArrayList<>();
                    Queue<int[]> queue = new LinkedList<>();
                    queue.add(new int[]{x, y});
                    visited[y][x] = true;

                    while (!queue.isEmpty()) {
                        int[] pos = queue.poll();
                        component.add(pos);

                        // Check 4-neighbors
                        int[][] neighbors = {
                                {pos[0] - 1, pos[1]},
                                {pos[0] + 1, pos[1]},
                                {pos[0], pos[1] - 1},
                                {pos[0], pos[1] + 1}
                        };

                        for (int[] n : neighbors) {
                            int nx = n[0], ny = n[1];
                            if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                                    !visited[ny][nx] && classMap[ny][nx] == targetClass) {
                                visited[ny][nx] = true;
                                queue.add(new int[]{nx, ny});
                            }
                        }
                    }

                    // Create ROI from component
                    if (!component.isEmpty()) {
                        ROI roi = createROIFromComponent(component, offsetX, offsetY);
                        if (roi != null) {
                            rois.add(roi);
                        }
                    }
                }
            }
        }

        return rois;
    }

    /**
     * Creates a ROI from a connected component.
     */
    private ROI createROIFromComponent(List<int[]> component, int offsetX, int offsetY) {
        if (component.isEmpty()) return null;

        // Find bounding box
        int minX = Integer.MAX_VALUE, maxX = Integer.MIN_VALUE;
        int minY = Integer.MAX_VALUE, maxY = Integer.MIN_VALUE;

        for (int[] pos : component) {
            minX = Math.min(minX, pos[0]);
            maxX = Math.max(maxX, pos[0]);
            minY = Math.min(minY, pos[1]);
            maxY = Math.max(maxY, pos[1]);
        }

        // For simplicity, create a rectangle ROI
        // A more sophisticated implementation would trace the boundary
        return ROIs.createRectangleROI(
                offsetX + minX,
                offsetY + minY,
                maxX - minX + 1,
                maxY - minY + 1,
                ImagePlane.getDefaultPlane()
        );
    }

    /**
     * Creates a classification overlay image.
     *
     * @param predictions probability map [height][width][numClasses]
     * @return overlay image
     */
    public BufferedImage createOverlay(float[][][] predictions) {
        int height = predictions.length;
        int width = predictions[0].length;
        int numClasses = predictions[0][0].length;

        BufferedImage overlay = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);

        // Get class colors
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Find winning class
                int maxClass = 0;
                float maxProb = predictions[y][x][0];
                for (int c = 1; c < numClasses; c++) {
                    if (predictions[y][x][c] > maxProb) {
                        maxProb = predictions[y][x][c];
                        maxClass = c;
                    }
                }

                // Get color for class
                int rgb;
                if (maxClass < classes.size()) {
                    rgb = parseColor(classes.get(maxClass).color());
                } else {
                    rgb = 0xFF808080; // Gray default
                }

                // Apply alpha based on confidence
                int alpha = (int) (maxProb * 128); // Semi-transparent
                int argb = (alpha << 24) | (rgb & 0x00FFFFFF);
                overlay.setRGB(x, y, argb);
            }
        }

        return overlay;
    }

    /**
     * Parses a hex color string to RGB.
     */
    private int parseColor(String colorStr) {
        if (colorStr == null || colorStr.isEmpty()) {
            return 0xFF808080;
        }
        try {
            if (colorStr.startsWith("#")) {
                colorStr = colorStr.substring(1);
            }
            return Integer.parseInt(colorStr, 16) | 0xFF000000;
        } catch (NumberFormatException e) {
            return 0xFF808080;
        }
    }

    /**
     * Generates measurements output for a parent annotation from multiple tile results.
     *
     * @param parent      the parent annotation
     * @param tileResults list of probability maps for each tile
     * @param tileSpecs   list of tile specifications
     */
    public void addMeasurements(PathObject parent,
                                List<float[][][]> tileResults,
                                List<TileProcessor.TileSpec> tileSpecs) {
        if (tileResults.isEmpty()) {
            logger.warn("No tile results to process for measurements");
            return;
        }

        logger.info("Adding measurements from {} tiles to: {}", tileResults.size(), parent.getName());

        // Aggregate counts across all tiles
        int numClasses = tileResults.get(0)[0][0].length;
        long[] classCounts = new long[numClasses];
        long totalPixels = 0;

        for (float[][][] predictions : tileResults) {
            int height = predictions.length;
            int width = predictions[0].length;
            totalPixels += (long) height * width;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    // Find winning class
                    int maxClass = 0;
                    float maxProb = predictions[y][x][0];
                    for (int c = 1; c < numClasses; c++) {
                        if (predictions[y][x][c] > maxProb) {
                            maxProb = predictions[y][x][c];
                            maxClass = c;
                        }
                    }
                    classCounts[maxClass]++;
                }
            }
        }

        // Convert to area
        double pixelArea = pixelSizeMicrons * pixelSizeMicrons;
        double[] classAreas = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            classAreas[c] = classCounts[c] * pixelArea;
        }

        // Add measurements
        var ml = parent.getMeasurementList();
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();

        for (int c = 0; c < numClasses; c++) {
            String className = c < classes.size() ? classes.get(c).name() : "Class " + c;
            double percentage = totalPixels > 0 ? 100.0 * classCounts[c] / totalPixels : 0;

            ml.put("DL: " + className + " area (um^2)", classAreas[c]);
            ml.put("DL: " + className + " %", percentage);
        }

        ml.put("DL: Total pixels", totalPixels);
        ml.put("DL: Classifier", metadata.getName());

        logger.info("Added {} measurements", numClasses * 2 + 2);
    }

    /**
     * Creates detection objects from multiple tile classification results.
     *
     * @param tileResults list of probability maps for each tile
     * @param tileSpecs   list of tile specifications
     * @param parentROI   the parent region for filtering
     * @return list of created detection objects
     */
    public List<PathObject> createDetectionObjects(List<float[][][]> tileResults,
                                                   List<TileProcessor.TileSpec> tileSpecs,
                                                   ROI parentROI) {
        List<PathObject> allDetections = new ArrayList<>();

        if (tileResults.size() != tileSpecs.size()) {
            logger.error("Mismatch between tile results ({}) and specs ({})",
                    tileResults.size(), tileSpecs.size());
            return allDetections;
        }

        for (int i = 0; i < tileResults.size(); i++) {
            float[][][] predictions = tileResults.get(i);
            TileProcessor.TileSpec spec = tileSpecs.get(i);

            List<PathObject> detections = createObjects(predictions, spec.x(), spec.y());

            // Filter to only include objects within the parent ROI
            for (PathObject detection : detections) {
                ROI detectionROI = detection.getROI();
                if (parentROI == null || parentROI.contains(
                        detectionROI.getCentroidX(), detectionROI.getCentroidY())) {
                    allDetections.add(detection);
                }
            }
        }

        logger.info("Created {} detection objects from {} tiles", allDetections.size(), tileResults.size());
        return allDetections;
    }

    /**
     * Creates PathObjects from a merged classification map.
     * <p>
     * This method processes the ENTIRE merged classification map at once, enabling
     * objects that span tile boundaries to be correctly identified as single objects.
     * Uses Union-Find for efficient connected component labeling.
     *
     * @param classMap   merged classification map [height][width] with class indices
     * @param offsetX    X offset in image coordinates
     * @param offsetY    Y offset in image coordinates
     * @param objectType type of PathObject to create (DETECTION or ANNOTATION)
     * @return list of created PathObjects
     */
    public List<PathObject> createObjectsFromMergedMap(int[][] classMap,
                                                        int offsetX, int offsetY,
                                                        OutputObjectType objectType) {
        logger.info("Creating objects from merged map ({}x{}) at ({},{}), type={}",
                classMap[0].length, classMap.length, offsetX, offsetY, objectType);

        List<PathObject> objects = new ArrayList<>();
        int height = classMap.length;
        int width = classMap[0].length;
        int numClasses = metadata.getClasses().size();

        // Process each class (skip background = class 0)
        for (int classIdx = 1; classIdx < numClasses; classIdx++) {
            List<PathObject> classObjects = extractClassObjects(
                    classMap, classIdx, offsetX, offsetY, objectType);
            objects.addAll(classObjects);
        }

        logger.info("Created {} objects from merged classification map", objects.size());
        return objects;
    }

    /**
     * Creates objects from merged tiles using TileProcessor's complete merge.
     * <p>
     * This is the preferred method for OBJECTS output as it correctly handles
     * objects spanning tile boundaries.
     *
     * @param tileProcessor tile processor to use for merging
     * @param tileResults   probability maps for each tile
     * @param tileSpecs     tile specifications
     * @param parentROI     parent region for coordinate offset and filtering
     * @param numClasses    number of classification classes
     * @param objectType    type of PathObject to create
     * @return list of created PathObjects
     */
    public List<PathObject> createObjectsFromTiles(TileProcessor tileProcessor,
                                                    List<float[][][]> tileResults,
                                                    List<TileProcessor.TileSpec> tileSpecs,
                                                    ROI parentROI,
                                                    int numClasses,
                                                    OutputObjectType objectType) {
        if (tileResults.isEmpty() || tileSpecs.isEmpty()) {
            logger.warn("No tile results to process for object creation");
            return Collections.emptyList();
        }

        int regionX = (int) parentROI.getBoundsX();
        int regionY = (int) parentROI.getBoundsY();
        int regionWidth = (int) parentROI.getBoundsWidth();
        int regionHeight = (int) parentROI.getBoundsHeight();

        // Use edge-aware merging to get complete merged result
        TileProcessor.MergedResult merged = tileProcessor.mergeTileResultsWithEdgeHandling(
                tileSpecs, tileResults,
                regionX, regionY, regionWidth, regionHeight,
                numClasses
        );

        // Create objects from the merged classification map
        List<PathObject> objects = createObjectsFromMergedMap(
                merged.classificationMap(),
                regionX, regionY,
                objectType
        );

        // Filter objects to only include those within the parent ROI
        if (parentROI != null) {
            objects = objects.stream()
                    .filter(obj -> parentROI.contains(
                            obj.getROI().getCentroidX(),
                            obj.getROI().getCentroidY()))
                    .toList();
        }

        return new ArrayList<>(objects);
    }

    /**
     * Extracts connected component objects for a single class using Union-Find.
     */
    private List<PathObject> extractClassObjects(int[][] classMap, int targetClass,
                                                  int offsetX, int offsetY,
                                                  OutputObjectType objectType) {
        int height = classMap.length;
        int width = classMap[0].length;

        // Union-Find for connected component labeling
        UnionFind uf = new UnionFind(width * height);

        // First pass: union adjacent pixels of the same class
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (classMap[y][x] != targetClass) continue;

                int idx = y * width + x;

                // Check right neighbor
                if (x + 1 < width && classMap[y][x + 1] == targetClass) {
                    uf.union(idx, idx + 1);
                }

                // Check bottom neighbor
                if (y + 1 < height && classMap[y + 1][x] == targetClass) {
                    uf.union(idx, (y + 1) * width + x);
                }
            }
        }

        // Second pass: collect pixels by component
        Map<Integer, List<int[]>> components = new HashMap<>();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (classMap[y][x] != targetClass) continue;

                int idx = y * width + x;
                int root = uf.find(idx);
                components.computeIfAbsent(root, k -> new ArrayList<>())
                        .add(new int[]{x, y});
            }
        }

        // Convert components to PathObjects
        List<PathObject> objects = new ArrayList<>();
        List<ClassifierMetadata.ClassInfo> classes = metadata.getClasses();
        String className = targetClass < classes.size() ? classes.get(targetClass).name() : "Class " + targetClass;
        PathClass pathClass = PathClass.fromString(className);

        for (List<int[]> component : components.values()) {
            // Calculate area in microns
            double areaPixels = component.size();
            double areaMicrons = areaPixels * pixelSizeMicrons * pixelSizeMicrons;

            // Apply minimum size filter
            if (areaMicrons < config.getMinObjectSizeMicrons()) {
                continue;
            }

            // Create ROI from component boundary
            ROI roi = createROIFromComponentBoundary(component, offsetX, offsetY);
            if (roi == null) continue;

            // Apply hole filling and smoothing would be done here in a more
            // sophisticated implementation. For now, we just create the object.

            // Create object based on type
            PathObject pathObject;
            if (objectType == OutputObjectType.ANNOTATION) {
                pathObject = PathObjects.createAnnotationObject(roi, pathClass);
            } else {
                pathObject = PathObjects.createDetectionObject(roi, pathClass);
            }

            // Add area measurement
            pathObject.getMeasurementList().put("Area (um^2)", areaMicrons);
            pathObject.getMeasurementList().put("Area (pixels)", areaPixels);

            objects.add(pathObject);
        }

        return objects;
    }

    /**
     * Creates a ROI from component boundary points using convex hull or bounding box.
     */
    private ROI createROIFromComponentBoundary(List<int[]> component, int offsetX, int offsetY) {
        if (component.isEmpty()) return null;

        // For small components, use bounding box
        if (component.size() < 10) {
            return createBoundingBoxROI(component, offsetX, offsetY);
        }

        // For larger components, trace the boundary
        return createBoundaryROI(component, offsetX, offsetY);
    }

    /**
     * Creates a bounding box ROI from component pixels.
     */
    private ROI createBoundingBoxROI(List<int[]> component, int offsetX, int offsetY) {
        int minX = Integer.MAX_VALUE, maxX = Integer.MIN_VALUE;
        int minY = Integer.MAX_VALUE, maxY = Integer.MIN_VALUE;

        for (int[] pos : component) {
            minX = Math.min(minX, pos[0]);
            maxX = Math.max(maxX, pos[0]);
            minY = Math.min(minY, pos[1]);
            maxY = Math.max(maxY, pos[1]);
        }

        return ROIs.createRectangleROI(
                offsetX + minX,
                offsetY + minY,
                maxX - minX + 1,
                maxY - minY + 1,
                ImagePlane.getDefaultPlane()
        );
    }

    /**
     * Creates a polygon ROI by tracing the component boundary.
     */
    private ROI createBoundaryROI(List<int[]> component, int offsetX, int offsetY) {
        // Create a binary mask for the component
        int minX = Integer.MAX_VALUE, maxX = Integer.MIN_VALUE;
        int minY = Integer.MAX_VALUE, maxY = Integer.MIN_VALUE;
        for (int[] pos : component) {
            minX = Math.min(minX, pos[0]);
            maxX = Math.max(maxX, pos[0]);
            minY = Math.min(minY, pos[1]);
            maxY = Math.max(maxY, pos[1]);
        }

        int localWidth = maxX - minX + 3; // +3 for border
        int localHeight = maxY - minY + 3;
        boolean[][] mask = new boolean[localHeight][localWidth];

        for (int[] pos : component) {
            mask[pos[1] - minY + 1][pos[0] - minX + 1] = true;
        }

        // Find boundary points using marching squares (simplified)
        List<double[]> boundaryPoints = traceBoundary(mask);

        if (boundaryPoints.size() < 3) {
            return createBoundingBoxROI(component, offsetX, offsetY);
        }

        // Convert to polygon
        double[] xPoints = new double[boundaryPoints.size()];
        double[] yPoints = new double[boundaryPoints.size()];

        for (int i = 0; i < boundaryPoints.size(); i++) {
            xPoints[i] = offsetX + minX - 1 + boundaryPoints.get(i)[0];
            yPoints[i] = offsetY + minY - 1 + boundaryPoints.get(i)[1];
        }

        return ROIs.createPolygonROI(xPoints, yPoints, ImagePlane.getDefaultPlane());
    }

    /**
     * Traces the boundary of a binary mask (simplified contour tracing).
     */
    private List<double[]> traceBoundary(boolean[][] mask) {
        List<double[]> boundary = new ArrayList<>();
        int height = mask.length;
        int width = mask[0].length;

        // Find starting point on boundary
        int startX = -1, startY = -1;
        outer:
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (mask[y][x] && isBoundaryPixel(mask, x, y)) {
                    startX = x;
                    startY = y;
                    break outer;
                }
            }
        }

        if (startX < 0) return boundary;

        // Simple boundary following
        boolean[][] visited = new boolean[height][width];
        int x = startX, y = startY;
        int direction = 0; // 0=right, 1=down, 2=left, 3=up
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

        int maxSteps = width * height * 2;
        int steps = 0;

        do {
            if (!visited[y][x]) {
                boundary.add(new double[]{x, y});
                visited[y][x] = true;
            }

            // Try to turn left first, then straight, then right, then back
            boolean moved = false;
            for (int turn = -1; turn <= 2; turn++) {
                int newDir = (direction + turn + 4) % 4;
                int nx = x + dirs[newDir][0];
                int ny = y + dirs[newDir][1];

                if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                        mask[ny][nx] && isBoundaryPixel(mask, nx, ny)) {
                    x = nx;
                    y = ny;
                    direction = newDir;
                    moved = true;
                    break;
                }
            }

            if (!moved) break;
            steps++;
        } while ((x != startX || y != startY) && steps < maxSteps);

        return boundary;
    }

    /**
     * Checks if a pixel is on the boundary (has at least one non-object neighbor).
     */
    private boolean isBoundaryPixel(boolean[][] mask, int x, int y) {
        int height = mask.length;
        int width = mask[0].length;

        int[][] neighbors = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int[] n : neighbors) {
            int nx = x + n[0];
            int ny = y + n[1];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height || !mask[ny][nx]) {
                return true;
            }
        }
        return false;
    }

    // ==================== Geometry Union Utilities ====================

    /** Batch size for hierarchical geometry union. */
    private static final int UNION_BATCH_SIZE = 64;

    /** Threshold above which geometry union is parallelized via ForkJoinPool. */
    private static final int PARALLEL_THRESHOLD = 128;

    /**
     * Merges a list of ROIs into a single unified ROI using hierarchical
     * batched union. For large lists (>{@value PARALLEL_THRESHOLD} ROIs),
     * processing is parallelized using a ForkJoinPool.
     * <p>
     * This is useful for combining adjacent or overlapping detection ROIs
     * into a single annotation region.
     *
     * @param rois the ROIs to merge
     * @return the merged ROI, or {@code null} if the list is null or empty
     */
    public static ROI mergeGeometries(List<ROI> rois) {
        if (rois == null || rois.isEmpty()) return null;
        if (rois.size() == 1) return rois.get(0);

        ImagePlane plane = rois.get(0).getImagePlane();
        List<Geometry> geometries = rois.stream()
                .map(GeometryTools::roiToGeometry)
                .filter(Objects::nonNull)
                .toList();

        if (geometries.isEmpty()) return null;
        if (geometries.size() == 1) {
            return GeometryTools.geometryToROI(geometries.get(0), plane);
        }

        Geometry merged;
        if (geometries.size() > PARALLEL_THRESHOLD) {
            merged = ForkJoinPool.commonPool().invoke(
                    new GeometryUnionTask(geometries, 0, geometries.size()));
        } else {
            merged = mergeGeometriesBatched(geometries, UNION_BATCH_SIZE, false);
        }

        return merged != null ? GeometryTools.geometryToROI(merged, plane) : null;
    }

    /**
     * Merges JTS Geometry objects using hierarchical batched union.
     * <p>
     * In each round, geometries are grouped into batches of {@code batchSize},
     * each batch is unified via {@link UnaryUnionOp}, and the results form the
     * input for the next round. This continues until a single geometry remains.
     *
     * @param geometries the geometries to merge
     * @param batchSize  number of geometries per batch
     * @param parallel   whether to use ForkJoinPool (ignored for small lists)
     * @return the merged geometry, or {@code null} if empty
     */
    public static Geometry mergeGeometriesBatched(List<Geometry> geometries,
                                                   int batchSize,
                                                   boolean parallel) {
        if (geometries == null || geometries.isEmpty()) return null;
        if (geometries.size() == 1) return geometries.get(0);

        if (parallel && geometries.size() > PARALLEL_THRESHOLD) {
            return ForkJoinPool.commonPool().invoke(
                    new GeometryUnionTask(geometries, 0, geometries.size()));
        }

        // Iterative hierarchical batched union
        List<Geometry> current = new ArrayList<>(geometries);
        while (current.size() > 1) {
            List<Geometry> next = new ArrayList<>();
            for (int i = 0; i < current.size(); i += batchSize) {
                int end = Math.min(i + batchSize, current.size());
                List<Geometry> batch = current.subList(i, end);
                Geometry unified = UnaryUnionOp.union(batch);
                if (unified != null) {
                    next.add(unified);
                }
            }
            current = next;
        }
        return current.isEmpty() ? null : current.get(0);
    }

    /**
     * ForkJoinPool task for parallel hierarchical geometry union.
     * <p>
     * Recursively splits the geometry list at the midpoint. When a partition
     * is small enough ({@value UNION_BATCH_SIZE} or fewer), it is unified
     * directly with {@link UnaryUnionOp}. The two halves are then merged
     * with a single {@code union()} call.
     */
    private static class GeometryUnionTask extends RecursiveTask<Geometry> {
        private final List<Geometry> geometries;
        private final int start;
        private final int end;

        GeometryUnionTask(List<Geometry> geometries, int start, int end) {
            this.geometries = geometries;
            this.start = start;
            this.end = end;
        }

        @Override
        protected Geometry compute() {
            int size = end - start;
            if (size <= 0) return null;
            if (size <= UNION_BATCH_SIZE) {
                // Base case: union the batch directly
                return UnaryUnionOp.union(geometries.subList(start, end));
            }

            // Fork-join split at midpoint
            int mid = start + size / 2;
            GeometryUnionTask left = new GeometryUnionTask(geometries, start, mid);
            GeometryUnionTask right = new GeometryUnionTask(geometries, mid, end);
            left.fork();
            Geometry rightResult = right.compute();
            Geometry leftResult = left.join();

            if (leftResult != null && rightResult != null) {
                return leftResult.union(rightResult);
            }
            return leftResult != null ? leftResult : rightResult;
        }
    }

    /**
     * Union-Find (Disjoint Set Union) data structure for efficient connected component labeling.
     */
    private static class UnionFind {
        private final int[] parent;
        private final int[] rank;

        public UnionFind(int size) {
            parent = new int[size];
            rank = new int[size];
            for (int i = 0; i < size; i++) {
                parent[i] = i;
                rank[i] = 0;
            }
        }

        /**
         * Finds the root of the set containing element x with path compression.
         */
        public int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]); // Path compression
            }
            return parent[x];
        }

        /**
         * Unions the sets containing elements x and y using rank.
         */
        public void union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);

            if (rootX == rootY) return;

            // Union by rank
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }
}
