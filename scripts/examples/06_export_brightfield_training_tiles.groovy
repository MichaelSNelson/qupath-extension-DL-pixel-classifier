/**
 * Export RGB Brightfield Training Tiles for DL Pixel Classifier
 *
 * Exports paired image tiles and label masks from annotated RGB brightfield
 * images, producing training data compatible with qupath-extension-DL-pixel-classifier.
 *
 * This is a standalone script that uses only QuPath's built-in APIs -- no
 * extension dependencies are required. It is intended as an intermediate step
 * for users who want to:
 *   - Inspect training patches before sending them to the Python server
 *   - Troubleshoot the Python training pipeline with known-good data
 *   - Export data for training outside of the extension's built-in workflow
 *
 * Prerequisites:
 *   - An open RGB brightfield image in QuPath (within a project)
 *   - Annotations with assigned PathClass names (e.g., "Tumor", "Stroma")
 *   - Both area annotations (polygons, brushes) and line annotations are supported
 *
 * Output structure:
 *   output_dir/
 *     train/
 *       images/        RGB image tiles (.tiff)
 *       masks/         8-bit indexed label masks (.png)
 *     validation/
 *       images/
 *       masks/
 *     config.json      Class mappings, weights, and channel info
 *
 * Mask pixel values:
 *   0, 1, 2, ...  = class indices (alphabetical by class name)
 *   255            = unlabeled / unannotated (ignored by training loss)
 *
 * To run on multiple images:
 *   Use QuPath's "Run for project" (Run -> Run for project) which executes
 *   this script on every image in the project. Each image exports to its own
 *   subdirectory. You can then combine the train/ and validation/ folders
 *   from each export into a single dataset, or train on them individually.
 *
 * @author UW-LOCI
 * @see qupath.ext.dlclassifier.utilities.AnnotationExtractor
 */

import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.images.writers.TileExporter
import qupath.lib.common.GeneralTools
import com.google.gson.GsonBuilder  
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.nio.file.Files
import java.nio.file.StandardCopyOption

// ============================================================
// USER CONFIGURATION - Adjust these settings as needed
// ============================================================

// Tile size in pixels. Must be divisible by 32 for encoder-decoder
// architectures (UNet, DeepLabV3+, etc.). Common values: 256, 512, 1024.
// Larger tiles capture more context but require more GPU memory.
int TILE_SIZE = 512

// Overlap between adjacent tiles in pixels. Overlap helps the model
// see context at tile borders. Set to 0 for no overlap, or a small
// value like 64 for moderate overlap. Must be less than TILE_SIZE / 2.
int OVERLAP = 0

// Fraction of tiles to hold out for validation (0.0 to 1.0).
// 0.2 means 20% validation, 80% training.
double VALIDATION_SPLIT = 0.2

// Line annotation thickness in pixels. When using line or polyline
// annotations (common for pixel classifiers), this controls how many
// pixels around the line are labeled. Larger values create wider
// labeled bands. Set to 0 to use only area annotations.
float LINE_THICKNESS = 5.0f

// Desired pixel size in calibrated units (typically microns).
// Set to 0.0 to export at full resolution.
// Example: 1.0 exports at ~1 um/px, 0.5 at ~0.5 um/px.
double REQUESTED_PIXEL_SIZE = 0.0

// Output directory. Set to null to auto-generate inside the project.
// Example: "C:/training_data/my_export"
def OUTPUT_DIR = null

// Image format for tiles. TIFF preserves calibration metadata and is
// lossless. PNG is also lossless but slower for large tiles.
String IMAGE_EXTENSION = '.tiff'

// Random seed for reproducible train/validation splits.
long RANDOM_SEED = 42

// Minimum percentage of labeled pixels for a tile to be exported.
// Tiles with fewer labeled pixels than this threshold (relative to
// total pixels) are skipped. Helps filter out nearly-empty tiles.
// Set to 0.0 to include all tiles that overlap any annotation.
double MIN_LABELED_PERCENT = 0.0

// ============================================================
// SCRIPT LOGIC - No changes needed below this line
// ============================================================

// --- Validate image ---

def imageData = getCurrentImageData()
if (imageData == null) {
    println "ERROR: No image is open. Please open an image first."
    return
}

def server = imageData.getServer()
def imageName = GeneralTools.stripExtension(server.getMetadata().getName())

if (!imageData.isBrightfield()) {
    println "WARNING: Image does not appear to be brightfield."
    println "This script is designed for RGB brightfield images."
    println "Continuing anyway -- results may not be optimal.\n"
}

println "========================================================"
println "DL Pixel Classifier - Brightfield Training Tile Export"
println "========================================================\n"
println "Image: " + imageName
println "Size:  " + server.getWidth() + " x " + server.getHeight() + " px"
println "Channels: " + server.nChannels()

def pixelCal = server.getPixelCalibration()
if (pixelCal.hasPixelSizeMicrons()) {
    def pxSize = pixelCal.getAveragedPixelSizeMicrons()
    println "Pixel size: " + String.format("%.4f", pxSize) + " um/px"
} else {
    println "Pixel size: uncalibrated"
}

// --- Validate annotations ---

def allAnnotations = getAnnotationObjects()
def annotations = allAnnotations.findAll { it.getPathClass() != null }

if (allAnnotations.isEmpty()) {
    println "\nERROR: No annotations found."
    println "Draw annotations on the image and assign a class to each:"
    println "  - Right-click annotation -> Set class"
    println "  - Or use the Annotations tab"
    return
}

int unclassified = allAnnotations.size() - annotations.size()
if (unclassified > 0) {
    println "\nWARNING: " + unclassified + " unclassified annotation(s) will be skipped."
    println "Assign a PathClass to include them in the export."
}

if (annotations.isEmpty()) {
    println "\nERROR: No classified annotations found."
    return
}

// Discover classes (sorted alphabetically for consistent indexing)
def classNames = annotations.collect { it.getPathClass().getName() }.unique().sort()

if (classNames.size() < 2) {
    println "\nERROR: Need at least 2 different classes for training."
    println "Found only: " + classNames
    println "Add annotations of a second class and try again."
    return
}

// Report annotation summary
println "\nAnnotations by class:"
def classCounts = annotations.groupBy { it.getPathClass().getName() }
classNames.eachWithIndex { name, idx ->
    def count = classCounts[name]?.size() ?: 0
    def lineCount = classCounts[name]?.count { it.getROI().isLine() } ?: 0
    def areaCount = count - lineCount
    println "  [" + idx + "] " + name + ": " + count + " (" + areaCount + " area, " + lineCount + " line)"
}

// --- Calculate downsample ---

double downsample = 1.0
if (REQUESTED_PIXEL_SIZE > 0) {
    if (pixelCal.hasPixelSizeMicrons()) {
        downsample = REQUESTED_PIXEL_SIZE / pixelCal.getAveragedPixelSizeMicrons()
        println "\nResolution: " + REQUESTED_PIXEL_SIZE + " um/px (downsample = " +
                String.format("%.2f", downsample) + ")"
    } else {
        println "\nWARNING: Image has no pixel size calibration."
        println "Using full resolution (downsample = 1.0)."
    }
} else {
    println "\nResolution: full resolution (downsample = 1.0)"
}

// --- Determine output directory ---

def pathOutput
if (OUTPUT_DIR != null) {
    pathOutput = OUTPUT_DIR
} else if (PROJECT_BASE_DIR != null) {
    pathOutput = buildFilePath(PROJECT_BASE_DIR, 'dl_training', imageName)
} else {
    pathOutput = buildFilePath(
            System.getProperty("user.home"), 'QuPath_DL_Training', imageName)
}

println "Output: " + pathOutput
println "Tile size: " + TILE_SIZE + " px, Overlap: " + OVERLAP + " px"
println "Validation split: " + ((int)(VALIDATION_SPLIT * 100)) + "%"

// --- Build LabeledImageServer ---

println "\nBuilding label server..."

def labelBuilder = new LabeledImageServer.Builder(imageData)
        .backgroundLabel(255)           // Unlabeled pixels -> 255
        .downsample(downsample)
        .multichannelOutput(false)      // Single-channel indexed output

if (LINE_THICKNESS > 0) {
    labelBuilder.lineThickness(LINE_THICKNESS)
}

// Map each class name to its index (0, 1, 2, ...)
classNames.eachWithIndex { name, idx ->
    labelBuilder.addLabel(name, idx)
}

def labelServer = labelBuilder.build()

// --- Export tiles to staging directory ---

def stagingDir = buildFilePath(pathOutput, '_staging')
mkdirs(stagingDir)

println "Exporting tiles to staging area..."

new TileExporter(imageData)
        .downsample(downsample)
        .imageExtension(IMAGE_EXTENSION)
        .labeledImageExtension('.png')
        .tileSize(TILE_SIZE)
        .overlap(OVERLAP)
        .labeledServer(labelServer)
        .imageSubDir('images')
        .labeledImageSubDir('masks')
        .annotatedTilesOnly(true)
        .exportJson(true)
        .writeTiles(stagingDir)

// --- Collect exported files ---

def stagingImages = new File(stagingDir, 'images')
def stagingMasks = new File(stagingDir, 'masks')

def imageFiles = stagingImages.listFiles()?.findAll {
    it.name.endsWith('.tiff') || it.name.endsWith('.tif') || it.name.endsWith('.png')
}?.sort { it.name }

if (imageFiles == null || imageFiles.isEmpty()) {
    println "\nERROR: No tiles were exported."
    println "This can happen if:"
    println "  - Annotations are outside the image bounds"
    println "  - Annotations are smaller than the tile size"
    println "  - The downsample factor makes annotations too small"
    // Clean up
    new File(stagingDir).deleteDir()
    return
}

println "Exported " + imageFiles.size() + " tile(s) to staging area."

// --- Create final directory structure ---

def trainImagesDir = new File(pathOutput, 'train/images')
def trainMasksDir = new File(pathOutput, 'train/masks')
def valImagesDir = new File(pathOutput, 'validation/images')
def valMasksDir = new File(pathOutput, 'validation/masks')

trainImagesDir.mkdirs()
trainMasksDir.mkdirs()
valImagesDir.mkdirs()
valMasksDir.mkdirs()

// --- Split into train / validation ---

println "Splitting into train/validation..."

def random = new Random(RANDOM_SEED)
def shuffled = new ArrayList(imageFiles)
Collections.shuffle(shuffled, random)

int valTarget = Math.max(1, (int)(shuffled.size() * VALIDATION_SPLIT))
int trainTarget = shuffled.size() - valTarget

int trainIdx = 0
int valIdx = 0
int skipped = 0

// Accumulators for class pixel statistics
long[] classPixelCounts = new long[classNames.size()]
long totalLabeledPixels = 0
int totalPixelsPerTile = TILE_SIZE * TILE_SIZE
double minLabeledThreshold = MIN_LABELED_PERCENT / 100.0

for (int i = 0; i < shuffled.size(); i++) {
    def imgFile = shuffled[i]

    // Find corresponding mask (same stem, .png extension)
    def stem = imgFile.name.substring(0, imgFile.name.lastIndexOf('.'))
    def maskFile = new File(stagingMasks, stem + '.png')

    if (!maskFile.exists()) {
        println "  WARNING: No mask for " + imgFile.name + ", skipping"
        skipped++
        continue
    }

    // Read mask and count labeled pixels
    BufferedImage mask = ImageIO.read(maskFile)
    int maskW = mask.getWidth()
    int maskH = mask.getHeight()
    int[] pixels = new int[maskW * maskH]
    mask.getRaster().getPixels(0, 0, maskW, maskH, pixels)

    long patchLabeledCount = 0
    long[] patchClassCounts = new long[classNames.size()]

    for (int p = 0; p < pixels.length; p++) {
        int val = pixels[p]
        if (val != 255 && val >= 0 && val < classNames.size()) {
            patchClassCounts[val]++
            patchLabeledCount++
        }
    }

    // Skip tiles below the labeled pixel threshold
    if (minLabeledThreshold > 0) {
        double labeledFraction = (double) patchLabeledCount / pixels.length
        if (labeledFraction < minLabeledThreshold) {
            skipped++
            continue
        }
    }

    // Determine train vs validation
    boolean isValidation = (i >= trainTarget)

    def targetImgDir = isValidation ? valImagesDir : trainImagesDir
    def targetMaskDir = isValidation ? valMasksDir : trainMasksDir
    int patchIdx = isValidation ? valIdx : trainIdx

    // Copy with sequential naming
    def newImgName = String.format("patch_%04d" + IMAGE_EXTENSION, patchIdx)
    def newMaskName = String.format("patch_%04d.png", patchIdx)

    Files.copy(imgFile.toPath(), new File(targetImgDir, newImgName).toPath(),
            StandardCopyOption.REPLACE_EXISTING)
    Files.copy(maskFile.toPath(), new File(targetMaskDir, newMaskName).toPath(),
            StandardCopyOption.REPLACE_EXISTING)

    // Accumulate statistics
    for (int c = 0; c < classNames.size(); c++) {
        classPixelCounts[c] += patchClassCounts[c]
    }
    totalLabeledPixels += patchLabeledCount

    if (isValidation) valIdx++
    else trainIdx++
}

println "  Train:      " + trainIdx + " tiles"
println "  Validation: " + valIdx + " tiles"
if (skipped > 0) {
    println "  Skipped:    " + skipped + " tiles (no mask or below threshold)"
}

if (trainIdx == 0 || valIdx == 0) {
    println "\nWARNING: One split has 0 tiles. Consider:"
    println "  - Adding more annotations"
    println "  - Reducing TILE_SIZE"
    println "  - Adjusting VALIDATION_SPLIT"
}

// --- Clean up staging ---

new File(stagingDir).deleteDir()

// --- Compute class weights ---

println "\nClass distribution:"

def classWeights = []
classNames.eachWithIndex { name, idx ->
    double weight = 1.0
    if (classPixelCounts[idx] > 0 && totalLabeledPixels > 0) {
        weight = (double) totalLabeledPixels / (classNames.size() * classPixelCounts[idx])
    }
    classWeights.add(weight)

    def pctStr = totalLabeledPixels > 0 ?
            String.format("%.1f", 100.0 * classPixelCounts[idx] / totalLabeledPixels) : "0.0"
    println "  [" + idx + "] " + name + ": " + classPixelCounts[idx] + " px (" + pctStr +
            "%), weight=" + String.format("%.4f", weight)
}

// --- Generate config.json ---

def classes = classNames.withIndex().collect { name, idx ->
    [index: idx, name: name, pixel_count: classPixelCounts[idx]]
}

def config = [
    patch_size       : TILE_SIZE,
    overlap          : OVERLAP,
    downsample       : downsample,
    unlabeled_index  : 255,
    line_stroke_width: (int) LINE_THICKNESS,
    total_labeled_pixels: totalLabeledPixels,
    classes          : classes,
    class_weights    : classWeights.collect { Math.round(it * 1000000.0d) / 1000000.0d },
    channel_config   : [
        num_channels : 3,
        channel_names: ["Red", "Green", "Blue"],
        bit_depth    : 8,
        normalization: [
            strategy      : "percentile_99",
            per_channel   : false,
            clip_percentile: 99.0
        ]
    ],
    metadata: [
        source_image    : imageName,
        image_width     : server.getWidth(),
        image_height    : server.getHeight(),
        pixel_size_um   : pixelCal.hasPixelSizeMicrons() ?
                pixelCal.getAveragedPixelSizeMicrons() : -1,
        train_count     : trainIdx,
        validation_count: valIdx,
        annotation_count: annotations.size(),
        random_seed     : RANDOM_SEED,
        export_date     : new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date())
    ]
]

def configFile = new File(pathOutput, 'config.json')
 configFile.text = new
  GsonBuilder().setPrettyPrinting().create().toJson(config)

// --- Summary ---

println "\n========================================================"
println "Export complete!"
println "========================================================\n"
println "Output: " + pathOutput
println "  train/images/       " + trainIdx + " tiles (" + IMAGE_EXTENSION + ")"
println "  train/masks/        " + trainIdx + " masks (.png)"
println "  validation/images/  " + valIdx + " tiles"
println "  validation/masks/   " + valIdx + " masks"
println "  config.json         Class mappings and weights"

println "\nMask legend:"
classNames.eachWithIndex { name, idx ->
    println "  " + idx + " = " + name
}
println "  255 = Unlabeled (ignored during training)"

println "\nNext steps:"
println "  1. Review tiles in train/images/ and corresponding masks/"
println "     (open masks in FIJI -- pixel values are small integers,"
println "      so they will appear black; use Image -> Adjust -> Brightness)"
println "  2. Use Extensions -> DL Pixel Classifier -> Train Classifier"
println "     and point 'Data Path' to:"
println "     " + pathOutput
println "  3. Or train directly from the command line:"
println "     python -m dlclassifier_server --train --data-path \"" + pathOutput + "\""

println "\nDone."
