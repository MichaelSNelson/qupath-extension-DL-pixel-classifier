/**
 * Export Training Data for DL Classification
 *
 * This script exports annotated regions as training data for
 * deep learning pixel classification. Annotations should be
 * classified with PathClass names matching the target classes.
 *
 * Prerequisites:
 * - Image with classified annotations (set PathClass for each)
 * - Annotations can be polygons, rectangles, or line annotations
 *
 * Output:
 * - Image patches (TIFF) in train/images/ and validation/images/
 * - Mask patches (PNG) in train/masks/ and validation/masks/
 * - config.json with class mappings and weights
 */

import qupath.ext.dlclassifier.utilities.AnnotationExtractor
import qupath.lib.io.PathIO
import java.nio.file.Paths

// ============ Configuration ============

// Output directory for training data
def outputDir = Paths.get(System.getProperty("user.home"), "qupath-dl-training", "export_" + System.currentTimeMillis())

// Patch size (should match model tile size, must be divisible by 32)
def patchSize = 512

// Validation split (fraction of data for validation)
def validationSplit = 0.2

// Minimum patches per annotation (for small annotations)
def minPatchesPerAnnotation = 1

// ============ Main Script ============

println "DL Classification Training Data Export"
println "======================================\n"

// Get current image
def imageData = getCurrentImageData()
if (imageData == null) {
    println "ERROR: No image is open"
    return
}

def server = imageData.getServer()
println "Image: " + server.getMetadata().getName()
println "Size: " + server.getWidth() + " x " + server.getHeight()
println "Channels: " + server.nChannels()

// Get classified annotations
def annotations = getAnnotationObjects()
if (annotations.isEmpty()) {
    println "\nERROR: No annotations found"
    println "Please create annotations and assign PathClass to each."
    return
}

// Group by class
def classCounts = [:]
annotations.each { ann ->
    def pathClass = ann.getPathClass()
    def className = pathClass != null ? pathClass.getName() : "Unclassified"
    classCounts[className] = (classCounts[className] ?: 0) + 1
}

println "\nAnnotations by class:"
classCounts.each { className, count ->
    println "  " + className + ": " + count
}

if (classCounts.containsKey("Unclassified")) {
    println "\nWARNING: Some annotations have no PathClass assigned."
    println "These will be skipped. Assign a class to include them."
}

// Filter out unclassified
def classifiedAnnotations = annotations.findAll { it.getPathClass() != null }
if (classifiedAnnotations.size() < 2) {
    println "\nERROR: Need at least 2 classified annotations"
    return
}

// Get unique classes
def classes = classifiedAnnotations.collect { it.getPathClass().getName() }.unique().sort()
println "\nClasses to export: " + classes.join(", ")

// Create output directory
outputDir.toFile().mkdirs()
println "\nOutput directory: " + outputDir

// Export using AnnotationExtractor
println "\nExporting training data..."
println "  Patch size: " + patchSize
println "  Validation split: " + (validationSplit * 100) + "%"

// Note: In production, this would use AnnotationExtractor.export()
// For this example, we just show the configuration

def configMap = [
    "classes": classes.withIndex().collect { name, idx -> [index: idx, name: name] },
    "patch_size": patchSize,
    "validation_split": validationSplit,
    "image_name": server.getMetadata().getName(),
    "num_channels": server.nChannels(),
    "annotation_counts": classCounts
]

// Save config
def configFile = outputDir.resolve("config.json").toFile()
configFile.text = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(configMap))
println "\nSaved config to: " + configFile

println """

Export configuration created.

To complete the export and start training:
1. Use the Training Dialog (Extensions -> DL Pixel Classifier -> Train Classifier)
2. Or run the full export programmatically with AnnotationExtractor

The output will include:
  ${outputDir}/train/images/    - Training image patches
  ${outputDir}/train/masks/     - Training mask patches
  ${outputDir}/validation/images/ - Validation image patches
  ${outputDir}/validation/masks/  - Validation mask patches
  ${outputDir}/config.json      - Class mappings and weights
"""

println "Done."
