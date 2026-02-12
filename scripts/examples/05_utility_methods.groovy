/**
 * DL Classification Utility Methods Demo
 *
 * This script demonstrates various utility methods available in the
 * DLClassifierScripts API for working with classification results.
 *
 * Available utilities:
 * - Check if annotations have classification measurements
 * - Clear classification measurements for re-processing
 * - Get summary of classification results
 * - List available classifiers
 */

import qupath.ext.dlclassifier.scripting.DLClassifierScripts

// ============ Server Status ============

println "=== Server Status ==="
if (DLClassifierScripts.isServerAvailable()) {
    println "Server: ONLINE"
    println "GPU: " + DLClassifierScripts.getGPUInfo()
} else {
    println "Server: OFFLINE"
    println "Some features may not work without the server."
}

// ============ Available Classifiers ============

println "\n=== Available Classifiers ==="
def classifiers = DLClassifierScripts.listClassifiers()
if (classifiers.isEmpty()) {
    println "(no classifiers found)"
} else {
    classifiers.each { id ->
        try {
            def meta = DLClassifierScripts.loadClassifier(id)
            println "  - " + id
            println "      Name: " + meta.getName()
            println "      Type: " + meta.getModelType()
            println "      Classes: " + meta.getClassNames().join(", ")
        } catch (Exception e) {
            println "  - " + id + " (error loading)"
        }
    }
}

// ============ Current Image Status ============

println "\n=== Current Image Analysis ==="

def imageData = getCurrentImageData()
if (imageData == null) {
    println "No image is open."
    println "Open an image and re-run this script to see annotation analysis."
    return
}

def server = imageData.getServer()
println "Image: " + server.getMetadata().getName()
println "Size: " + server.getWidth() + " x " + server.getHeight()

def annotations = getAnnotationObjects()
println "Annotations: " + annotations.size()

if (annotations.isEmpty()) {
    println "No annotations to analyze."
    return
}

// Check for existing measurements
println "\n=== Classification Status ==="
def hasResults = DLClassifierScripts.hasClassificationMeasurements(annotations)
println "Has DL measurements: " + (hasResults ? "YES" : "NO")

if (hasResults) {
    // Show summary
    println "\n=== Classification Summary ==="
    def summary = DLClassifierScripts.getClassificationSummary()
    if (summary.isEmpty()) {
        println "(no area measurements found)"
    } else {
        summary.each { name, value ->
            println "  " + name + ": " + String.format("%.2f", value)
        }
    }

    // Show detailed measurements
    println "\n=== Detailed Measurements ==="
    annotations.take(3).each { ann ->
        def name = ann.getName() ?: ann.getPathClass()?.getName() ?: "Unnamed"
        println "\n  Annotation: " + name
        ann.getMeasurementList().getMeasurementNames().findAll {
            it.startsWith("DL:")
        }.each { measName ->
            def value = ann.getMeasurementList().get(measName)
            println "    " + measName + ": " + String.format("%.4f", value)
        }
    }
    if (annotations.size() > 3) {
        println "  ... and " + (annotations.size() - 3) + " more annotations"
    }
}

// ============ Clear Measurements Demo ============

println """

=== Clear Measurements (Commented Out) ===
To re-run classification on images that have already been processed,
you can clear existing DL measurements:

  // Clear measurements for current image
  def cleared = DLClassifierScripts.clearCurrentImageMeasurements()
  println "Cleared measurements from \$cleared annotations"

  // Or clear specific annotations
  def selected = getSelectedObjects()
  DLClassifierScripts.clearClassificationMeasurements(selected)

Uncomment and run to clear measurements.
"""

// Uncomment to actually clear measurements:
// def cleared = DLClassifierScripts.clearCurrentImageMeasurements()
// println "Cleared DL measurements from " + cleared + " annotation(s)"

println "\nUtility demo complete."
