/**
 * DL Classification Server Status Check
 *
 * This script checks the status of the DL classification server and
 * displays available classifiers and GPU information.
 *
 * Use this to verify the server is running before classification.
 */

import qupath.ext.dlclassifier.scripting.DLClassifierScripts

println "============================================"
println "    DL Pixel Classifier Server Status"
println "============================================\n"

// Check server health
print "Server status: "
if (DLClassifierScripts.isServerAvailable()) {
    println "ONLINE"
} else {
    println "OFFLINE"
    println "\nThe classification server is not running."
    println "Start it with:"
    println "  cd python_server"
    println "  python -m uvicorn dlclassifier_server.main:app --port 8765"
    return
}

// GPU info
println "\nGPU Information:"
println "  " + DLClassifierScripts.getGPUInfo()

// List classifiers
println "\nAvailable Classifiers:"
def classifiers = DLClassifierScripts.listClassifiers()
if (classifiers.isEmpty()) {
    println "  (none)"
    println "\n  To train a classifier, use: Extensions -> DL Pixel Classifier -> Train Classifier"
} else {
    classifiers.each { id ->
        try {
            def meta = DLClassifierScripts.loadClassifier(id)
            println "  - " + id
            println "      Name: " + meta.getName()
            println "      Type: " + meta.getModelType() + " / " + meta.getBackbone()
            println "      Classes: " + meta.getClassNames().join(", ")
            println "      Input: " + meta.getInputChannels() + "ch, " +
                    meta.getInputWidth() + "x" + meta.getInputHeight()
        } catch (Exception e) {
            println "  - " + id + " (error loading metadata)"
        }
    }
}

// Handler types
println "\nSupported Model Types:"
import qupath.ext.dlclassifier.classifier.ClassifierRegistry
ClassifierRegistry.getAllHandlers().each { handler ->
    println "  - " + handler.getDisplayName() + " (" + handler.getType() + ")"
    println "      " + handler.getDescription().take(80) + "..."
}

println "\n============================================"
println "Server is ready for classification."
println "============================================"
