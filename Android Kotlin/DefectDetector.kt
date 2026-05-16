package com.metaldefect

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.*
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

class DefectDetector(context: Context) {

    private val classNames = listOf(
        "Crazing",
        "Inclusion",
        "Patches",
        "Pitted Surface",
        "Rolled-in Scale",
        "Scratches"
    )

    private val classColors = mapOf(
        "Crazing"          to "#FF6B6B",
        "Inclusion"        to "#4ECDC4",
        "Patches"          to "#45B7D1",
        "Pitted Surface"   to "#96CEB4",
        "Rolled-in Scale"  to "#FFEAA7",
        "Scratches"        to "#DDA0DD"
    )

    private val HIGH_CONFIDENCE   = 0.95f
    private val MEDIUM_CONFIDENCE = 0.80f

    private val ortEnvironment: OrtEnvironment
    private val ortSession: OrtSession

    init {
        ortEnvironment = OrtEnvironment.getEnvironment()

        val modelFile = copyAssetToInternalStorage(
            context,
            "metal_defect_single2.onnx"
        )

        ortSession = ortEnvironment.createSession(
            modelFile.absolutePath,
            OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(
                    OrtSession.SessionOptions.OptLevel.ALL_OPT
                )
            }
        )

        android.util.Log.d("DefectDetector",
            "Model loaded: ${ortSession.inputNames}")
    }

    private fun copyAssetToInternalStorage(
        context: Context,
        assetName: String
    ): File {
        val outFile = File(context.filesDir, assetName)

        if (!outFile.exists()) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
        }

        return outFile
    }

    data class DetectionResult(
        val classIdx:    Int,
        val className:   String,
        val confidence:  Float,
        val allProbs:    FloatArray,
        val status:      String,
        val color:       String,
        val inferenceMs: Long
    )

    fun detect(bitmap: Bitmap): DetectionResult {
        val startTime = System.currentTimeMillis()

        val inputBuffer = ImagePreprocessor.preprocess(bitmap)

        val inputTensor = OnnxTensor.createTensor(
            ortEnvironment,
            inputBuffer,
            longArrayOf(1, 1, 224, 224)
        )

        val outputs = ortSession.run(
            mapOf(ortSession.inputNames.first() to inputTensor)
        )

        @Suppress("UNCHECKED_CAST")
        val probArray = (outputs[0].value as Array<FloatArray>)[0]

        val topIdx  = probArray.indices.maxByOrNull { probArray[it] }!!
        val topProb = probArray[topIdx]

        val status = when {
            topProb >= HIGH_CONFIDENCE   -> "HIGH"
            topProb >= MEDIUM_CONFIDENCE -> "MEDIUM"
            topProb >= 0.60f             -> "LOW"
            else                         -> "UNCERTAIN"
        }

        val inferenceMs = System.currentTimeMillis() - startTime

        inputTensor.close()
        outputs.close()

        return DetectionResult(
            classIdx    = topIdx,
            className   = classNames[topIdx],
            confidence  = topProb,
            allProbs    = probArray,
            status      = status,
            color       = classColors[classNames[topIdx]] ?: "#FFFFFF",
            inferenceMs = inferenceMs
        )
    }

    fun close() {
        ortSession.close()
        ortEnvironment.close()
    }
}
