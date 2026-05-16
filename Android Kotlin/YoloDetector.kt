package com.metaldefect

import android.content.Context
import android.graphics.*
import ai.onnxruntime.*
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

/**
 * YOLOv8 Object Detector
 * Finds WHERE defects are and draws boxes around them
 *
 * Output format: [1, 10, 8400]
 * → 8400 candidate boxes
 * → Each box: [cx, cy, w, h, class0, class1, ..., class5]
 */
class YoloDetector(context: Context) {

    // ── Config ─────────────────────────────────────────────────
    private val INPUT_SIZE    = 640
    private val CONF_THRESH   = 0.30f   // min confidence to show box
    private val IOU_THRESH    = 0.45f   // overlap threshold for NMS
    private val NUM_CLASSES   = 6
    private val NUM_ANCHORS   = 8400

    private val CLASS_NAMES = listOf(
        "Crazing", "Inclusion", "Patches",
        "Pitted Surface", "Rolled-in Scale", "Scratches"
    )

    private val CLASS_COLORS = listOf(
        Color.parseColor("#FF6B6B"),   // Crazing     - Red
        Color.parseColor("#4ECDC4"),   // Inclusion   - Teal
        Color.parseColor("#45B7D1"),   // Patches     - Blue
        Color.parseColor("#96CEB4"),   // Pitted      - Green
        Color.parseColor("#FFEAA7"),   // Rolled      - Yellow
        Color.parseColor("#DDA0DD"),   // Scratches   - Purple
    )

    // ── Detection Result ───────────────────────────────────────
    data class Detection(
        val bbox:       RectF,    // bounding box in IMAGE coordinates
        val classIdx:   Int,
        val className:  String,
        val confidence: Float,
        val color:      Int,
    )

    // ── ONNX Session ───────────────────────────────────────────
    private val ortEnv:     OrtEnvironment
    private val ortSession: OrtSession

    init {
        ortEnv     = OrtEnvironment.getEnvironment()
        val modelFile = copyAsset(context, "yolo_defect.onnx")
        ortSession = ortEnv.createSession(
            modelFile.absolutePath,
            OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setOptimizationLevel(
                    OrtSession.SessionOptions.OptLevel.ALL_OPT
                )
            }
        )
        android.util.Log.d("YoloDetector",
            "✅ YOLO loaded: ${ortSession.inputNames}")
    }

    private fun copyAsset(context: Context, name: String): File {
        val out = File(context.filesDir, name)
        if (!out.exists()) {
            context.assets.open(name).use { i ->
                FileOutputStream(out).use { o -> i.copyTo(o) }
            }
        }
        return out
    }

    // ── Main Detection Function ────────────────────────────────
    fun detect(bitmap: Bitmap): List<Detection> {
        // Step 1: Preprocess image
        val (inputBuffer, scale, padX, padY) = preprocess(bitmap)

        // Step 2: Run YOLO
        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            inputBuffer,
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )

        val outputs = ortSession.run(
            mapOf(ortSession.inputNames.first() to inputTensor)
        )

        // Step 3: Parse output [1, 10, 8400]
        @Suppress("UNCHECKED_CAST")
        val rawOutput = outputs[0].value as Array<Array<FloatArray>>
        val output    = rawOutput[0] // shape: [10, 8400]

        // Step 4: Decode boxes
        val detections = decodeOutput(
            output, bitmap.width, bitmap.height,
            scale, padX, padY
        )

        // Step 5: Non-Maximum Suppression
        val finalDetections = applyNMS(detections)

        inputTensor.close()
        outputs.close()

        return finalDetections
    }

    // ── Preprocessing ──────────────────────────────────────────
    data class PreprocessResult(
        val buffer: FloatBuffer,
        val scale:  Float,
        val padX:   Float,
        val padY:   Float,
    )

    private fun preprocess(bitmap: Bitmap): PreprocessResult {
        val imgW  = bitmap.width.toFloat()
        val imgH  = bitmap.height.toFloat()

        // Letterbox: scale to fit 640×640 keeping aspect ratio
        val scale = minOf(INPUT_SIZE / imgW, INPUT_SIZE / imgH)
        val newW  = (imgW * scale).toInt()
        val newH  = (imgH * scale).toInt()
        val padX  = (INPUT_SIZE - newW) / 2f
        val padY  = (INPUT_SIZE - newH) / 2f

        // Create 640×640 canvas with gray padding
        val canvas = Bitmap.createBitmap(
            INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888
        )
        val c      = Canvas(canvas)
        c.drawColor(Color.rgb(114, 114, 114))  // gray padding

        // Draw scaled image centered
        val scaledBmp = Bitmap.createScaledBitmap(
            bitmap, newW, newH, true
        )
        c.drawBitmap(scaledBmp, padX, padY, null)
        scaledBmp.recycle()

        // Convert to float RGB [0,1] in CHW format
        // CHW = Channel first (3 × 640 × 640)
        // YOLO expects RGB not BGR
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        canvas.getPixels(pixels, 0, INPUT_SIZE, 0, 0,
            INPUT_SIZE, INPUT_SIZE)
        canvas.recycle()

        val buffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        val size   = INPUT_SIZE * INPUT_SIZE

        for (i in pixels.indices) {
            val p = pixels[i]
            buffer.put(i,          Color.red(p)   / 255f)  // R
            buffer.put(i + size,   Color.green(p) / 255f)  // G
            buffer.put(i + size*2, Color.blue(p)  / 255f)  // B
        }
        buffer.rewind()

        return PreprocessResult(buffer, scale, padX, padY)
    }

    // ── Decode YOLO Output ─────────────────────────────────────
    private fun decodeOutput(
        output:   Array<FloatArray>,  // [10, 8400]
        imgW:     Int,
        imgH:     Int,
        scale:    Float,
        padX:     Float,
        padY:     Float,
    ): List<Detection> {

        val detections = mutableListOf<Detection>()

        // output[0..3] = cx, cy, w, h (640×640 space)
        // output[4..9] = class scores

        for (a in 0 until NUM_ANCHORS) {
            // Get class scores and find best class
            var maxScore = 0f
            var maxClass = 0

            for (c in 0 until NUM_CLASSES) {
                val score = output[4 + c][a]
                if (score > maxScore) {
                    maxScore = score
                    maxClass = c
                }
            }

            // Filter by confidence
            if (maxScore < CONF_THRESH) continue

            // Get box in 640×640 space
            val cx = output[0][a]
            val cy = output[1][a]
            val w  = output[2][a]
            val h  = output[3][a]

            // Convert from center format to corners
            val x1 = cx - w / 2f
            val y1 = cy - h / 2f
            val x2 = cx + w / 2f
            val y2 = cy + h / 2f

            // Remove letterbox padding and scale back to image space
            val imgX1 = ((x1 - padX) / scale).coerceIn(0f, imgW.toFloat())
            val imgY1 = ((y1 - padY) / scale).coerceIn(0f, imgH.toFloat())
            val imgX2 = ((x2 - padX) / scale).coerceIn(0f, imgW.toFloat())
            val imgY2 = ((y2 - padY) / scale).coerceIn(0f, imgH.toFloat())

            // Skip tiny boxes
            if (imgX2 - imgX1 < 5 || imgY2 - imgY1 < 5) continue

            detections.add(Detection(
                bbox       = RectF(imgX1, imgY1, imgX2, imgY2),
                classIdx   = maxClass,
                className  = CLASS_NAMES[maxClass],
                confidence = maxScore,
                color      = CLASS_COLORS[maxClass],
            ))
        }

        return detections
    }

    // ── Non-Maximum Suppression ────────────────────────────────
    /**
     * NMS removes duplicate boxes for the same object.
     *
     * WHY needed?
     * YOLO generates 8400 boxes — many overlap the same defect.
     * NMS keeps only the BEST box per defect.
     *
     * HOW it works:
     * 1. Sort boxes by confidence (highest first)
     * 2. Keep the highest confidence box
     * 3. Remove any box that overlaps it by >45% (IOU threshold)
     * 4. Repeat for remaining boxes
     */
    private fun applyNMS(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        // Sort by confidence descending
        val sorted  = detections.sortedByDescending { it.confidence }
        val kept    = mutableListOf<Detection>()
        val removed = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (removed[i]) continue
            kept.add(sorted[i])

            for (j in i + 1 until sorted.size) {
                if (removed[j]) continue
                // Same class only
                if (sorted[i].classIdx != sorted[j].classIdx) continue

                val iou = calculateIOU(sorted[i].bbox, sorted[j].bbox)
                if (iou > IOU_THRESH) {
                    removed[j] = true
                }
            }
        }

        return kept
    }

    private fun calculateIOU(a: RectF, b: RectF): Float {
        val interX1 = maxOf(a.left,   b.left)
        val interY1 = maxOf(a.top,    b.top)
        val interX2 = minOf(a.right,  b.right)
        val interY2 = minOf(a.bottom, b.bottom)

        val interW  = maxOf(0f, interX2 - interX1)
        val interH  = maxOf(0f, interY2 - interY1)
        val interArea = interW * interH

        val aArea = (a.right - a.left) * (a.bottom - a.top)
        val bArea = (b.right - b.left) * (b.bottom - b.top)

        return interArea / (aArea + bArea - interArea + 1e-6f)
    }

    fun close() {
        ortSession.close()
        ortEnv.close()
    }
}
