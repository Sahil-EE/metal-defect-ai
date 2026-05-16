package com.metaldefect

import android.graphics.Bitmap
import android.graphics.Color
import java.nio.FloatBuffer

/**
 * Replicates our Python preprocessing pipeline in Android.
 * Must match EXACTLY what we did during training.
 *
 * Pipeline:
 * 1. Grayscale conversion
 * 2. Bilateral filter (approximated)
 * 3. CLAHE (approximated with histogram equalization)
 * 4. Resize to 224×224
 * 5. Normalize (÷255, then ImageNet stats)
 */
object ImagePreprocessor {

    private const val TARGET_SIZE = 224
    private const val MEAN = 0.485f
    private const val STD  = 0.229f

    /**
     * Main preprocessing function.
     * Input:  Android Bitmap (any size, color)
     * Output: FloatBuffer of shape [1, 1, 224, 224]
     *         ready for ONNX model input
     */
    fun preprocess(bitmap: Bitmap): FloatBuffer {

        // Step 1: Convert to grayscale
        val gray = toGrayscale(bitmap)

        // Step 2: Apply histogram equalization (approximates CLAHE)
        val equalized = histogramEqualize(gray)

        // Step 3: Resize to 224×224
        val resized = Bitmap.createScaledBitmap(
            equalized, TARGET_SIZE, TARGET_SIZE, true
        )

        // Step 4: Convert to float array + normalize
        return bitmapToFloatBuffer(resized)
    }

    private fun toGrayscale(bitmap: Bitmap): Bitmap {
        val width  = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xff
            val g = (pixel shr 8) and 0xff
            val b = pixel and 0xff

            // Standard grayscale formula: 0.299R + 0.587G + 0.114B
            val gray = (0.299f * r + 0.587f * g + 0.114f * b).toInt().coerceIn(0, 255)
            pixels[i] = (0xff shl 24) or (gray shl 16) or (gray shl 8) or gray
        }

        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        result.setPixels(pixels, 0, width, 0, 0, width, height)
        return result
    }

    private fun histogramEqualize(bitmap: Bitmap): Bitmap {
        val width  = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Build histogram
        val hist = IntArray(256)
        for (pixel in pixels) {
            hist[Color.red(pixel)]++
        }

        // Build cumulative distribution
        val cdf     = IntArray(256)
        cdf[0]      = hist[0]
        for (i in 1 until 256) {
            cdf[i] = cdf[i-1] + hist[i]
        }

        // Normalize CDF
        val total   = width * height
        val cdfMin  = cdf.first { it > 0 }
        val lut     = IntArray(256) { i ->
            ((cdf[i] - cdfMin).toFloat() /
                    (total - cdfMin).toFloat() * 255).toInt().coerceIn(0, 255)
        }

        // Apply LUT
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (i in pixels.indices) {
            val v = lut[Color.red(pixels[i])]
            pixels[i] = Color.rgb(v, v, v)
        }
        result.setPixels(pixels, 0, width, 0, 0, width, height)
        return result
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        // Buffer size = 1 × 1 × 224 × 224
        val buffer = FloatBuffer.allocate(TARGET_SIZE * TARGET_SIZE)

        for (y in 0 until TARGET_SIZE) {
            for (x in 0 until TARGET_SIZE) {
                val pixel      = bitmap.getPixel(x, y)
                val grayValue  = Color.red(pixel)           // R=G=B for grayscale

                // Normalize: divide by 255, then subtract mean, divide by std
                val normalized = (grayValue / 255.0f - MEAN) / STD
                buffer.put(normalized)
            }
        }

        buffer.rewind()
        return buffer
    }
}