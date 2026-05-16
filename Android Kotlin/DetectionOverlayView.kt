package com.metaldefect

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * Custom View that draws YOLO detection boxes
 * overlaid on the camera preview.
 *
 * Usage: Add to XML layout on top of PreviewView
 * Call setDetections() to update boxes
 */
class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detections: List<YoloDetector.Detection> = emptyList()
    private var imageWidth  = 1
    private var imageHeight = 1

    // ── Paint styles ───────────────────────────────────────────
    private val boxPaint = Paint().apply {
        style     = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val fillPaint = Paint().apply {
        style   = Paint.Style.FILL
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color       = Color.WHITE
        textSize    = 38f
        typeface    = Typeface.DEFAULT_BOLD
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        style   = Paint.Style.FILL
        isAntiAlias = true
    }

    private val cornerPaint = Paint().apply {
        style     = Paint.Style.STROKE
        strokeWidth = 8f
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }

    // ── Update detections ──────────────────────────────────────
    fun setDetections(
        dets:    List<YoloDetector.Detection>,
        imgW:    Int,
        imgH:    Int,
    ) {
        detections  = dets
        imageWidth  = imgW
        imageHeight = imgH
        invalidate()  // triggers redraw
    }

    fun clearDetections() {
        detections = emptyList()
        invalidate()
    }

    // ── Draw ───────────────────────────────────────────────────
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return

        val scaleX = width.toFloat()  / imageWidth
        val scaleY = height.toFloat() / imageHeight

        for (det in detections) {
            drawDetection(canvas, det, scaleX, scaleY)
        }
    }

    private fun drawDetection(
        canvas: Canvas,
        det:    YoloDetector.Detection,
        scaleX: Float,
        scaleY: Float,
    ) {
        // Scale bbox to view coordinates
        val left   = det.bbox.left   * scaleX
        val top    = det.bbox.top    * scaleY
        val right  = det.bbox.right  * scaleX
        val bottom = det.bbox.bottom * scaleY
        val rect   = RectF(left, top, right, bottom)

        val color  = det.color
        val alpha  = 200  // slight transparency

        // ── 1. Filled background (very transparent) ───────────
        fillPaint.color = color
        fillPaint.alpha = 30
        canvas.drawRoundRect(rect, 8f, 8f, fillPaint)

        // ── 2. Box border ─────────────────────────────────────
        boxPaint.color = color
        boxPaint.alpha = alpha
        canvas.drawRoundRect(rect, 8f, 8f, boxPaint)

        // ── 3. Corner accents (like scan frame style) ─────────
        val cornerLen = minOf(
            (rect.width()  * 0.2f).coerceAtLeast(20f),
            (rect.height() * 0.2f).coerceAtLeast(20f),
        )
        cornerPaint.color = color
        cornerPaint.alpha = 255

        // Top-left
        canvas.drawLine(left, top, left + cornerLen, top, cornerPaint)
        canvas.drawLine(left, top, left, top + cornerLen, cornerPaint)

        // Top-right
        canvas.drawLine(right, top, right - cornerLen, top, cornerPaint)
        canvas.drawLine(right, top, right, top + cornerLen, cornerPaint)

        // Bottom-left
        canvas.drawLine(left, bottom, left + cornerLen, bottom, cornerPaint)
        canvas.drawLine(left, bottom, left, bottom - cornerLen, cornerPaint)

        // Bottom-right
        canvas.drawLine(right, bottom, right - cornerLen, bottom, cornerPaint)
        canvas.drawLine(right, bottom, right, bottom - cornerLen, cornerPaint)

        // ── 4. Label background ───────────────────────────────
        val label    = "${det.className} ${(det.confidence*100).toInt()}%"
        val textW    = textPaint.measureText(label)
        val textH    = textPaint.textSize
        val padding  = 12f
        val labelTop = if (top - textH - padding * 2 > 0) {
            top - textH - padding * 2  // above box
        } else {
            top + 4f                   // inside box if no room
        }

        val labelRect = RectF(
            left,
            labelTop,
            left + textW + padding * 2,
            labelTop + textH + padding
        )

        textBgPaint.color = color
        textBgPaint.alpha = 230
        canvas.drawRoundRect(labelRect, 8f, 8f, textBgPaint)

        // ── 5. Label text ─────────────────────────────────────
        canvas.drawText(
            label,
            left + padding,
            labelTop + textH,
            textPaint
        )
    }
}
