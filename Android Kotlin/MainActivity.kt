package com.metaldefect

import android.Manifest
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.content.ContentValues
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.media.AudioManager
import android.media.ToneGenerator
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.provider.MediaStore
import android.util.Size
import android.view.View
import android.view.animation.LinearInterpolator
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    // ── Views ──────────────────────────────────────────────────
    private lateinit var previewView:      PreviewView
    private lateinit var tvClass:          TextView
    private lateinit var tvConfBadge:      TextView
    private lateinit var tvSpeed:          TextView
    private lateinit var scanFrame:        ImageView
    private lateinit var scanLine:         View
    private lateinit var dangerBorder:     View
    private lateinit var tvTotalScans:     TextView
    private lateinit var tvDefectsFound:   TextView
    private lateinit var tvDefectRate:     TextView
    private lateinit var tvFps:            TextView
    private lateinit var tvSoundIcon:      TextView
    private lateinit var tvSaveToast:      TextView
    private lateinit var btnScan:          TextView
    private lateinit var tvScanLabel:      TextView

    // ── Add these new variables ────────────────────────────────────
    private lateinit var yoloDetector:    YoloDetector
    private lateinit var detectionOverlay: DetectionOverlayView
    private var yoloEnabled = true

    // ── AI ─────────────────────────────────────────────────────
    private lateinit var detector:         DefectDetector
    private lateinit var cameraExecutor:   ExecutorService

    // ── State ──────────────────────────────────────────────────
    private var isScanning          = true   // scan on/off toggle
    private var soundEnabled        = true
    private var vibrationEnabled    = true
    private var confidenceThreshold = 0.80f
    private var lastInferenceTime   = 0L
    private val INFERENCE_INTERVAL  = 150L

    // ── Stats ──────────────────────────────────────────────────
    private var totalScans   = 0
    private var defectsFound = 0
    private var frameCount   = 0
    private var fpsTimer     = System.currentTimeMillis()

    // ── Photo ──────────────────────────────────────────────────
    private var currentBitmap: Bitmap? = null
    private var currentResult: DefectDetector.DetectionResult? = null
    private val PICK_IMAGE    = 200
    private val CAMERA_PERM   = 100

    // ── Prefs ──────────────────────────────────────────────────
    private lateinit var prefs: SharedPreferences

    // ── Colors ─────────────────────────────────────────────────
    private val defectColors = mapOf(
        "Crazing"         to "#FF6B6B",
        "Inclusion"       to "#4ECDC4",
        "Patches"         to "#45B7D1",
        "Pitted Surface"  to "#96CEB4",
        "Rolled-in Scale" to "#FFEAA7",
        "Scratches"       to "#DDA0DD"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        prefs = getSharedPreferences("MetalDefectPrefs", MODE_PRIVATE)
        loadSettings()
        bindViews()
        setupButtons()
        startScanAnimation()

        detector       = DefectDetector(this)
        // ADD after: detector = DefectDetector(this)
        yoloDetector     = YoloDetector(this)
        detectionOverlay = findViewById(R.id.detection_overlay)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (hasCameraPermission()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERM
            )
        }
    }

    private fun loadSettings() {
        soundEnabled        = prefs.getBoolean("sound", true)
        vibrationEnabled    = prefs.getBoolean("vibration", true)
        confidenceThreshold = prefs.getFloat("threshold", 0.80f)
    }

    private fun bindViews() {
        previewView    = findViewById(R.id.camera_preview)
        tvClass        = findViewById(R.id.tv_class)
        tvConfBadge    = findViewById(R.id.tv_confidence_badge)
        tvSpeed        = findViewById(R.id.tv_speed)
        scanFrame      = findViewById(R.id.scan_frame)
        scanLine       = findViewById(R.id.scan_line)
        dangerBorder   = findViewById(R.id.danger_border)
        tvTotalScans   = findViewById(R.id.tv_total_scans)
        tvDefectsFound = findViewById(R.id.tv_defects_found)
        tvDefectRate   = findViewById(R.id.tv_defect_rate)
        tvFps          = findViewById(R.id.tv_fps)
        tvSoundIcon    = findViewById(R.id.tv_sound_icon)
        tvSaveToast    = findViewById(R.id.tv_save_toast)
        btnScan        = findViewById(R.id.btn_scan)
        tvScanLabel    = findViewById(R.id.tv_scan_label)
    }

    private fun setupButtons() {
        // Settings
        findViewById<TextView>(R.id.btn_settings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        // History
        findViewById<LinearLayout>(R.id.btn_history).setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }

        // Upload Photo — FIX: shows result properly
        findViewById<LinearLayout>(R.id.btn_upload_photo).setOnClickListener {
            pickImageFromGallery()
        }

        // Save Photo
        findViewById<LinearLayout>(R.id.btn_save_photo).setOnClickListener {
            saveCurrentPhoto()
        }

        // Sound Toggle
        findViewById<LinearLayout>(R.id.btn_sound_toggle).setOnClickListener {
            soundEnabled = !soundEnabled
            tvSoundIcon.text = if (soundEnabled) "🔊" else "🔇"
            prefs.edit().putBoolean("sound", soundEnabled).apply()
        }

        // ── SCAN BUTTON: pause/resume scanning ────────────────
        btnScan.setOnClickListener {
            isScanning = !isScanning

            if (isScanning) {
                btnScan.text     = "⏸️"
                tvScanLabel.text = "Scanning"
                tvScanLabel.setTextColor(Color.parseColor("#4ECDC4"))
                scanLine.visibility = View.VISIBLE
                tvClass.text     = "Point at metal surface"
                tvConfBadge.text = ""
                dangerBorder.visibility = View.INVISIBLE
                scanFrame.setImageResource(R.drawable.scan_frame)
            } else {
                btnScan.text     = "▶️"
                tvScanLabel.text = "Paused"
                tvScanLabel.setTextColor(Color.parseColor("#FF6B6B"))
                scanLine.visibility = View.INVISIBLE
            }
        }
    }

    private fun startScanAnimation() {
        Handler(Looper.getMainLooper()).postDelayed({
            val h = scanFrame.height.toFloat()
            if (h > 0) {
                ObjectAnimator.ofFloat(scanLine, "translationY", 0f, h).apply {
                    duration     = 2000
                    repeatCount  = ValueAnimator.INFINITE
                    repeatMode   = ValueAnimator.REVERSE
                    interpolator = LinearInterpolator()
                    start()
                }
            }
        }, 600)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(
                    ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                )
                .build().also { ia ->
                    ia.setAnalyzer(cameraExecutor) { proxy ->
                        processFrame(proxy)
                    }
                }

            provider.unbindAll()
            provider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(proxy: ImageProxy) {
        if (!isScanning) {
            proxy.close()
            return
        }

        val now = System.currentTimeMillis()
        if (now - lastInferenceTime < INFERENCE_INTERVAL) {
            proxy.close()
            return
        }
        lastInferenceTime = now

        // FPS counter
        frameCount++
        if (now - fpsTimer >= 1000) {
            val fps = frameCount; frameCount = 0; fpsTimer = now
            runOnUiThread { tvFps.text = fps.toString() }
        }

        val bitmap = proxy.toBitmap()
        proxy.close()

        currentBitmap = bitmap

        // ── Run both models ────────────────────────────────────────
        // 1. EfficientNet: WHAT is the defect? (99.72% accurate)
        val classResult = detector.detect(bitmap)
        currentResult   = classResult

        // 2. YOLOv8: WHERE is the defect? (draws boxes)
        val yoloResults = yoloDetector.detect(bitmap)

        runOnUiThread {
            // Update classification result
            updateUI(classResult)

            // Update detection boxes
            if (yoloResults.isNotEmpty()) {
                detectionOverlay.setDetections(
                    yoloResults,
                    bitmap.width,
                    bitmap.height
                )
            } else {
                detectionOverlay.clearDetections()
            }
        }
    }

    private fun updateUI(result: DefectDetector.DetectionResult) {
        val pct    = (result.confidence * 100).toInt()
        val color  = defectColors[result.className] ?: "#4ECDC4"
        val isHigh = result.confidence >= confidenceThreshold

        // Stats
        totalScans++
        if (isHigh && result.status != "UNCERTAIN") defectsFound++
        tvTotalScans.text   = totalScans.toString()
        tvDefectsFound.text = defectsFound.toString()
        val rate = if (totalScans > 0) defectsFound * 100 / totalScans else 0
        tvDefectRate.text   = "$rate%"

        // Result chip
        if (result.status == "UNCERTAIN" || !isHigh) {
            tvClass.text    = "✅ Surface OK"
            tvClass.setTextColor(Color.parseColor("#4ECDC4"))
            tvConfBadge.text = ""
            scanFrame.setImageResource(R.drawable.scan_frame)
            dangerBorder.visibility = View.INVISIBLE
        } else {
            tvClass.text    = "⚠️ ${result.className}"
            tvClass.setTextColor(Color.parseColor(color))
            tvConfBadge.text = "$pct%"
            tvConfBadge.setTextColor(Color.parseColor(color))
            scanFrame.setImageResource(R.drawable.scan_frame_danger)
            showDangerBorder(color)
            if (soundEnabled)     playAlert()
            if (vibrationEnabled) vibrate()
            addToHistory(result)
        }

        tvSpeed.text = "${result.inferenceMs}ms"
    }

    // ── Show result for UPLOADED photo ────────────────────────
    private fun showUploadedResult(
        result: DefectDetector.DetectionResult
    ) {
        val pct    = (result.confidence * 100).toInt()
        val color  = defectColors[result.className] ?: "#4ECDC4"
        val isHigh = result.confidence >= confidenceThreshold

        // Stop live scanning while showing upload result
        isScanning   = false
        btnScan.text = "▶️"
        tvScanLabel.text = "Paused"
        tvScanLabel.setTextColor(Color.parseColor("#FF6B6B"))

        if (!isHigh || result.status == "UNCERTAIN") {
            tvClass.text = "✅ No Defect Detected ($pct%)"
            tvClass.setTextColor(Color.parseColor("#4ECDC4"))
            tvConfBadge.text = ""
            scanFrame.setImageResource(R.drawable.scan_frame)
            dangerBorder.visibility = View.INVISIBLE
        } else {
            tvClass.text = "⚠️ ${result.className} Detected!"
            tvClass.setTextColor(Color.parseColor(color))
            tvConfBadge.text = "$pct% confidence"
            tvConfBadge.setTextColor(Color.parseColor(color))
            scanFrame.setImageResource(R.drawable.scan_frame_danger)
            showDangerBorder(color)
        }

        // Show save toast prompting user
        tvSaveToast.text       = "📸 Tap Save to keep this result"
        tvSaveToast.visibility = View.VISIBLE
        Handler(Looper.getMainLooper()).postDelayed({
            tvSaveToast.visibility = View.GONE
        }, 3000)
    }

    private fun showDangerBorder(colorHex: String) {
        val border = GradientDrawable()
        border.setStroke(10, Color.parseColor(colorHex))
        border.setColor(Color.TRANSPARENT)
        dangerBorder.background  = border
        dangerBorder.visibility  = View.VISIBLE

        // Pulse animation
        dangerBorder.animate()
            .alpha(0.3f).setDuration(400)
            .withEndAction {
                dangerBorder.animate()
                    .alpha(1f).setDuration(400).start()
            }.start()
    }

    private fun playAlert() {
        try {
            val tg = ToneGenerator(AudioManager.STREAM_ALARM, 80)
            tg.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 200)
        } catch (e: Exception) { /* ignore */ }
    }

    private fun vibrate() {
        val v = getSystemService(VIBRATOR_SERVICE) as Vibrator
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            v.vibrate(
                VibrationEffect.createOneShot(
                    200, VibrationEffect.DEFAULT_AMPLITUDE
                )
            )
        } else {
            @Suppress("DEPRECATION")
            v.vibrate(200)
        }
    }

    private fun saveCurrentPhoto() {
        val bitmap = currentBitmap ?: run {
            showToast("⚠️ No image to save yet")
            return
        }
        val result   = currentResult
        val ts       = SimpleDateFormat(
            "yyyyMMdd_HHmmss", Locale.getDefault()
        ).format(Date())
        val name     = "defect_${result?.className ?: "scan"}_$ts.jpg"
        val values   = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH,
                    "Pictures/MetalDefectAI")
            }
        }
        val uri = contentResolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values
        )
        uri?.let {
            val out: OutputStream? = contentResolver.openOutputStream(it)
            out?.use { s -> bitmap.compress(Bitmap.CompressFormat.JPEG, 95, s) }
            showToast("✅ Saved to Gallery!")
        }
    }

    private fun showToast(msg: String) {
        tvSaveToast.text       = msg
        tvSaveToast.visibility = View.VISIBLE
        Handler(Looper.getMainLooper()).postDelayed({
            tvSaveToast.visibility = View.GONE
        }, 2500)
    }

    private fun pickImageFromGallery() {
        val intent = Intent(
            Intent.ACTION_PICK,
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        )
        intent.type = "image/*"
        startActivityForResult(intent, PICK_IMAGE)
    }

    @Suppress("DEPRECATION")
    override fun onActivityResult(
        requestCode: Int, resultCode: Int, data: Intent?
    ) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_IMAGE &&
            resultCode == RESULT_OK &&
            data?.data != null) {

            val uri    = data.data!!
            showToast("🔍 Analyzing image...")

            // Run detection in background
            cameraExecutor.execute {
                try {
                    val bitmap = MediaStore.Images.Media
                        .getBitmap(contentResolver, uri)
                    val result = detector.detect(bitmap)
                    currentBitmap = bitmap
                    currentResult = result

                    // ── FIX: show result on UI thread ─────────
                    runOnUiThread {
                        showUploadedResult(result)
                    }
                } catch (e: Exception) {
                    runOnUiThread {
                        showToast("❌ Could not load image")
                    }
                }
            }
        }
    }

    private fun addToHistory(result: DefectDetector.DetectionResult) {
        val ts  = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        val obj = JSONObject().apply {
            put("className",  result.className)
            put("confidence", result.confidence)
            put("timestamp",  ts)
            put("color", defectColors[result.className] ?: "#FFFFFF")
        }
        val histStr = prefs.getString("history", "[]")!!
        val arr     = JSONArray(histStr)
        val newArr  = JSONArray()
        newArr.put(obj)
        for (i in 0 until minOf(arr.length(), 99)) {
            newArr.put(arr.getJSONObject(i))
        }
        prefs.edit().putString("history", newArr.toString()).apply()
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERM &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.close()
        yoloDetector.close()      // ← ADD THIS
        cameraExecutor.shutdown()
    }
}
