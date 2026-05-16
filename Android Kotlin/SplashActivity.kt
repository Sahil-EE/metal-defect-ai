package com.metaldefect

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.animation.*
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class SplashActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)

        val tvIcon      = findViewById<TextView>(R.id.tv_logo_icon)
        val tvName      = findViewById<TextView>(R.id.tv_app_name)
        val tvSubtitle  = findViewById<TextView>(R.id.tv_subtitle)
        val tvLoading   = findViewById<TextView>(R.id.tv_loading_text)
        val progressBar = findViewById<ProgressBar>(R.id.progress_loading)

        // Start everything invisible
        tvIcon.alpha     = 0f
        tvName.alpha     = 0f
        tvSubtitle.alpha = 0f
        tvLoading.alpha  = 0f
        progressBar.alpha = 0f
        progressBar.progress = 0

        val handler = Handler(Looper.getMainLooper())

        // Step 1: Icon bounces in (0ms)
        tvIcon.scaleX = 0f
        tvIcon.scaleY = 0f
        tvIcon.animate()
            .scaleX(1f).scaleY(1f).alpha(1f)
            .setDuration(700)
            .setInterpolator(BounceInterpolator())
            .start()

        // Step 2: App name slides up (500ms)
        tvName.translationY = 60f
        handler.postDelayed({
            tvName.animate()
                .translationY(0f).alpha(1f)
                .setDuration(500)
                .setInterpolator(DecelerateInterpolator())
                .start()
        }, 500)

        // Step 3: Subtitle fades in (800ms)
        handler.postDelayed({
            tvSubtitle.animate()
                .alpha(1f).setDuration(400).start()
        }, 800)

        // Step 4: Progress bar appears (1000ms)
        handler.postDelayed({
            progressBar.animate().alpha(1f).setDuration(300).start()
            tvLoading.animate().alpha(1f).setDuration(300).start()
            animateProgress(progressBar, tvLoading, handler)
        }, 1000)
    }

    private fun animateProgress(
        bar: ProgressBar,
        tvText: TextView,
        handler: Handler
    ) {
        data class Step(val progress: Int, val text: String, val delay: Long)

        val steps = listOf(
            Step(15,  "🔍 Loading AI Model...",       0),
            Step(35,  "⚙️  Initializing Engine...",   400),
            Step(55,  "📷 Preparing Camera...",        400),
            Step(75,  "🧠 Loading Neural Network...", 400),
            Step(90,  "✅ Almost Ready...",            400),
            Step(100, "🚀 Let's Go!",                 400),
        )

        var totalDelay = 0L
        steps.forEach { step ->
            totalDelay += step.delay
            handler.postDelayed({
                // Animate progress smoothly
                val animator = android.animation.ObjectAnimator
                    .ofInt(bar, "progress", step.progress)
                animator.duration = 350
                animator.interpolator = DecelerateInterpolator()
                animator.start()

                tvText.text = step.text

                // On last step → launch MainActivity
                if (step.progress == 100) {
                    handler.postDelayed({
                        startActivity(
                            Intent(this, MainActivity::class.java)
                        )
                        overridePendingTransition(
                            android.R.anim.fade_in,
                            android.R.anim.fade_out
                        )
                        finish()
                    }, 600)
                }
            }, totalDelay)
        }
    }
}
