package com.metaldefect

import android.content.SharedPreferences
import android.os.Bundle
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar

class SettingsActivity : AppCompatActivity() {

    private lateinit var prefs: SharedPreferences

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        prefs = getSharedPreferences("MetalDefectPrefs", MODE_PRIVATE)

        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        toolbar.setNavigationOnClickListener { finish() }

        // Load settings
        val switchSound     = findViewById<Switch>(R.id.switch_sound)
        val switchVibration = findViewById<Switch>(R.id.switch_vibration)
        val seekThreshold   = findViewById<SeekBar>(R.id.seekbar_threshold)
        val tvThreshold     = findViewById<TextView>(R.id.tv_threshold_value)

        switchSound.isChecked     = prefs.getBoolean("sound", true)
        switchVibration.isChecked = prefs.getBoolean("vibration", true)
        val currentThreshold      = (prefs.getFloat("threshold", 0.80f) * 100).toInt()
        seekThreshold.progress    = currentThreshold
        tvThreshold.text          = "$currentThreshold%"

        // Save on change
        switchSound.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("sound", checked).apply()
        }

        switchVibration.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("vibration", checked).apply()
        }

        seekThreshold.setOnSeekBarChangeListener(
            object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(
                    sb: SeekBar?, progress: Int, fromUser: Boolean
                ) {
                    tvThreshold.text = "$progress%"
                    prefs.edit()
                        .putFloat("threshold", progress / 100f)
                        .apply()
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            }
        )
    }
}