package com.metaldefect

import android.content.SharedPreferences
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import org.json.JSONArray
import org.json.JSONObject

class HistoryActivity : AppCompatActivity() {

    private lateinit var prefs: SharedPreferences

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)

        prefs = getSharedPreferences("MetalDefectPrefs", MODE_PRIVATE)

        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        toolbar.setNavigationOnClickListener { finish() }

        loadHistory()
    }

    private fun loadHistory() {
        val histStr  = prefs.getString("history", "[]") ?: "[]"
        val arr      = JSONArray(histStr)
        val list     = mutableListOf<JSONObject>()
        for (i in 0 until arr.length()) {
            list.add(arr.getJSONObject(i))
        }

        val recycler = findViewById<RecyclerView>(R.id.recycler_history)
        val tvEmpty  = findViewById<TextView>(R.id.tv_empty)
        val tvTotal  = findViewById<TextView>(R.id.tv_summary_total)
        val tvMost   = findViewById<TextView>(R.id.tv_summary_most)
        val btnClear = findViewById<TextView>(R.id.btn_clear)

        tvTotal.text = list.size.toString()

        // Most common defect
        val counts = mutableMapOf<String, Int>()
        list.forEach {
            val c = it.getString("className")
            counts[c] = (counts[c] ?: 0) + 1
        }
        tvMost.text = counts.maxByOrNull { it.value }?.key ?: "--"

        fun refreshUI() {
            if (list.isEmpty()) {
                tvEmpty.visibility  = View.VISIBLE
                recycler.visibility = View.GONE
            } else {
                tvEmpty.visibility  = View.GONE
                recycler.visibility = View.VISIBLE
            }
            tvTotal.text = list.size.toString()
        }

        refreshUI()

        recycler.layoutManager = LinearLayoutManager(this)
        recycler.adapter       = HistoryAdapter(list)

        btnClear.setOnClickListener {
            list.clear()
            prefs.edit().remove("history").apply()
            recycler.adapter?.notifyDataSetChanged()
            tvMost.text = "--"
            refreshUI()
        }
    }

    inner class HistoryAdapter(
        private val items: MutableList<JSONObject>
    ) : RecyclerView.Adapter<HistoryAdapter.VH>() {

        inner class VH(view: View) : RecyclerView.ViewHolder(view) {
            val dot:    View     = view.findViewById(R.id.color_dot)
            val name:   TextView = view.findViewById(R.id.tv_class_name)
            val time:   TextView = view.findViewById(R.id.tv_timestamp)
            val conf:   TextView = view.findViewById(R.id.tv_confidence_item)
        }

        override fun onCreateViewHolder(
            parent: ViewGroup, viewType: Int
        ): VH {
            val v = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_history, parent, false)
            return VH(v)
        }

        override fun getItemCount() = items.size

        override fun onBindViewHolder(holder: VH, pos: Int) {
            val item  = items[pos]
            val color = try {
                item.getString("color")
            } catch (e: Exception) { "#4ECDC4" }
            val conf  = (item.getDouble("confidence") * 100).toInt()

            holder.name.text = item.getString("className")
            holder.time.text = item.getString("timestamp")
            holder.conf.text = "$conf%"
            holder.conf.setTextColor(Color.parseColor(color))

            val dot = GradientDrawable()
            dot.shape = GradientDrawable.OVAL
            dot.setColor(Color.parseColor(color))
            holder.dot.background = dot
        }
    }
}