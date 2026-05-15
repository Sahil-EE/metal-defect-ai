# metal-defect-detection-ai
Real-time metal surface defect detection using EfficientNet-B0 (99.72% accuracy) + YOLOv8 deployed on Android
# 🏭 Metal Surface Defect Detection AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)
![Android](https://img.shields.io/badge/Android-Kotlin-green)
![Accuracy](https://img.shields.io/badge/Accuracy-99.72%25-brightgreen)
![YOLOv8](https://img.shields.io/badge/YOLOv8-72.9%25_mAP-orange)

**Real-time industrial metal surface defect detection 
using deep learning, deployed as an Android application**

</div>

---

## 📱 App Demo

<div align="center">
<img src="results/app_screenshots/screenshot1.png" width="250"/>
<img src="results/app_screenshots/screenshot2.png" width="250"/>
</div>

---

## 🎯 What It Does

This system automatically detects and classifies 
**6 types of metal surface defects** in real-time 
using the phone camera — replacing manual human 
inspection in factories.

| Defect Type | Description |
|-------------|-------------|
| 🔴 Crazing | Network of fine surface cracks |
| 🟢 Inclusion | Foreign material embedded in metal |
| 🔵 Patches | Surface oxidation/rust patches |
| 🟡 Pitted Surface | Small holes/pits on surface |
| 🟣 Rolled-in Scale | Surface contamination |
| 🩷 Scratches | Linear surface damage |

---

## 🏆 Results

### Classification (EfficientNet-B0)
| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **99.72%** |
| Errors | 1 out of 360 images |
| Training Time | 20 minutes (T4 GPU) |

### Per-Class Performance
| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Crazing | 100.0% | 100.0% | 100.0% |
| Inclusion | 100.0% | 98.3% | 99.2% |
| Patches | 100.0% | 100.0% | 100.0% |
| Pitted Surface | 98.4% | 100.0% | 99.2% |
| Rolled-in Scale | 100.0% | 100.0% | 100.0% |
| Scratches | 100.0% | 100.0% | 100.0% |

### Detection (YOLOv8n)
| Metric | Score |
|--------|-------|
| **mAP50** | **72.9%** |
| Inference Speed | 93.8ms/frame |
| Model Size | 11.7 MB |

---

## 🧠 Model Architecture
