# 🏭 Metal Surface Defect Detection AI

![Accuracy](https://img.shields.io/badge/Accuracy-99.72%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Android](https://img.shields.io/badge/Android-Kotlin-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-72.9%25_mAP-orange)

> Real-time AI system that detects metal surface defects 
> using deep learning — deployed as an Android app

---

## 🎯 What It Does

Automatically detects **6 types of metal surface defects** 
in real-time using phone camera — replacing manual 
human inspection in factories.

| Defect Type | Description |
|-------------|-------------|
| Crazing | Network of fine surface cracks |
| Inclusion | Foreign material embedded in metal |
| Patches | Surface oxidation patches |
| Pitted Surface | Small holes on surface |
| Rolled-in Scale | Surface contamination |
| Scratches | Linear surface damage |

---

## 🏆 Results

### Classification Model (EfficientNet-B0)
| Metric | Score |
|--------|-------|
| Overall Accuracy | **99.72%** |
| Total Errors | 1 out of 360 images |
| Training Time | 20 minutes on GPU |

### Per-Class Performance
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Crazing | 100% | 100% | 100% |
| Inclusion | 100% | 98.3% | 99.2% |
| Patches | 100% | 100% | 100% |
| Pitted Surface | 98.4% | 100% | 99.2% |
| Rolled-in Scale | 100% | 100% | 100% |
| Scratches | 100% | 100% | 100% |

### Detection Model (YOLOv8n)
| Metric | Score |
|--------|-------|
| mAP50 | **72.9%** |
| Speed | 93.8ms per frame |
| Model Size | 11.7 MB |

---

## 📱 App Features

- ✅ Real-time camera defect scanning
- ✅ YOLO bounding boxes around defects
- ✅ Sound + vibration alerts
- ✅ Detection history log
- ✅ Save defect photos to gallery
- ✅ Upload photos for analysis
- ✅ Live statistics dashboard
- ✅ Works 100% OFFLINE
- ✅ Animated splash screen
- ✅ Settings customization

---

## 🧠 Model Architecture
| Model Size | 11.7 MB |

---

## 🧠 Model Architecture
