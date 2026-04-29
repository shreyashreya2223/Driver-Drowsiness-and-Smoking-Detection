# Driver-Drowsiness-and-Smoking-Detection
# Driver Monitoring System using YOLOv26n

## Project Overview

This project is an AI-based Driver Monitoring System developed to improve road safety by detecting unsafe driver behavior in real time.

The system focuses on two major problems:

1. Driver Drowsiness Detection
2. Driver Smoking Detection

Both modules are built using **YOLOv26 Nano (YOLOv26n)** for real-time object detection through a webcam.

If the system detects that the driver is drowsy or smoking while driving, it immediately triggers an alarm and displays a warning on the screen.

This project helps reduce road accidents caused by fatigue and distracted driving.

---

## Problem Statement

Many road accidents happen because drivers become sleepy during long drives or engage in unsafe habits like smoking while driving.

Traditional monitoring systems are expensive and not easily accessible.

This project provides a low-cost AI-based solution using computer vision and deep learning to monitor the driver continuously and generate alerts whenever dangerous behavior is detected.

---

## Objectives

* Detect driver drowsiness in real time
* Detect driver smoking behavior in real time
* Trigger an alarm when unsafe behavior is detected
* Improve road safety using AI-based monitoring
* Build a practical and deployable smart driver assistance system

---

## Technologies Used

### Programming Language

* Python

### Libraries and Frameworks

* OpenCV
* Ultralytics YOLO
* YOLOv26n
* NumPy
* Pygame
* TensorFlow (used in earlier version)
* MediaPipe (used in earlier version)

### Development Tools

* VS Code
* Kaggle Notebook
* Google Colab
* GitHub

---

## Model Details

## 1. Driver Drowsiness Detection

### Model Used

YOLOv26 Nano (YOLOv26n)

### Dataset Classes

* microsleep
* neutral
* yawning

### Working

The webcam captures live video frames.

The YOLOv26n model detects:

* Microsleep
* Yawning

If either is detected continuously, the system marks the driver as drowsy and triggers the alarm.

---

## 2. Driver Smoking Detection

### Model Used

YOLOv26 Nano (YOLOv26n)

### Dataset Classes

* smoking
* drinking

### Working

The webcam captures live driver activity.

The YOLOv26n model detects smoking behavior.

If smoking is detected, the system immediately triggers the warning and alarm.

---

## System Workflow

1. Webcam starts capturing live video
2. Drowsiness model checks for microsleep and yawning
3. Smoking model checks for smoking behavior
4. If dangerous activity is detected:

   * Warning box appears
   * Alarm starts playing
5. If driver returns to normal state:

   * Alarm stops
   * System shows ACTIVE status

---

## Project Structure

```text
Driver_DSD/
│
├── drowsiness_model/
│   ├── drowsiness_model.pt
│   └── alarm.mp3
│
├── smoke_detection/
│   └── smoking_model.pt
│
├── main_app/
│   └── driver_monitor.py
│
├── alerts/
├── test_images/
├── venv311/
└── README.md
```

---

## Final Integrated File

### Main File

`driver_monitor.py`

This file handles:

* Camera access
* YOLO model loading
* Drowsiness detection
* Smoking detection
* Alarm triggering
* Final monitoring system

---

## Training Process

### Platform Used

Kaggle Notebook

### Steps Performed

1. Added dataset to Kaggle
2. Installed Ultralytics
3. Loaded YOLOv26n pretrained weights
4. Trained model using custom datasets
5. Saved final trained model (`best.pt`)
6. Downloaded and integrated into local project

---

## Performance Results

### Smoking Detection Model

* Precision: Good
* Recall: Good
* mAP50: Strong project-level performance

### Drowsiness Detection Model

* Precision: ~0.95
* Recall: ~0.93
* mAP50: ~0.98
* mAP50-95: ~0.83

These results show strong real-time detection performance.

---

## Features

* Real-time webcam monitoring
* Dual safety detection system
* Alarm-based alert mechanism
* YOLOv26n object detection
* Live warning display
* Easy deployment
* Scalable for future upgrades

---

## Future Enhancements

* Mobile app integration
* Cloud monitoring dashboard
* GPS accident alert system
* Face recognition for driver identity
* Alcohol detection integration
* Seatbelt detection
* Phone usage detection
* Emergency auto-alert to family/police

---

## Conclusion

The Driver Monitoring System using YOLOv26n successfully detects driver drowsiness and smoking behavior in real time.

It provides an intelligent safety mechanism that can help reduce accidents caused by fatigue and distracted driving.

This project demonstrates how Artificial Intelligence and Computer Vision can be used to solve real-world transportation safety problems.

It is practical, scalable, and suitable for smart vehicle safety applications.

---

## Developed By

Shreya

B.Tech Computer Science Engineering

IILM University

---

## Guided By

Faculty Mentor: Saurabhi Purwar

---

## Project Type

Major Project / Final Year Project / Research Project
