# 🎯 Head Pose Dataset Collection & CNN Training Pipeline

## 📖 Project Overview
This project provides a pipeline for:
   - Capturing head pose data (images) using MediaPipe and OpenCV
   - Training a CNN model to classify head poses
  - Real-time inference to predict head pose using webcam input


## 📂 Folder Structure

```bash
├── data/
│   └── dataset/
│       └── [person_name]/
│           └── [pose_label].jpg
├── model/
│   └── head_pose_model.pth
├── capture.py
├── train.py
├── inference.py
└── README.md
```


## 🖼️ 1. Data Collection (Head Pose Dataset)

Capture images of head poses:
- Poses: `forward`, `up`, `down`, `left`, `right`, `up-left`, `up-right`, `down-left`, `down-right`
- Tech: MediaPipe Face Mesh + OpenCV
- Output: Cropped face images saved per pose for each individual
