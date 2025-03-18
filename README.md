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
├── Main_Model.py
├── SetUpID_Mediapipe.py
├── TranferLearningModel.py
└── README.md
```


## 🖼️ 1. Data Collection (Head Pose Dataset)

Capture images of head poses:
- Poses: `forward`, `up`, `down`, `left`, `right`, `up-left`, `up-right`, `down-left`, `down-right`
- Tech: MediaPipe Face Mesh + OpenCV
- Output: Cropped face images saved per pose for each individual
- Prompts for person name
- Captures images for different head angles automatically

## 🧠 2. CNN Model Training

Train a CNN model on the collected dataset to classify head poses.

### Features:
- Customizable architecture (default: simple CNN)
- Data Augmentation (optional)
- Accuracy visualization after training


## 🖥️ 3. Real-Time Head Pose Inference
Use your webcam to predict head poses in real time
- Loads the trained model
- Displays the predicted pose on the video feed

## 📚 Dependencies
- Python 3.x
- OpenCV
- MediaPipe
- PyTorch
- NumPy
- Matplotlib (optional for training plots)


## ✅ Pose Labels (Example)
Label	Description
forward	Looking straight at camera
up	Looking up
down	Looking down
left	Looking left
right	Looking right
up-left	Looking up and left
up-right	Looking up and right
down-left	Looking down and left
down-right	Looking down and right


## 🚀 Future Improvements
- Add more diverse dataset samples
- Fine-tune model with transfer learning


### ⏳ Project Still in progress .. 
