## 🤖 Face Recognition System using Haar Cascade & LBPH

A Computer Vision project to detect and recognize human faces in real-time using Haar Cascade Classifier and LBPH (Local Binary Patterns Histograms) Face Recognizer. The system captures face datasets, trains a recognition model, and identifies people live through a webcam feed. Now, let’s dive into the complete workflow and understand how the motion detection pipeline works step-by-step 🚀

---

## 📚 Index

1 📖 [Project Overview](#-project-overview)  
2 🎯 [Problem Statement](#-problem-statement)  
3 🧠 [Key Features & Terminologies](#-key-features--terminologies)  
4 ⚙️ [Workflow Summary](#-workflow-summary)  
5 🎥 [Real-Time Face Recognition](#-real-time-face-recognition)  
6 📊 [Project Structure](#-project-structure)  
7 🛠️ [Technologies Used](#️-technologies-used)  
8 📈 [Future Improvements](#-future-improvements)

---

## 📌 Project Overview

This project is a complete Face Recognition System built using Python and OpenCV. It uses Haar Cascade for detecting faces and the LBPH Face Recognizer algorithm for identifying people based on trained face datasets.

The project consists of three major stages:

* 📸 Collecting face datasets
* 🧠 Training the recognizer model
* 🎥 Performing live face recognition

The system detects faces from webcam frames, predicts the identity of the person, and displays the predicted name in real-time.

---

## 🎯 Problem Statement

Traditional identification systems require manual verification and are not suitable for automated real-time recognition tasks. The goal of this project is to build a lightweight real-time face recognition system capable of:

* Detecting faces from webcam streams
* Storing face datasets person-wise
* Training a recognizer model
* Recognizing known individuals live through webcam

---

## ✨ Key Features & Terminologies

🔍 Haar Cascade Classifier: A pre-trained object detection algorithm provided by OpenCV used for detecting human faces from images and video frames.

🧠 LBPH Face Recognizer: LBPH (Local Binary Patterns Histograms) is a classical face recognition algorithm that learns facial texture patterns and predicts identities based on trained datasets.

🖼️ Grayscale Conversion: Images are converted from BGR to grayscale because Haar Cascade and LBPH work more efficiently on grayscale images.

🏷️ Label Mapping: Each person folder is assigned a unique numeric ID.

---

## 🔄 Workflow Summary

```text
Webcam Feed
     ↓
Face Detection using Haar Cascade
     ↓
Crop Face Region
     ↓
Resize Face Image
     ↓
LBPH Face Recognition
     ↓
Predict Label
     ↓
Convert Label → Name
     ↓
Display Name on Webcam Feed
```
---

## 🎥 Real-Time Face Recognition

The final recognition system performs:

* Webcam Capture
* Face Detection
* Face Cropping
* Face Resizing
* Identity Prediction
* Live Name Display

---

## 📊 Project Structure

```text
FaceRecognitionProject/
│
├── dataset/
│   ├── Reddy/
│   ├── Rahul/
│   └── Sai/
│
├── collect_faces.py
├── train_model.py
├── main.py
│
├── trainer.yml
│
└── README.md
```

---

## 🛠️ Technologies Used

* 🐍 Python
* 👁️ OpenCV
* 📊 NumPy
* 🎥 Haar Cascade Classifier
* 🧠 LBPH Face Recognizer

---

## 📈 Future Improvements

1. Integrate YOLO Face Detection
2. Add Attendance System
3 Store user details in Database
4. Add Face Mask Detection
5. Improve unknown face detection accuracy
6. Deploy as Web Application

---


## Author 🙌

Anil Reddy Perugu 💝

📧 Email: peruguanilreddy6@gmail.com

📍 Feel free to reach out for queries, suggestions, or collaborations!
