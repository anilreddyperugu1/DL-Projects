# 🚶‍♂️ Social Distance Monitoring using YOLOv8

A computer vision project that monitors social distancing using real-time video input. It leverages a pretrained YOLOv8 model to detect people and compute distances between them. The system highlights violations visually, helping analyze crowd safety in public spaces. Let's dive in 🚀

---

## 📌 Index
- 📖 [Project Overview](#-project-overview)   
- 🎯 [Problem Statement](#-problem-statement) 
- 🧠 [Key Features & Terminologies](#-key-features--terminologies)  
- ⚙️ [Workflow Summary](#-workflow-summary)   
- 📊 [Model Evaluation]((#-model-evaluation)) 
- 💡 [Key Takeaways](#-key-takeaways)  

---

## 📖 Project Overview
A deep learning–based computer vision project to monitor social distancing in real-time using video input. The system detects people using the YOLOv8 object detection model and calculates the distance between individuals to identify violations. It visually highlights safe and unsafe distances using bounding boxes.

---

## 🎯 Problem Statement
During crowded situations (like public places, malls, or streets), maintaining social distance becomes difficult. The goal of this project is to:
- Automatically detect people in a video stream  
- Measure distances between them  
- Identify and highlight social distancing violations  

---

## 🧠 Key Features & Terminologies

### 🔹 YOLOv8 (You Only Look Once)
- A real-time object detection model  
- Detects multiple objects (here, we use only **person class**)  

### 🔹 Bounding Box
- Rectangle drawn around detected objects  
- Represented as `(x1, y1, x2, y2)`  

### 🔹 Centroid
- Center point of each detected person  
- Used for distance calculation  

### 🔹 Euclidean Distance
- Measures distance between two people  
- Formula:
  ```
  d = √((x2 - x1)² + (y2 - y1)²)
  ```

### 🔹 Violation Threshold
- If distance < threshold (e.g., 50 pixels) → violation  

---

## 🔄 Workflow Summary

```
Video Input → YOLOv8 Detection → Extract People → Compute Centroids 
→ Calculate Distances → Detect Violations → Draw Bounding Boxes → Display Output
```

### 🟢 Steps:
1. Capture video using OpenCV  
2. Detect people using YOLOv8  
3. Extract bounding boxes and centroids  
4. Compute pairwise distances  
5. Identify violations based on threshold  
6. Draw:
   - 🟢 Green box → Safe  
   - 🔴 Red box → Violation  
7. Display results in real-time  

---

## 📊 Model Evaluation
- Model Used: YOLOv8 (pretrained on COCO dataset)  
- Performance:
  - ⚡ Real-time detection (~15–25 ms per frame)  
  - 🎯 High accuracy for person detection   

---

## 💡 Key Takeaways
- Learned real-time object detection using YOLOv8  
- Understood distance-based violation logic  
- Gained hands-on experience with OpenCV video processing  
- Built an end-to-end deep learning application  
- Improved debugging and system design skills  

---

## 🚀 Future Enhancements
- 📏 Convert pixel distance to real-world distance (meters)  
- 🛰️ Bird’s-eye view transformation  
- 🔊 Alert system for violations  
- 💾 Save output video and violation snapshots  
- 📊 Heatmap for crowd density  

---

## 🛠️ Tech Stack
- Python 🐍  
- OpenCV 📷  
- Ultralytics YOLOv8 🤖  
- NumPy  
- SciPy  

---

## ▶️ How to Run

```bash
pip install ultralytics opencv-python numpy scipy
```

```bash
python main.py
```

---

## 🎮 Controls
- Press **Q** → Quit program  
- Press **ESC** → Exit  

---

## 📁 Project Structure

```
Social-Distance-Monitoring/
│
├── main.py
├── videos/
│   └── vid1.mp4
├── output/
└── README.md
```

---

## ⭐ Final Note
This project demonstrates how deep learning and computer vision can be applied to solve real-world problems like crowd monitoring and safety enforcement

---

## 🙌 Author

Anil Reddy Perugu💝

📧 Email: peruguanilreddy6@gmail.com

📍 Feel free to reach out for queries, suggestions, or collaborations!
