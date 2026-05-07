# 🎥 Moving Object Detection using OpenCV

A real-time computer vision project that detects moving objects using OpenCV by comparing consecutive video frames and highlighting detected motion with bounding boxes. The system intelligently identifies movement while filtering out small noise and unnecessary pixel variations. Now, let’s dive into the complete workflow and understand how the motion detection pipeline works step-by-step 🚀

---

📌 Index

1 📖 [Project Overview](#-project-overview)  
2 🎯 [Problem Statement](#-problem-statement)  
3 🧠 [Key Features & Terminologies](#-key-features--terminologies)  
4 ⚙️ [Workflow Summary](#-workflow-summary)  
5 🛠️ [Technologies Used](#️-technologies-used)  
7 🚀 [Motion Detection Pipeline](#-motion-detection-pipeline)  
8 📌 [Key OpenCV Functions Used](#-key-opencv-functions-used)  
9 🎥 [Model Output / Detection Logic](#-model-output--detection-logic)  
10 🔮 [Future Improvements](#-future-improvements)  
11 📖 [Key Takeaways](#-key-takeaways)  

---

## 🚀 Project Overview

A real-time computer vision project that detects moving objects using a webcam and highlights them with bounding boxes. The system continuously compares consecutive video frames and identifies significant motion changes.

If motion is detected, the system displays: **Moving Object Detected**

Otherwise: **Normal**

---

## ❓ Problem Statement

Traditional CCTV systems continuously record video but do not intelligently identify movement.

The goal of this project is to:

* Detect moving objects in real time
* Ignore small noise and unnecessary pixel variations
* Draw bounding boxes around moving objects
* Display motion detection status dynamically

---

## ✨ Key Features & Terminologies

✅ Features

* 📷 Real-time webcam video capture
* 🎯 Motion detection using frame differencing
* 🧠 Noise reduction using Gaussian Blur
* ⚡ Thresholding and contour detection
* 🟩 Bounding box visualization
* 🖥️ Live status display

----

## 📚 Important Terminologies

🔹 **Frame Differencing**: Compares two consecutive frames to identify changes.

🔹 **Gaussian Blur**: Smooths the image and removes small noise.

🔹 **Thresholding**: Converts the difference image into a binary black-and-white image.

🔹 **Contours**: Boundaries of white motion regions detected in the frame.

🔹 **Bounding Box**: Rectangle drawn around detected moving objects.

---

## 🔄 Workflow Summary

```text
Capture Video
      ↓
Resize Frame
      ↓
Convert to Grayscale
      ↓
Apply Gaussian Blur
      ↓
Compare Consecutive Frames
      ↓
Thresholding
      ↓
Dilation
      ↓
Find Contours
      ↓
Draw Bounding Boxes
      ↓
Display Motion Status
```
---

## 🛠️ Technologies Used

* 🐍 Python
* 👁️ OpenCV
* ⚡ Imutils

---

## ⚙️ Project Workflow

1️⃣ Capture Live Video: The webcam continuously captures frames in real time.

2️⃣ Resize the Frame: Frames are resized for faster processing.

3️⃣ Convert to Grayscale: Color information is unnecessary for motion detection.

4️⃣ Apply Gaussian Blur: Reduces noise and small pixel variations.

5️⃣ Compare Frames: Calculates the absolute difference between previous and current frame.

6️⃣ Apply Thresholding: Highlights significant changes.

7️⃣ Dilate the Image: Strengthens white motion regions.

8️⃣ Find Contours: Detects boundaries of moving regions.

9️⃣ Ignore Small Contours: Tiny contours are filtered to avoid false motion detection.

🔟 Draw Bounding Boxes: Bounding boxes are drawn around detected motion.

---

## 🧠 Motion Detection Pipeline
```text
Previous Frame
       ↓
Current Frame
       ↓
Absolute Difference
       ↓
Thresholding
       ↓
Contour Detection
       ↓
Bounding Box Drawing
       ↓
Motion Detection Result
```
---

## 📌 Key OpenCV Functions Used

| Function | Purpose |
|---|---|
| `cv2.VideoCapture()` | Captures webcam video |
| `cv2.cvtColor()` | Converts image to grayscale |
| `cv2.GaussianBlur()` | Removes image noise |
| `cv2.absdiff()` | Finds frame differences |
| `cv2.threshold()` | Converts image to binary |
| `cv2.dilate()` | Expands white motion regions |
| `cv2.findContours()` | Detects moving regions |
| `cv2.contourArea()` | Filters small contours |
| `cv2.boundingRect()` | Gets rectangle coordinates |
| `cv2.rectangle()` | Draws bounding boxes |
| `cv2.imshow()` | Displays video output |

---

## 🎯 Model Output / Detection Logic

🟢 No Motion →  Normal

🔴 Motion Detected → Moving Object Detected

Bounding boxes are displayed around moving objects in the video frame.

---

## 🚀 Future Improvements

* 🤖 Human detection using YOLOv8
* 📸 Save image when motion is detected
* 🔔 Sound alert system
* ☁️ Cloud-based surveillance system
* 🧠 Background subtraction using MOG2
* 📱 Mobile notification integration

---

## 📖 Key Takeaways

1. Learned real-time video processing

2. Understood frame differencing concept

3. Implemented contour-based motion detection

4. Learned image preprocessing techniques

5. Understood thresholding and dilation

6. Implemented bounding box visualization

7. Built a complete real-time computer vision project using OpenCV

---

## Author 😉

Anil Reddy Perugu💝

📧 Email: peruguanilreddy6@gmail.com

📍 Feel free to reach out for queries, suggestions, or collaborations!
