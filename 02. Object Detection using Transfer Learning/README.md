# 🧠 Real-Time Object Detection using MobileNetSSD

---

## 🧾 **Index**
1. [Project Overview](#-project-overview)  
2. [Problem Statement](#-problem-statement)  
3. [Key Features & Terminologies](#-key-features--terminologies)  
4. [Workflow Summary](#-workflow-summary) 
5. [Model Evaluation](#-model-evaluation)   
6. [Key Takeaways](#-key-takeaways)

---

## 🌍 Project Overview
A real-time object detection project leveraging the **MobileNetSSD** model — a lightweight and efficient deep learning architecture that combines **MobileNet** (for feature extraction) and **SSD** (Single Shot MultiBox Detector) for object detection.  
The model is capable of identifying 20 common object categories from live webcam feed or video streams with high speed and decent accuracy.

---

## 🎯 **Problem Statement**
To develop a **real-time object detection system** that can:
- Detect and classify multiple objects in a single frame.  
- Run efficiently on CPU or GPU in real-time.  
- Utilize a **pre-trained MobileNetSSD model** trained on the **PASCAL VOC dataset**.

---

## 🧩 **Key Features & Terminologies**
- **MobileNet:** A lightweight CNN optimized for mobile and embedded devices.  
- **SSD (Single Shot Detector):** Detects multiple objects in one forward pass of the network.  
- **PASCAL VOC Classes:** 20 object categories (e.g., person, dog, car, bus, etc.).  
- **OpenCV DNN Module:** Used for loading and running the pre-trained model efficiently.  
- **Real-time Processing:** Uses webcam frames for continuous detection at near real-time FPS.

---

##  🛠 **Workflow Summary** 
1. **Load Pre-trained Model**  
   - Files used: `MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel`.  

2. **Initialize Object Classes**  
   ```python
   CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor"]
   
3. **Capture Video Frames**
    - Streamed from webcam using cv2.VideoCapture(0)
    
4. **Preprocess Each Frame**
    - Resize → Convert to blob → Pass to network → Get detections.

5. **Draw Bounding Boxes & Labels**
    - Visualize predictions with class name and confidence score.

6. **Exit Condition**
    - Press Esc or q to terminate the real-time loop safely.
  
---
  
## 📈 Model Evaluation

- **Pre-trained Dataset:** PASCAL VOC 2007 + 2012

- **Objects Detected:** 20 categories (person, dog, cat, car, etc.)

- **Performance:**

    - **Real-time FPS:** ~20–25 on CPU, ~40+ on GPU

    - **Mean Average Precision (mAP):** ~72% (as reported by original MobileNetSSD paper)
  
---
 
## 🪄 Key Takeaways

- ✨ MobileNetSSD achieves an excellent trade-off between speed and accuracy, making it ideal for real-time applications.
- 🚀 Demonstrates how pre-trained models (transfer learning) can be reused effectively without retraining.
- 🧠 Provides foundational understanding for future enhancements like custom dataset retraining or object tracking.
- 💻 Can be easily deployed on edge devices, including Raspberry Pi, Jetson Nano, or mobile platforms.

---

## 📇 Author

Anil Reddy Perugu💝

📧 Email: peruguanilreddy6@gmail.com

📍 Feel free to reach out for queries, suggestions, or collaborations!
