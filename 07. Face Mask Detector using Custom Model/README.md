# 🧠😷 Face Mask Detection using MobileNetV2

A real-time face mask detection system built using MobileNetV2 and OpenCV that identifies whether a person is wearing a mask through webcam input. It combines transfer learning with computer vision to deliver fast and accurate predictions. Let’s dive in to see more about the project  🚀

---

## 📑 Index
1. 📖 [Project Overview](#-project-overview)   
2. 🎯 [Problem Statement](#-problem-statement)   
3. 🧠 [Key Features & Terminologies](#-key-features--terminologies)  
4. ⚙️ [Workflow Summary](#-workflow-summary)   
5. 📊 [Model Evaluation]((#-model-evaluation))   
6. 💡 [Key Takeaways](#-key-takeaways)  

---

## 📌 Project Overview
A deep learning project to detect whether a person is wearing a face mask or not in real-time using a webcam. The model is built using **MobileNetV2 (Transfer Learning)** and integrated with **OpenCV** for live face detection and prediction. This system captures video input, detects faces, and classifies each face as **Mask 😷** or **No Mask ❌**.

---

## ❓ Problem Statement
During health crises like COVID-19, ensuring mask compliance in public places becomes crucial. Manual monitoring is inefficient and error-prone.

👉 The goal of this project is to:
- Automatically detect faces in real-time  
- Classify whether a mask is worn or not  
- Provide instant visual feedback  

---

## 🧩 Key Features & Terminologies

### 🔹 Transfer Learning
Using a pre-trained model (**MobileNetV2**) trained on ImageNet to leverage learned features like edges, textures, and shapes.

### 🔹 Binary Classification
Classifying images into two categories:
- `with_mask`
- `without_mask`

### 🔹 OpenCV DNN Face Detector
Used instead of Haar Cascade for:
- Better accuracy  
- Stability  
- Real-time performance  

### 🔹 Image Preprocessing
- Resizing to `(224, 224)`  
- Normalization (`/255`)  
- Reshaping for model input  

### 🔹 Real-Time Inference
- Webcam input  
- Frame-by-frame prediction  
- Bounding box + label display

---

## 🔄 Workflow Summary

```text
Dataset → Preprocessing → Train MobileNetV2 → Save Model → 
Load Model → Capture Webcam → Detect Face → Predict Mask → Display Output
```
---

## 🔄 Steps:

1. 📁 Prepare dataset (with_mask, without_mask)
2. 🔀 Split into train & validation
3. 🏗️ Build model using MobileNetV2
4. 🏋️ Train model on dataset
5. 💾 Save trained model
6. 🎥 Capture real-time video
7. 🧍 Detect faces using DNN
8. 🤖 Predict mask usage
9. 🟩 Display results with bounding boxes

---

## 📊 Model Evaluation

The trained MobileNetV2 model achieved strong performance on the validation dataset.

| Metric               | Value      |
|---------------------|------------|
| Validation Accuracy | **98.83%** |
| Validation Loss     | **0.0298** |

---

## 💡 Key Takeaways

* ✅ Transfer Learning significantly reduces training time
* ✅ MobileNetV2 is lightweight and ideal for real-time applications
* ⚠️ Haar Cascade is unreliable → DNN detector preferred
* ⚠️ Real-world performance depends on dataset diversity
* 🚀 End-to-end pipeline from training → deployment achieved

---

## 🚀 Future Improvements

* 🔊 Add alert system for no-mask detection
* 📈 Improve dataset with real-world variations
* 🌐 Deploy as web application
* 🤖 Upgrade to YOLO for unified detection

---

## 🙌 Conclusion

This project successfully demonstrates how deep learning can be applied to real-world problems using:

* Transfer Learning
* Computer Vision
* Real-time inference

👉 A complete ML pipeline from data → model → deployment 🚀

---

## Author 😉

Anil Reddy Perugu💝

📧 Email: peruguanilreddy6@gmail.com

📍 Feel free to reach out for queries, suggestions, or collaborations!
