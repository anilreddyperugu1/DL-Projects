# 🧠 Brain Tumor Classification using Neural Networks

🧠 A deep learning project that classifies brain MRI images into Benign, Malignant, and Normal categories using a Convolutional Neural Network (CNN). It aims to assist in early and accurate detection of brain tumors through automated medical image analysis.

---

## 📘 Index  
1. 📌 [Project Overview](#-project-overview)  
2. 🎯 [Problem Statement](#-problem-statement)  
3. 📚 [Key Features & Terminologies](#-key-features--terminologies)  
4. 🔄 [Workflow Summary](#-workflow-summary)  
6. 🎯 [Key Takeaways](#-key-takeaways)    

---

## 🧩 Project Overview  
A deep learning project designed to classify **brain MRI images** into three categories — **Benign**, **Malignant**, and **Normal** — using a **Convolutional Neural Network (CNN)**.  
The model leverages image preprocessing and data augmentation to achieve high classification accuracy on unseen MRI scans.  

---

## 🎯 Problem Statement  
Brain tumors pose a critical diagnostic challenge. Manual diagnosis from MRI scans is time-consuming and prone to human error.  
This project aims to automate the classification of MRI brain images to assist in **early and accurate tumor detection**, reducing the diagnostic burden on radiologists.  

---

## 🔑 Key Features & Terminologies  

- 🧠 **Convolutional Neural Network (CNN)** — Core architecture used for feature extraction and image classification.  
- 🌀 **ImageDataGenerator** — Used for image preprocessing and augmentation to enhance generalization.  
- ✂️ **Shear Range** — Randomly shears the image for geometric transformation and variance.  
- 🔍 **Zoom Range** — Randomly zooms images in/out to make the model invariant to scale.  
- 🧾 **Softmax Activation** — Converts final layer outputs into class probabilities.  

---

## 🔄 Workflow Summary 

1. **Data Preprocessing**  
   - MRI images are resized to `(128, 128)` pixels.  
   - Augmentation applied: random shear, zoom, and horizontal flip.  

2. **Model Building**  
   - CNN architecture with multiple convolution, pooling, and dense layers.  
   - Output layer with 3 neurons (for 3 classes) and softmax activation.  

3. **Model Compilation**  
   - Optimizer: `Adam`  
   - Loss Function: `categorical_crossentropy`  
   - Metrics: `accuracy`  

4. **Model Training**  
   - Trained using `train_generator` and validated on `validation_generator`.   

5. **Prediction**  
   - Single image predictions made using `keras.utils.load_img()` and `img_to_array()`.  
   - Image array expanded to match batch input dimension before prediction.  

---
## 💡 Key Takeaways

- CNNs can effectively classify MRI brain scans with high accuracy.
- Data augmentation significantly improves generalization and reduces overfitting.
- Precision and recall are more meaningful metrics in medical imaging than accuracy alone.
- Model can be deployed as a diagnostic assistant to support radiologists in identifying tumor types

---

📇 Author

Anil Reddy Perugu💝

📧 Email: peruguanilreddy6@gmail.com

📍 Feel free to reach out for queries, suggestions, or collaborations!
