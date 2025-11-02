# 🍪 Custom Food Image Classification using MobileNetV2

A transfer learning project leveraging MobileNetV2 to identify various food categories from images. This project demonstrates how **lightweight architectures like MobileNetV2** can be fine-tuned effectively for custom classification tasks, achieving **97% accuracy** 🍴✨ while maintaining speed and deployment efficiency. Let's dive in 🚀 


---

## 🧭 **Index**
1. 📌 [Project Overview](#-project-overview)  
2. 🎯 [Problem Statement](#-problem-statement)  
3. 📚 [Key Features & Terminologies](#-key-features--terminologies)  
4. ⚙️ [Workflow Summary](#-workflow-summary) 
5. 📊 [Model Evaluation](#-model-evaluation)
6. 🎯 [Key Takeaways](#-key-takeaways)   

---

## 🧩 Project Overview

A **deep learning project** to classify images of food items using **MobileNetV2** through **Transfer Learning**.  
The model leverages pre-trained weights and fine-tuning to adapt to a **custom food dataset**, enabling accurate classification of multiple food categories.  This approach combines efficiency, speed, and accuracy for real-world food recognition tasks.

---

## 🎯 Problem Statement

The goal is to build a **robust image classification model** capable of identifying different types of food from images.  
By utilizing **MobileNetV2**, we minimize training time while maintaining high accuracy — suitable for deployment on mobile and edge devices.

---

## 🔑 Key Features & Terminologies

- **📚 Transfer Learning:** Reusing pre-trained knowledge from ImageNet to accelerate training.  
- **🧠 MobileNetV2:** A lightweight CNN optimized for speed and efficiency on low-resource devices.  
- **⚙️ Fine-Tuning:** Unfreezing the final layers of the base model to improve domain-specific learning.  
- **📈 Normalization:** Input image values scaled using the correct preprocessing (e.g., `preprocess_input()` for [-1,1] range).  
- **💾 Model Saving:** Model saved in the new `.keras` format (`model.save('model_mobilenet.keras')`) instead of legacy `.h5`.  

---

## 🛠 **Workflow Summary**
1. **Data Preparation:**  
   - Loaded and preprocessed custom food images.  
   - Applied normalization and augmentation (rescaling, rotation, flips).  

2. **Model Construction:**  
   - Imported **MobileNetV2** with pretrained ImageNet weights.  
   - Removed the final fully connected (FC) layers.  
   - Added custom Dense and Dropout layers for classification.  
   - Unfroze the last few convolutional blocks for fine-tuning.  

3. **Training:**  
   - Compiled using `Adam` optimizer and `categorical_crossentropy`.  
   - Trained the model

4. **Model Saving:**  
   - Saved as `model_mobilenet.keras`.  

5. **Prediction:**  
   - Loaded the model and passed test images using:
     ```python
     img_array = preprocess_input(img_array)
     img_array = np.expand_dims(img_array, axis=0)
     prediction = model.predict(img_array)
     ```
   - Example output:
     ```
     Predicted class: cookies with confidence of 18.6%
     ```

---

## 📊 Model Evaluation

- **Test Accuracy:** 🎯 **97%**  
- Evaluated on unseen test images from the custom dataset.  
- Model demonstrated strong generalization while maintaining low computational cost.  

---

## 💡 Key Takeaways

- 🚀 **Transfer learning** with MobileNetV2 drastically reduces training time while achieving high accuracy.  
- ⚙️ Correct **preprocessing consistency** (training vs inference) is critical for confidence stability.  
- 🔁 Replacing deprecated methods like `fit_generator()` with `fit()` ensures compatibility with TensorFlow ≥2.20.  
- 💾 The **`.keras` model format** is preferred over legacy `.h5`.  
- 📉 Low confidence on some real-world samples can occur due to **domain shifts**, not necessarily poor performance.  

---

## 📇 Author

Anil Reddy Perugu💝

📧 Email: peruguanilreddy6@gmail.com

📍 Feel free to reach out for queries, suggestions, or collaborations!

---

