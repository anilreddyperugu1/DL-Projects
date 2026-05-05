# 📈 Stock Price Prediction using LSTM

A deep learning project that predicts stock prices using LSTM by learning patterns from historical time-series data. The model leverages sequence modeling to forecast future prices with strong trend alignment. Let's dive in 🚀

---

## 📑 Index
1 📖 [Project Overview](#-project-overview)   
2 🎯 [Problem Statement](#-problem-statement)   
3 🧠 [Key Features & Terminologies](#-key-features--terminologies)  
4 ⚙️ [Workflow Summary](#-workflow-summary)   
5 📊 [Model Evaluation]((#-model-evaluation))   
6 💡 [Key Takeaways](#-key-takeaways)  
 

---

## 📌 Project Overview

This project focuses on predicting stock prices using a Long Short-Term Memory (LSTM) model trained on historical time-series data. It involves preprocessing the data, creating sequential inputs using a sliding window approach, and training a deep learning model to capture temporal patterns. The goal is to accurately forecast future stock prices and evaluate the model’s performance using appropriate regression metrics.

---

## ❓ Problem Statement

Stock prices are highly dynamic and depend on past trends.  

The objective of this project is to:
- Learn patterns from historical stock price data  
- Predict future prices using sequential modeling  
- Evaluate how well LSTM captures time-based dependencies  

---

## 🧠 Key Features & Terminologies

### 🔹 LSTM (Long Short-Term Memory)
A type of Recurrent Neural Network (RNN) designed to learn **long-term dependencies** in sequential data.

### 🔹 Time Series Data
Data points collected over time (e.g., daily stock prices).

### 🔹 Sliding Window Technique
Used to create sequences:
- Input → last 60 days  
- Output → next day  

### 🔹 Scaling (MinMaxScaler)
- Transforms data into range **0 to 1**  
- Helps stabilize neural network training  

### 🔹 Sequence Creation
Transforms raw data into:
X → (samples, timesteps, features)
y → (samples,)

---

### 🔄 Workflow Summary

1️⃣ Data Preprocessing
  * Loaded dataset
  * Removed null values
  * Sorted data chronologically
  * Selected Close price

2️⃣ Data Scaling
  * Applied MinMaxScaler
  * Prevents unstable training

3️⃣ Train-Test Split
  * 80% → Training
  * 20% → Testing
  * No shuffling (time-based split)

4️⃣ Sequence Creation
  * Used sliding window of 60 days
  * Generated:
      * X_train, y_train
      * X_test, y_test
  
5️⃣ Model Building
  * LSTM Layer
  * Dropout (for regularization)
  * Dense output layer

6️⃣ Model Training
  * Optimizer → Adam
  * Loss → Mean Squared Error
  * Trained for multiple epochs

7️⃣ Predictions
  * Predicted stock prices on test data
  * Applied inverse scaling

8️⃣ Visualization 📊
  * Compared:
      * Actual prices
      * Predicted prices
  * Observed strong trend alignment

9️⃣ Model Evaluation

  🔢 Metrics Used:
  * MSE (Mean Squared Error)
  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)

📈 Results:
  MSE  = 5.34
  MAE  = 1.64
  RMSE = 2.31

---

### 🧠 Key Takeaways:
  * Average prediction error ≈ 1.64 units (MAE)
  * RMSE ≈ 2.31 indicates no deviation
  * Model captures overall trend very effectively

---

## 🙌 Author

Anil Reddy Perugu💝

📧 Email: peruguanilreddy6@gmail.com

📍 Feel free to reach out for queries, suggestions, or collaborations!
