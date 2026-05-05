import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


dataset = pd.read_csv("/Users/anilreddyperugu/Git/DL-Projects/08. Stock Price Prediction using LSTM/dataset.csv")

# print(dataset)
# print(dataset.head())
# print(dataset.tail())
# print(dataset.shape)
# print(dataset.info())
# print(dataset.describe)

# print(dataset.isnull().sum())

df = dataset.dropna()

# print(df.isnull().sum())

df = df.sort_values("Date")

data = df['close']

train_data=data.iloc[:2399]
test_data=data.iloc[2399:]

mm_scalar = MinMaxScaler()
train_transformed = mm_scalar.fit_transform(train_data)
test_transformed = mm_scalar.transform(test_data)

X_train = []
Y_train = []
window_size = 60
for i in range(window_size, len(train_transformed)):

    