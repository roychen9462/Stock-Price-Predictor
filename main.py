# -*- coding: utf-8 -*-

# Predict the price of the stock of a company

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import os

# Loading the dataset
df_train = pd.read_csv('train.csv')['price'].values
df_train = df_train.reshape(-1, 1)
df_test = pd.read_csv('test.csv')['price'].values
df_test = df_test.reshape(-1, 1)

dataset_train = np.array(df_train)
dataset_test = np.array(df_test)

# Preprocess your data
scaler = StandardScaler()
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.fit_transform(dataset_test)
##########################################

# We create the X_train and Y_train from the dataset train
# We take a price on a date as y_train and save the previous 50 closing prices as x_train

trace_back = 50
def create_dataset(df):
    x, y = [], []
    for i in range(trace_back, len(df)):
        x.append(df[i-trace_back:i, 0])
        y.append(df[i, 0])
    return np.array(x),np.array(y)

x_train, y_train = create_dataset(dataset_train)

x_test, y_test = create_dataset(dataset_test)

# Build the RNN model
model = keras.Sequential()
model.add(layers.Input((50,1)))
model.add(layers.LSTM(150, return_sequences=True))
model.add(layers.Dense(50))
model.add(layers.LSTM(150))
model.add(layers.Dense(50))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer="adam", loss="mse")
model.fit(x_train, y_train, batch_size=5, epochs=10)
##########################################

# Predictions on X_test

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

"""## [Step 6]: Checking the Root Mean Square Error on X_test"""

rmse_score = mean_squared_error([x[0] for x in y_test_scaled], [x[0] for x in predictions], squared=False)
print("RMSE:",rmse_score)

