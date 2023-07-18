import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv('BTCUSDT-1m-2023-05.csv', delimiter=';')

data = df['close'].values

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# ARIMA modeling
model_arima = ARIMA(train_data, order=(2, 1, 2))
model_arima_fit = model_arima.fit()

# Get residuals from ARIMA model
residuals = model_arima_fit.resid

# Prepare lagged input data for the neural network
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

# Prepare lagged input data for the neural network
look_back = 500

residuals_mean = np.mean(residuals)
residuals_std = np.std(residuals)

residuals = (residuals - residuals_mean) / residuals_std

train_X, train_Y = create_dataset(residuals, look_back)

# Define and train the neural network
model_nn = keras.Sequential()
model_nn.add(keras.layers.Dense(32, activation='relu', input_shape=(look_back,)))
model_nn.add(keras.layers.Dense(16, activation='relu'))
model_nn.add(keras.layers.Dense(1))
model_nn.compile(loss='mean_squared_error', optimizer='adam')
prev_loss = float('inf')  # Set an initial loss value
for epoch in range(1000):
    print(f"Epoch {epoch+1}/{1000}")
    model_history = model_nn.fit(train_X, train_Y, epochs=1, batch_size=64, verbose=1)
    loss = model_history.history['loss'][0]
    
    if loss > prev_loss:
        pass
    elif prev_loss - loss < 0.001:
        print("Convergence reached.")
        break
    
    prev_loss = loss

# Make predictions on the test set using ARIMA
history = list(train_data) + list(test_data[:1000])
test_predictions = []

for i in range(1000):
    model_arima = ARIMA(history, order=(2, 1, 2))
    model_arima_fit = model_arima.fit()
    all_residuals = model_arima_fit.resid
    residuals = all_residuals[-look_back:]  # Use the latest residuals
    residuals_mean = np.mean(residuals)
    residuals_std = np.std(residuals)
    residuals = (residuals - residuals_mean) / residuals_std

    input_data = np.array(residuals)
    input_data = np.reshape(input_data, (1, look_back))
    prediction = model_nn.predict(input_data)
    predicted_price = history[-1] - ((prediction[0][0] + residuals_mean) * residuals_std)  # Reverse differencing
    test_predictions.append(predicted_price)
    history.append(predicted_price)

    all_residuals_mean = np.mean(all_residuals)
    all_residuals_std = np.std(all_residuals)

    all_residuals = (all_residuals - all_residuals_mean) / all_residuals_std

    train_X, train_Y = create_dataset(all_residuals, look_back)

    model_nn = keras.Sequential()
    model_nn.add(keras.layers.Dense(32, activation='relu', input_shape=(look_back,)))
    model_nn.add(keras.layers.Dense(16, activation='relu'))
    model_nn.add(keras.layers.Dense(1))
    model_nn.compile(loss='mean_squared_error', optimizer='adam')
    prev_loss = float('inf')  # Set an initial loss value
    for epoch in range(1000):
        print(f"Epoch {epoch+1}/{1000}")
        model_history = model_nn.fit(train_X, train_Y, epochs=1, batch_size=64, verbose=1)
        loss = model_history.history['loss'][0]

        if loss > prev_loss:
            pass
        elif prev_loss - loss < 0.001:
            print("Convergence reached.")
            break

        prev_loss = loss

    if len(history) % 10 == 0:
        plt.plot(test_data[1000:], label='Actual')
        plt.plot(test_predictions, label='Predicted')
        plt.legend()
        plt.show()