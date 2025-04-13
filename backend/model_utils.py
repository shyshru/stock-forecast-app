import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Fetch stock data (Yahoo Finance doesn't provide Open Interest, so we'll use Volume)
def get_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data[["Close", "Volume"]]

# Prepare data with multiple features for LSTM
def prepare_data(data, look_back=90):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    x_data, y_data = [], []
    for i in range(look_back, len(scaled_data)):
        x_data.append(scaled_data[i-look_back:i])
        y_data.append(scaled_data[i, 0])  # Only predicting 'Close' price

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data, scaler

# LSTM model for multi-feature input
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model

# Predict future values
def predict_future(model, last_sequence, look_back, steps, scaler, num_features):
    predictions = []
    input_seq = last_sequence.reshape(1, look_back, num_features)
    for _ in range(steps):
        prediction = model.predict(input_seq, verbose=0)
        next_input = np.copy(input_seq)
        next_input = np.append(next_input[:, 1:, :], [[np.append(prediction, input_seq[0, -1, 1])]], axis=1)
        input_seq = next_input
        predictions.append(prediction[0, 0])
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Evaluate model
def evaluate_model(model, x_test, y_test, scaler):
    predictions = model.predict(x_test)
    predictions_rescaled = scaler.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), 1)))))
    y_test_rescaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), 1)))))

    rmse = np.sqrt(mean_squared_error(y_test_rescaled[:, 0], predictions_rescaled[:, 0]))
    mae = mean_absolute_error(y_test_rescaled[:, 0], predictions_rescaled[:, 0])
    r2 = r2_score(y_test_rescaled[:, 0], predictions_rescaled[:, 0])

    print("\nModel Accuracy Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    return rmse, mae, r2, predictions_rescaled[:, 0], y_test_rescaled[:, 0]

# Plot results
def plot_results(actual, predicted, data):
    plt.figure(figsize=(10, 5))
    plt.plot(data.index[-len(actual):], actual, label="Actual Prices", color="blue")
    plt.plot(data.index[-len(predicted):], predicted, label="Predicted Prices", color="red")
    plt.title("Stock Price Prediction vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# ========== RUN PIPELINE ==========

stock = input("Enter Stock Symbol (e.g., AAPL): ").strip().upper()
forecast_days = int(input("Enter Number of Days to Forecast: "))

data = get_stock_data(stock)
look_back = 90

x_data, y_data, scaler = prepare_data(data, look_back=look_back)
train_size = int(len(x_data) * 0.8)
x_train, y_train = x_data[:train_size], y_data[:train_size]
x_test, y_test = x_data[train_size:], y_data[train_size:]

model = build_lstm_model((x_train.shape[1], x_train.shape[2]))
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)

rmse, mae, r2, predicted, actual = evaluate_model(model, x_test, y_test, scaler)
plot_results(actual, predicted, data)

forecast = predict_future(model, x_data[-1], look_back, forecast_days, scaler, x_data.shape[2])
print("\nFuture Forecasted Prices:")
for i, price in enumerate(forecast):
    print(f"Day {i+1}: ${price[0]:.2f}")

def run_forecast(stock, forecast_days):
    data = get_stock_data(stock)
    x_data, y_data, scaler = prepare_data(data)
    train_size = int(len(x_data) * 0.8)
    x_train, y_train = x_data[:train_size], y_data[:train_size]
    model = build_lstm_model((x_train.shape[1], x_train.shape[2]))
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
    forecast = predict_future(model, x_data[-1], 90, forecast_days, scaler, x_data.shape[2])
    return [round(float(p[0]), 2) for p in forecast]
