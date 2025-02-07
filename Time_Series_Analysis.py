<paste your entire script here>
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_data(file):
    data = pd.read_csv(file)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])
    return data

def train_arima(series):
    try:
        model = ARIMA(series, order=(5,1,0))
        model_fit = model.fit()
        return model_fit.forecast(steps=30)
    except Exception as e:
        print("Error in ARIMA model:", e)
        return None

def train_lstm(series):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(series.values.reshape(-1,1))
    X, y = [], []
    for i in range(10, len(data_scaled)):
        X.append(data_scaled[i-10:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(units=50, return_sequences=False),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    X_forecast = data_scaled[-10:].reshape(1, 10, 1)
    forecast = model.predict(X_forecast)
    return scaler.inverse_transform(forecast)

def inventory_optimization(series):
    lead_time = 7
    safety_stock = np.std(series) * 1.65
    reorder_point = np.mean(series) * lead_time + safety_stock
    order_cost, holding_cost = 50, 2
    annual_demand = np.mean(series) * 365
    optimal_order_quantity = np.sqrt((2 * order_cost * annual_demand) / holding_cost)
    return reorder_point, optimal_order_quantity

if __name__ == "__main__":
    file_path = input("Enter the path to your CSV file: ")
    data = load_data(file_path)
    product_ids = data['Product_ID'].unique()
    print("Available Product IDs:", product_ids)
    product_id = input("Select a Product ID: ")
    try:
        product_id = int(product_id)
        if product_id not in product_ids:
            raise ValueError("Invalid Product ID")
    except ValueError:
        print("Invalid Product ID selected.")
        exit()
    
    data_product = data[data['Product_ID'] == product_id]
    time_series = data_product.set_index('Date')['Demand']
    
    plt.figure(figsize=(10,5))
    plt.plot(time_series)
    plt.title("Time Series of Product Demand")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.show()
    
    forecast_arima = train_arima(time_series)
    forecast_lstm = train_lstm(time_series)
    reorder_point, optimal_order_quantity = inventory_optimization(time_series)
    
    print("\n🔮 ARIMA Forecast:")
    print(forecast_arima)
    print("\n📌 Reorder Point:", round(reorder_point, 2), "units")
    print("📦 Optimal Order Quantity (EOQ):", round(optimal_order_quantity, 2), "units")
