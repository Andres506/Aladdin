import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
#from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st

@st.cache_data(ttl=600)
def prediccion_prophet(precios_df, dias):
    df_prophet = precios_df.reset_index()
    df_prophet.columns = ["ds", "y"]
    df_prophet["y"] = pd.to_numeric(df_prophet["y"], errors="coerce")
    df_prophet = df_prophet.dropna(subset=["y"])
    modelo = Prophet(daily_seasonality=True)
    modelo.fit(df_prophet)
    futuro = modelo.make_future_dataframe(periods=dias)
    forecast = modelo.predict(futuro)
    return modelo, forecast

@st.cache_data(ttl=600)
def prediccion_lstm(data, dias, pasos=30, epochs=10, batch_size=32):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    def crear_secuencias(data, pasos):
        X, y = [], []
        for i in range(pasos, len(data)):
            X.append(data[i - pasos : i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y_lstm = crear_secuencias(data_scaled, pasos)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    modelo_lstm = Sequential()
    modelo_lstm.add(LSTM(50, return_sequences=True, input_shape=(pasos, 1)))
    modelo_lstm.add(LSTM(50))
    modelo_lstm.add(Dense(1))
    modelo_lstm.compile(optimizer="adam", loss="mse")
    early_stop = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    modelo_lstm.fit(X, y_lstm, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])

    entrada = data_scaled[-pasos:].reshape(1, pasos, 1)
    predicciones = []
    for _ in range(dias):
        pred = modelo_lstm.predict(entrada, verbose=0)[0][0]
        predicciones.append(pred)
        entrada = np.append(entrada[:, 1:, :], [[[pred]]], axis=1)
    predicciones = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1)).flatten()
    return predicciones
