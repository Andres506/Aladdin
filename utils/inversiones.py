import streamlit as st
import pandas as pd
import json
import os
import requests
from datetime import datetime
from dateutil import parser

ARCHIVO_INV = "data/inversiones.json"

def cargar_datos():
    if os.path.exists(ARCHIVO_INV):
        with open(ARCHIVO_INV, "r") as f:
            return json.load(f)
    else:
        return []

def guardar_datos(datos):
    with open(ARCHIVO_INV, "w") as f:
        json.dump(datos, f, indent=4)

def obtener_precio_historico(ticker, fecha_hora):
    url = f"https://api.coingecko.com/api/v3/coins/{ticker}/history?date={fecha_hora.strftime('%d-%m-%Y')}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        try:
            return data["market_data"]["current_price"]["usd"]
        except:
            return None
    return None

def obtener_precio_actual(ticker):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ticker}&vs_currencies=usd"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        try:
            return data[ticker]["usd"]
        except:
            return None
    return None

def pestaña_inversiones():
    st.subheader("📈 Registro de Inversiones")

    datos = cargar_datos()

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker de criptomoneda (ej. xrp)").lower()
        cantidad = st.number_input("Cantidad invertida", min_value=0.0, step=0.01)
    with col2:
        fecha = st.date_input("Fecha de compra")
        hora = st.time_input("Hora de compra")

    if st.button("Agregar inversión"):
        if ticker and cantidad > 0:
            fecha_hora = datetime.combine(fecha, hora)
            precio_compra = obtener_precio_historico(ticker, fecha_hora)
            if precio_compra:
                datos.append({
                    "ticker": ticker,
                    "fecha": fecha_hora.isoformat(),
                    "cantidad": cantidad,
                    "precio_compra": precio_compra
                })
                guardar_datos(datos)
                st.success("✅ Inversión agregada correctamente.")
                st.rerun()
            else:
                st.error("❌ No se pudo obtener el precio histórico para esa fecha.")
        else:
            st.warning("⚠️ Ingresa todos los campos correctamente.")

    if datos:
        st.markdown("### 📋 Historial de Inversiones")

        tabla = []
        for i, fila in enumerate(datos):
            fecha_hora = parser.parse(fila["fecha"])
            actual = obtener_precio_actual(fila["ticker"])
            precio_compra = fila["precio_compra"]
            cantidad = fila["cantidad"]
            if actual:
                ganancia = round((actual - precio_compra) * cantidad, 2)
            else:
                ganancia = None

            color = "green" if ganancia and ganancia > 0 else "red"

            tabla.append({
                "N°": i + 1,
                "Ticker": fila["ticker"].upper(),
                "Fecha": fecha_hora.strftime("%Y-%m-%d %H:%M"),
                "Cantidad": cantidad,
                "💰 Compra ($)": round(precio_compra, 4),
                "📈 Actual ($)": round(actual, 4) if actual else "N/D",
                "🔄 Ganancia/Pérdida": f":{color}[${ganancia}]" if ganancia is not None else "N/D"
            })

        df = pd.DataFrame(tabla)
        st.dataframe(df, use_container_width=True, hide_index=True)

        eliminar = st.number_input("Eliminar inversión N°", min_value=1, max_value=len(datos), step=1)
        if st.button("Eliminar"):
            datos.pop(eliminar - 1)
            guardar_datos(datos)
            st.success("🗑️ Inversión eliminada.")
            st.rerun()
    else:
        st.info("No hay inversiones registradas.")
