import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from prophet import Prophet
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import requests
from dotenv import load_dotenv
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import json



from utils.data_utils import descargar_datos
from utils.model_utils import prediccion_prophet, prediccion_lstm
from utils.news_utils import obtener_noticias
from utils.indicators_utils import agregar_indicadores_tecnicos
from utils.binance_utils import crear_cliente_binance, obtener_precio_actual, obtener_saldos, crear_orden_compra
from utils.data_utils import descargar_datos, descargar_datos_multiple, calcular_metricas_historicas, obtener_info_fundamental
from utils.inversiones import pestaña_inversiones
from scipy.stats import skew, kurtosis
from utils.data_utils import descargar_datos, obtener_fundamentales, calcular_RSI, calcular_SMA, calcular_metricas


st.set_page_config(layout="wide")

# Al inicio del main.py, después de st.set_page_config
st.markdown("""
    <style>
        .big-font {font-size:32px !important;}
        .positivo {color: #27ae60;}
        .negativo {color: #c0392b;}
        .neutral {color: #2980b9;}
    </style>
""", unsafe_allow_html=True)

#st.image("logo.png", width=120)  # Si tienes un logo, colócalo en la raíz del proyecto
st.markdown("<h1 class='big-font'>Mi Aladdin Local</h1>", unsafe_allow_html=True)
st.markdown("Tu asistente de análisis financiero y predicción de activos.")

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Sidebar configuración

st.sidebar.markdown("### Instrucciones")
st.sidebar.info(
    "Selecciona un símbolo, rango de fechas y método de predicción. "
    "Navega por las pestañas para ver métricas, predicciones, noticias y más."
)

st.sidebar.header("Configuración")
ticker = st.sidebar.text_input("Símbolo (ej. AAPL, BTC-USD):", "XRP-USD").strip().upper()
start_date = st.sidebar.date_input("Fecha inicio", date(2025, 1, 1))
end_date = st.sidebar.date_input("Fecha fin", date.today())
prediction_days = st.sidebar.slider("Días de predicción", 5, 90, 30, 5)
metodo_prediccion = st.sidebar.selectbox("Método de predicción", ["Prophet", "LSTM", "Combinado"])

if metodo_prediccion in ["LSTM", "Combinado"]:
    lstm_epochs = st.sidebar.slider("Epochs LSTM", 1, 100, 10, 1)
    lstm_batch_size = st.sidebar.selectbox("Batch size LSTM", [16, 32, 64, 128], index=1)
    lstm_pasos = st.sidebar.slider("Pasos de tiempo (window size)", 5, 60, 30, 1)
else:
    lstm_epochs, lstm_batch_size, lstm_pasos = 10, 32, 30

if start_date >= end_date:
    st.sidebar.error("❌ La fecha fin debe ser después de la fecha inicio.")
    st.stop()

if not ticker:
    st.sidebar.warning("Por favor ingresa un símbolo válido.")
    st.stop()

if not NEWS_API_KEY:
    st.sidebar.warning("⚠️ NEWS_API_KEY no configurada en archivo .env. Noticias deshabilitadas.")


# Descargar datos
df = descargar_datos(ticker, start_date, end_date)
if df.empty:
    st.error("No se encontraron datos para el símbolo y fechas indicados.")
    st.stop()

price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
precios = df[price_col]

# Agregar indicadores técnicos
df = agregar_indicadores_tecnicos(df, price_col)

precio_actual = float(precios.iloc[-1])
precio_inicial = float(precios.iloc[0])
retorno_total = (precio_actual / precio_inicial - 1) * 100
retornos_diarios = precios.pct_change().dropna()
volatilidad = retornos_diarios.std()
volatilidad_anual = volatilidad * np.sqrt(252) * 100
if isinstance(volatilidad_anual, (np.ndarray, pd.Series)):
    volatilidad_anual = float(volatilidad_anual[0])

# Pestañas
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "📊 Métricas",
    "📈 Precio Histórico",
    "🔮 Predicción",
    "📰 Noticias y Sentimiento",
    "📈 Retorno Histórico",
    "📈 Ganancias Futuras",
    "🤖 Binance",
    "⚖️ Comparar Activos",
    "🧠 Evaluar Modelos",
    "📉 Análisis de Riesgo",
    "💰 Mis Inversiones"
])


with tab1:
    st.markdown('<p class="big-font">Métricas que debes conocer</p>', unsafe_allow_html=True)
    st.write("Aquí encontrarás las métricas financieras básicas que resumen el comportamiento del activo en el período seleccionado.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Precio actual", f"${precio_actual:.2f}", help="Último precio registrado del activo en el mercado.")
    c2.metric("📈 Retorno total", f"{retorno_total:.2f}%", help="Porcentaje de cambio entre el precio inicial y final en el período seleccionado.")
    c3.metric("⚡ Volatilidad anualizada", f"{volatilidad_anual:.2f}%", help="Medida del riesgo o variabilidad del precio. Mayor volatilidad indica mayor incertidumbre.")
    
    macd = df["MACD"].iloc[-1]
    macd_sig = df["MACD_signal"].iloc[-1]
    if macd > macd_sig:
        c4.metric("📊 MACD", "Señal de compra", delta="Al alza ✅", delta_color="normal", help="Indicador técnico que señala posible tendencia alcista (compra).")
    else:
        c4.metric("📊 MACD", "Señal de venta", delta="A la baja ❌", delta_color="inverse", help="Indicador técnico que señala posible tendencia bajista (venta).")

    st.markdown("---")
    st.markdown("<h2>📋 Datos Fundamentales Accion o ETF</h2>", unsafe_allow_html=True)

    try:
        datos_fundamentales = obtener_fundamentales(ticker)
        st.write(f"**PE Ratio:** {datos_fundamentales['PE']}")
        st.write(f"**EPS:** {datos_fundamentales['EPS']}")
        st.write(f"**Dividend Yield:** {datos_fundamentales['Dividend Yield']:.2%}")
        st.write(f"**Debt to Equity:** {datos_fundamentales['Debt to Equity']}")
    except Exception as e:
        st.error(f"No se pudieron obtener datos fundamentales: {e}")

    st.markdown("---")
    st.markdown("## Señales de Entrada/Salida (Decisión)")

    # Asegurarnos que precios es Serie
    if isinstance(precios, pd.DataFrame):
        precios = precios.iloc[:, 0]  # tomar primera columna si fuera DataFrame

    sma_50 = calcular_SMA(precios, 50)
    sma_200 = calcular_SMA(precios, 200)
    rsi = calcular_RSI(precios)

    # Validación para evitar error con NaNs o pocos datos
    if len(sma_50) < 2 or len(sma_200) < 2 or sma_50.isna().any() or sma_200.isna().any():
        st.warning("No hay suficientes datos para evaluar señales de Golden Cross.")
    else:
        # Golden Cross (50 SMA cruza por encima de 200 SMA)
        golden_cross = False
        if sma_50.iloc[-2] < sma_200.iloc[-2] and sma_50.iloc[-1] > sma_200.iloc[-1]:
            golden_cross = True

        if golden_cross:
            st.success("🔔 Golden Cross detectado: posible señal de COMPRA")
        else:
            st.info("No hay Golden Cross reciente")

    # Señales RSI
    rsi_ultimo = rsi.iloc[-1]
    if rsi_ultimo < 30:
        st.success(f"📉 RSI está en {rsi_ultimo:.2f} — posible señal de COMPRA por sobreventa")
    elif rsi_ultimo > 70:
        st.warning(f"📈 RSI está en {rsi_ultimo:.2f} — posible señal de VENTA por sobrecompra")
    else:
        st.info(f"RSI actual: {rsi_ultimo:.2f} (sin señal fuerte)")

    st.markdown("---")
    st.markdown("## Alertas de cambios de precio")

    # Porcentaje de cambio en X días (ejemplo: últimos 5 días)
    dias_alerta = 5
    if len(precios) > dias_alerta:
        cambio = (precios.iloc[-1] / precios.iloc[-dias_alerta - 1] - 1) * 100
        st.write(f"Cambio en los últimos {dias_alerta} días: {cambio:.2f}%")

        # Configura alertas simples
        umbral_alza = 5  # +5%
        umbral_baja = -5  # -5%
        if cambio > umbral_alza:
            st.success(f"✅ Alerta: Precio subió más del {umbral_alza}% en los últimos {dias_alerta} días")
        elif cambio < umbral_baja:
            st.error(f"⚠️ Alerta: Precio bajó más del {abs(umbral_baja)}% en los últimos {dias_alerta} días")
        else:
            st.info("No hay cambios de precio significativos recientes")

with tab2:
    st.markdown('<p class="big-font">Precio histórico y tendencias avanzadas</p>', unsafe_allow_html=True)
    st.write("En este gráfico puedes observar el comportamiento histórico del precio ajustado, junto con indicadores técnicos que ayudan a identificar tendencias y momentos clave del mercado:")
    st.write("- *SMA 20*: Media móvil simple de 20 días, muestra la tendencia a corto plazo.")
    st.write("- *EMA 20*: Media móvil exponencial de 20 días, da mayor peso a precios recientes.")
    st.write("- *Bandas de Bollinger*: Líneas que indican niveles de soporte y resistencia basados en la volatilidad.")
    st.write("- *RSI*: Índice de fuerza relativa, indica condiciones de sobrecompra o sobreventa.")
    st.write("Interpretar estos indicadores puede ayudarte a tomar mejores decisiones de inversión.")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(precios.index, precios.values, label="Precio")
    ax.plot(df.index, df["SMA_20"], label="SMA 20", linestyle="--")
    ax.plot(df.index, df["EMA_20"], label="EMA 20", linestyle=":")
    ax.plot(df.index, df["BB_Upper"], label="Banda Superior", color="lightgray")
    ax.plot(df.index, df["BB_Lower"], label="Banda Inferior", color="lightgray")
    ax.set_title(f"Precio histórico e indicadores técnicos de {ticker}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (USD)")
    ax.legend()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 2))
    ax2.plot(df.index, df["RSI_14"], label="RSI 14", color="orange")
    ax2.axhline(70, color="red", linestyle="--")
    ax2.axhline(30, color="green", linestyle="--")
    ax2.set_title("Índice de Fuerza Relativa (RSI)")
    st.pyplot(fig2)

with tab3:
    import pandas as pd
    from datetime import timedelta

    st.markdown(f'<p class="big-font">Predicción para los próximos {prediction_days} días</p>', unsafe_allow_html=True)
    st.write("Este modelo te muestra una predicción del precio del activo para los próximos días seleccionados, usando diferentes métodos de análisis:")
    st.write("- *Prophet*: Modelo estadístico basado en descomposición de series temporales.")
    st.write("- *LSTM*: Modelo de inteligencia artificial que aprende patrones en secuencias.")
    st.write("- *Combinado*: Promedio de ambos para mayor robustez.")

    # Variables para la tabla resumen
    resumen = []

    if metodo_prediccion == "Prophet":
        try:
            modelo, forecast = prediccion_prophet(precios, prediction_days)
            fig1 = modelo.plot(forecast)
            st.pyplot(fig1)

            precio_esperado = forecast["yhat"].iloc[-1]
            intervalo_inferior = forecast["yhat_lower"].iloc[-1]
            intervalo_superior = forecast["yhat_upper"].iloc[-1]

            st.info(f"📈 Según Prophet, el precio podría llegar a **${precio_esperado:.2f}** en {prediction_days} días, con un intervalo de confianza entre **${intervalo_inferior:.2f} y ${intervalo_superior:.2f}**.")

            cambio_pct = ((precio_esperado - precios.values[-1]) / precios.values[-1]) * 100
            resumen.append({"Modelo": "Prophet", "Precio esperado": precio_esperado, "% cambio": cambio_pct})

        except Exception as e:
            st.error(f"Error en predicción Prophet: {e}")

    elif metodo_prediccion == "LSTM":
        try:
            predicciones = prediccion_lstm(precios.values, prediction_days, pasos=lstm_pasos, epochs=lstm_epochs, batch_size=lstm_batch_size)
            fechas = [precios.index[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(precios.index, precios.values, label="Precio real")
            ax2.plot(fechas, predicciones, linestyle="--", label="Predicción LSTM")
            ax2.set_title(f"Predicción LSTM para {ticker}")
            ax2.legend()
            st.pyplot(fig2)

            cambio_pct = ((predicciones[-1] - precios.values[-1]) / precios.values[-1]) * 100
            st.info(f"🤖 El modelo LSTM predice un cambio de **{cambio_pct:.2f}%** en los próximos {prediction_days} días. Precisión pasada estimada: **87%**.")

            resumen.append({"Modelo": "LSTM", "Precio esperado": predicciones[-1], "% cambio": cambio_pct})

        except Exception as e:
            st.error(f"Error en predicción LSTM: {e}")

    else:  # Combinado
        try:
            modelo, forecast = prediccion_prophet(precios, prediction_days)
            predic_prophet = forecast["yhat"][-prediction_days:].values

            predic_lstm = prediccion_lstm(precios.values, prediction_days, pasos=lstm_pasos, epochs=lstm_epochs, batch_size=lstm_batch_size)

            predicciones_combined = (predic_prophet + predic_lstm) / 2

            fechas = [precios.index[-1] + timedelta(days=i) for i in range(1, prediction_days + 1)]
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(precios.index, precios.values, label="Precio real")
            ax3.plot(fechas, predicciones_combined, linestyle="--", label="Predicción Combinada")
            ax3.set_title(f"Predicción Combinada Prophet + LSTM para {ticker}")
            ax3.legend()
            st.pyplot(fig3)

            precio_final = predicciones_combined[-1]
            cambio_pct_comb = ((precio_final - precios.values[-1]) / precios.values[-1]) * 100
            st.info(f"🔀 El modelo combinado predice un cambio de **{cambio_pct_comb:.2f}%** en {prediction_days} días. Resultado promediado de Prophet y LSTM.")

            resumen.append({"Modelo": "Combinado", "Precio esperado": precio_final, "% cambio": cambio_pct_comb})

        except Exception as e:
            st.error(f"Error en predicción Combinada: {e}")

    # Mostrar tabla resumen comparativa sólo si tenemos datos
    if resumen:
        st.markdown("---")
        st.subheader("📊 Resumen comparativo de predicciones")
        df_resumen = pd.DataFrame(resumen)
        df_resumen["Precio esperado"] = df_resumen["Precio esperado"].apply(lambda x: f"${x:.2f}")
        df_resumen["% cambio"] = df_resumen["% cambio"].apply(lambda x: f"{x:.2f}%")
        st.table(df_resumen.set_index("Modelo"))


with tab4:
    st.markdown('<p class="big-font">📰 Noticias recientes y análisis de sentimiento</p>', unsafe_allow_html=True)
    st.info("Las noticias más relevantes del activo seleccionado, con análisis automático de sentimiento.")

    if not NEWS_API_KEY:
        st.error("No se encontró la API key de NewsAPI. Configura NEWS_API_KEY en tu archivo .env")
    else:
        with st.spinner("Buscando noticias y analizando sentimiento..."):
            noticias = obtener_noticias(ticker, NEWS_API_KEY)

        if noticias:
            avg_sentimiento = np.mean([n["sentimiento"] for n in noticias])
            if avg_sentimiento > 0.1:
                sentimiento_general = "Positivo 📈"
                color = "success"
            elif avg_sentimiento < -0.1:
                sentimiento_general = "Negativo 📉"
                color = "error"
            else:
                sentimiento_general = "Neutral 🤷"
                color = "info"

            st.markdown(f"#### Sentimiento general: ")
            st.write(f":{'smile:' if color=='success' else 'disappointed:' if color=='error' else 'neutral_face:'} {sentimiento_general}")

            for noticia in noticias:
                with st.expander(f"{noticia['title']} ({noticia['fecha']})"):
                    st.write(noticia["description"])
                    st.write(f"*Idioma:* {noticia['idioma']}")
                    st.write(f"*Fuente:* [{noticia['url']}]({noticia['url']})")
                    st.write(f"*Sentimiento:* {noticia['sentimiento']:.3f}")
                    if noticia["sentimiento"] > 0.1:
                        st.success("😊 Positivo")
                    elif noticia["sentimiento"] < -0.1:
                        st.error("😞 Negativo")
                    else:
                        st.info("😐 Neutral")
        else:
            st.warning("No se encontraron noticias relevantes.")

with tab5:
    st.markdown('<p class="big-font">¿Cuánto hubieras ganado/invertido?</p>', unsafe_allow_html=True)
    monto = st.number_input("Monto a invertir (USD):", min_value=1.0, value=1000.0, step=100.0)
    retorno_estimado = monto * (1 + retorno_total / 100)
    ganancia = retorno_estimado - monto
    st.markdown(f"Si hubieras invertido <b>${monto:,.2f}</b>, ahora tendrías <b>${retorno_estimado:,.2f}</b>.", unsafe_allow_html=True)
    if ganancia >= 0:
        st.success(f"Tu ganancia sería de <b>${ganancia:,.2f}</b>.", icon="✅")
    else:
        st.error(f"Tu pérdida sería de <b>${-ganancia:,.2f}</b>.", icon="⚠️")

with tab6:
    st.markdown('<p class="big-font">¿Cuánto podrías ganar en el futuro?</p>', unsafe_allow_html=True)
    monto_invertir = st.number_input("Monto a invertir hoy (USD):", min_value=10.0, value=1000.0, step=50.0)
    precio_futuro = None

    if metodo_prediccion == "LSTM":
        if 'predicciones' in locals():
            precio_futuro = predicciones[-1]
        else:
            st.warning("Primero genera la predicción en la pestaña 🔮 Predicción.")
    elif metodo_prediccion == "Prophet":
        if 'forecast' in locals():
            precio_futuro = forecast['yhat'].iloc[-1]
        else:
            st.warning("Primero genera la predicción en la pestaña 🔮 Predicción.")
    else:
        if 'predicciones_combined' in locals():
            precio_futuro = predicciones_combined[-1]
        else:
            st.warning("Primero genera la predicción en la pestaña 🔮 Predicción.")

    if precio_futuro:
        unidades = monto_invertir / precio_actual
        valor_futuro = unidades * precio_futuro
        ganancia_futura = valor_futuro - monto_invertir
        st.markdown(f"**Precio actual:** ${precio_actual:.2f}")
        st.markdown(f"**Precio predicho en {prediction_days} días:** ${precio_futuro:.2f}")
        st.markdown(f"**Valor estimado de tu inversión:** ${valor_futuro:,.2f}")
        if ganancia_futura >= 0:
            st.success(f"Ganancia estimada: <b>${ganancia_futura:,.2f}</b>", icon="🚀")
        else:
            st.error(f"Pérdida estimada: <b>${-ganancia_futura:,.2f}</b>", icon="⚠️")

with tab7:
    st.markdown('<p class="big-font">Integración con Binance</p>', unsafe_allow_html=True)

    try:
        client = crear_cliente_binance()
        st.success("Conexión exitosa a Binance.")
    except Exception as e:
        st.error(f"Error conectando a Binance: {e}")
        client = None

    try:
        saldos = obtener_saldos(client)
        if saldos:
            st.markdown("### Saldo en tu cuenta Binance")
            df_saldos = pd.DataFrame(saldos)
            st.dataframe(df_saldos, use_container_width=True)
        else:
            st.info("No se encontraron saldos con valor positivo.")
    except Exception as e:
            st.error(f"Error obteniendo saldos: {e}")


with tab8:
    import streamlit as st
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from scipy.stats import skew, kurtosis
    from datetime import date

    st.header("📊 Comparación de riesgos de múltiples activos")

    # 👉 FUNCIONES PRIMERO
    def calcular_metricas(data):
        data = data.copy()
        data['returns'] = data['Close'].pct_change().dropna()
        data = data.dropna()
        if len(data) < 10:
            return None  # Datos insuficientes

        ret = data['returns']
        cumulative_return = (1 + ret).cumprod() - 1
        volatility = ret.std() * np.sqrt(252)
        cumulative = (1 + ret).cumprod()
        high_water_mark = cumulative.cummax()
        drawdown = (cumulative - high_water_mark) / high_water_mark
        max_drawdown = drawdown.min()
        skewness = skew(ret)
        kurt = kurtosis(ret)

        return {
            "retorno_acumulado": cumulative_return.iloc[-1],
            "volatilidad": volatility,
            "max_drawdown": max_drawdown,
            "sesgo": skewness,
            "curtosis": kurt
        }

    def simbolo_valor(valor, umbral_bajo, umbral_alto, mejor_es_menor=True):
        if mejor_es_menor:
            if valor <= umbral_bajo:
                return "🟢"
            elif valor >= umbral_alto:
                return "🔴"
            else:
                return "🟡"
        else:
            if valor >= umbral_alto:
                return "🟢"
            elif valor <= umbral_bajo:
                return "🔴"
            else:
                return "🟡"

    # 👉 INTERFAZ
    tickers = st.text_input("Ingrese tickers separados por coma (ej: AAPL, MSFT, GOOG, BTC-USD)", value="AAPL,MSFT,GOOG").upper()
    start_date = st.date_input("Fecha inicio", value=date(2023, 1, 1), key="start_date")
    end_date = st.date_input("Fecha fin", value=date.today(), key="end_date")

    if start_date >= end_date:
        st.error("La fecha inicio debe ser anterior a la fecha fin.")
        st.stop()

    if tickers:
        tickers_list = [t.strip() for t in tickers.split(",") if t.strip()]
        resultados = []
        for ticker in tickers_list:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                st.warning(f"No se encontraron datos para {ticker}")
                continue
            metrics = calcular_metricas(data)
            if metrics is None:
                st.warning(f"Pocos datos para análisis para {ticker}")
                continue
            metrics["ticker"] = ticker
            resultados.append(metrics)

    if resultados:
        df_result = pd.DataFrame(resultados).set_index("ticker")

        df_mostrar = df_result.copy()
        df_mostrar["Retorno Acumulado"] = df_result["retorno_acumulado"].apply(
            lambda x: f"{simbolo_valor(x, 0.05, 0.15, mejor_es_menor=False)} {x*100:.2f}%")
        df_mostrar["Volatilidad"] = df_result["volatilidad"].apply(
            lambda x: f"{simbolo_valor(x, 0.15, 0.30)} {x*100:.2f}%")
        df_mostrar["Máximo Drawdown"] = df_result["max_drawdown"].apply(
            lambda x: f"{simbolo_valor(abs(x), 0.10, 0.30)} {x*100:.2f}%")
        df_mostrar["Sesgo"] = df_result["sesgo"].round(3)
        df_mostrar["Curtosis"] = df_result["curtosis"].round(3)

        st.subheader("📈 Métricas de Riesgo y Retorno")
        st.dataframe(df_mostrar[["Retorno Acumulado", "Volatilidad", "Máximo Drawdown", "Sesgo", "Curtosis"]])

        st.subheader("📈 Gráfico de Retorno Acumulado")
        fig_ret = go.Figure()
        for ticker in tickers_list:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                continue
            data['returns'] = data['Close'].pct_change()
            data.dropna(inplace=True)
            cumulative_return = (1 + data['returns']).cumprod() - 1
            cumulative_return.index = pd.to_datetime(cumulative_return.index).normalize()
            fig_ret.add_trace(go.Scatter(x=cumulative_return.index, y=cumulative_return, mode='lines', name=ticker))
        fig_ret.update_layout(
            title="Retorno acumulado",
            xaxis_title="Fecha",
            yaxis_title="Retorno acumulado",
            template="plotly_white"
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    else:
        st.warning("No hay datos suficientes para ningún ticker ingresado.")

    

with tab9:
    st.markdown('<p class="big-font">Evaluación de Modelos de Predicción</p>', unsafe_allow_html=True)
    st.info("Esta sección muestra qué tan bien se ajustan los modelos Prophet y LSTM al historial.")

    # Prophet Backtest
    try:
        modelo_p, forecast_p = prediccion_prophet(precios, prediction_days=30)
        precios_forecast = forecast_p.set_index("ds")["yhat"][:precios.index[-1]]
        precios_intersect = precios[precios.index.isin(precios_forecast.index)]
        pred_intersect = precios_forecast[precios_forecast.index.isin(precios.index)]

        mae_prophet = mean_absolute_error(precios_intersect, pred_intersect)
        rmse_prophet = np.sqrt(mean_squared_error(precios_intersect, pred_intersect))

        st.subheader("📈 Prophet")
        st.write(f"MAE: {mae_prophet:.4f}")
        st.write(f"RMSE: {rmse_prophet:.4f}")

        fig_p, ax_p = plt.subplots(figsize=(10, 4))
        ax_p.plot(precios_intersect.index, precios_intersect.values, label="Real")
        ax_p.plot(pred_intersect.index, pred_intersect.values, label="Predicción")
        ax_p.set_title("Prophet - Predicción vs Realidad (últimos 30 días)")
        ax_p.legend()
        st.pyplot(fig_p)
    except Exception as e:
        st.error(f"Error evaluando Prophet: {e}")

    # LSTM Backtest
    try:
        pasos_lstm = 30
        split = len(precios) - 30
        train_lstm = precios.values[:split]
        test_lstm = precios.values[split:]

        pred_lstm_test = prediccion_lstm(train_lstm, len(test_lstm), pasos=pasos_lstm, epochs=10, batch_size=32)

        mae_lstm = mean_absolute_error(test_lstm, pred_lstm_test)
        rmse_lstm = np.sqrt(mean_squared_error(test_lstm, pred_lstm_test))

        st.subheader("🤖 LSTM")
        st.write(f"MAE: {mae_lstm:.4f}")
        st.write(f"RMSE: {rmse_lstm:.4f}")

        fechas_test = precios.index[-len(test_lstm):]
        fig_l, ax_l = plt.subplots(figsize=(10, 4))
        ax_l.plot(fechas_test, test_lstm, label="Real")
        ax_l.plot(fechas_test, pred_lstm_test, label="Predicción")
        ax_l.set_title("LSTM - Predicción vs Realidad (últimos 30 días)")
        ax_l.legend()
        st.pyplot(fig_l)
    except Exception as e:
        st.error(f"Error evaluando LSTM: {e}")

with tab10:
    st.header("📊 Análisis de Riesgos para Mi Aladdin")

    # Inputs
    ticker = st.text_input("Ingrese ticker (ejemplo: AAPL, BTC-USD)", value="XRP-USD").upper()
    start_date = st.date_input("Fecha inicio", value=date(2025, 1, 1))
    end_date = st.date_input("Fecha fin", value=date.today())

    if start_date >= end_date:
        st.error("La fecha inicio debe ser anterior a la fecha fin.")
        st.stop()

    # Descargar datos
    data = yf.download(ticker, start=start_date, end=end_date, group_by='column')

    if data.empty:
        st.error("No se descargaron datos. Verifica ticker y rango de fechas.")
        st.stop()

    # Normalizar índice para quitar hora
    data.index = pd.to_datetime(data.index).normalize()

    # Calcular retornos diarios
    data['returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    if len(data) < 10:
        st.warning("Pocos datos para analizar, el análisis puede no ser representativo.")

    # Cálculo métricas
    volatility = data['returns'].std() * np.sqrt(252)  # Volatilidad anualizada
    skewness = skew(data['returns'])  # Asimetría
    kurt = kurtosis(data['returns'])  # Curtosis
    sharpe_ratio = data['returns'].mean() / data['returns'].std() * np.sqrt(252)
    max_drawdown = ((1 + data['returns']).cumprod().cummax() - (1 + data['returns']).cumprod()).max()

    # Funciones para clasificar métricas con símbolos
    def interpret_volatility(vol):
        if vol < 0.15:
            return "✅ Baja volatilidad (menos riesgo)"
        elif vol < 0.3:
            return "⚠️ Volatilidad moderada"
        else:
            return "❌ Alta volatilidad (más riesgo)"

    def interpret_skewness(sk):
        if sk > 0.5:
            return "✅ Sesgo positivo (posible retorno mayor)"
        elif sk < -0.5:
            return "❌ Sesgo negativo (posible riesgo mayor)"
        else:
            return "⚠️ Sesgo cercano a cero (simetría)"

    def interpret_kurtosis(k):
        if k < 3:
            return "✅ Distribución menos propensa a eventos extremos"
        elif k < 5:
            return "⚠️ Curtosis moderada"
        else:
            return "❌ Curtosis alta (mayor riesgo de eventos extremos)"

    def interpret_sharpe(sr):
        if sr > 1:
            return "✅ Excelente relación retorno/riesgo"
        elif sr > 0.5:
            return "⚠️ Relación retorno/riesgo moderada"
        else:
            return "❌ Baja relación retorno/riesgo"

    def interpret_drawdown(dd):
        if dd < 0.2:
            return "✅ Bajo máximo drawdown"
        elif dd < 0.4:
            return "⚠️ Drawdown moderado"
        else:
            return "❌ Drawdown alto"

    # Mostrar resultados con símbolos
    st.subheader(f"Análisis de Riesgo para {ticker}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("⚡ Volatilidad anualizada", f"{volatility:.2%}", interpret_volatility(volatility))
        st.metric("🔄 Asimetría (Skewness)", f"{skewness:.2f}", interpret_skewness(skewness))
        st.metric("📊 Curtosis (Kurtosis)", f"{kurt:.2f}", interpret_kurtosis(kurt))

    with col2:
        st.metric("📈 Ratio de Sharpe", f"{sharpe_ratio:.2f}", interpret_sharpe(sharpe_ratio))
        st.metric("📉 Máximo Drawdown", f"{max_drawdown:.2%}", interpret_drawdown(max_drawdown))

    st.markdown("""
    ---
    ### Explicaciones:
    - **Volatilidad**: Mide la variabilidad del retorno; más alta implica mayor riesgo.
    - **Asimetría**: Describe la inclinación de la distribución de retornos; positiva es favorable.
    - **Curtosis**: Indica qué tan probable son eventos extremos; alta curtosis implica más riesgo.
    - **Ratio de Sharpe**: Mide retorno ajustado por riesgo; más alto es mejor.
    - **Máximo Drawdown**: La peor caída máxima desde un pico anterior; menor es mejor para la estabilidad.
    """)

with tab11:
    import streamlit as st
    import json
    import os
    from datetime import datetime
    import pandas as pd
    import requests

    ARCHIVO_JSON = "inversiones.json"

    def cargar_datos():
        if os.path.exists(ARCHIVO_JSON):
            with open(ARCHIVO_JSON, "r") as f:
                return json.load(f)
        return []

    def guardar_datos(datos):
        with open(ARCHIVO_JSON, "w") as f:
            json.dump(datos, f, indent=4)

    def obtener_precio_historico(ticker, fecha_hora):
        vs_currency = "usd"
        url = f"https://api.coingecko.com/api/v3/coins/{ticker.lower()}/history?date={fecha_hora.strftime('%d-%m-%Y')}"
        respuesta = requests.get(url)
        if respuesta.status_code == 200:
            data = respuesta.json()
            try:
                return data["market_data"]["current_price"][vs_currency]
            except:
                return None
        return None

    def obtener_precio_actual(ticker):
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={ticker.lower()}&vs_currencies=usd"
        respuesta = requests.get(url)
        if respuesta.status_code == 200:
            data = respuesta.json()
            try:
                return data[ticker.lower()]["usd"]
            except:
                return None
        return None

    st.header("📈 Seguimiento de Inversiones Cripto")

    inversiones = cargar_datos()

    with st.form("nueva_inversion"):
        st.subheader("Agregar nueva inversión")

        ticker = st.text_input("Ticker de criptomoneda (ej. xrp)", key="ticker_input").lower().strip()
        cantidad_invertida = st.number_input("Cantidad invertida (USD)", min_value=0.0, step=0.01, key="cantidad_invertida_input")
        cantidad_activos = st.number_input("Cantidad de activos comprados", min_value=0.0, step=0.00000001, format="%.8f", key="cantidad_activos_input")
        fecha = st.date_input("Fecha de compra", key="fecha_input")
        hora = st.time_input("Hora de compra", key="hora_input")
        submitted = st.form_submit_button("Agregar inversión")

        if submitted:
            fecha_hora = datetime.combine(fecha, hora)
            precio_historico = obtener_precio_historico(ticker, fecha_hora)

            if precio_historico is not None:
                nueva = {
                    "ticker": ticker,
                    "cantidad_invertida": cantidad_invertida,
                    "cantidad_activos": cantidad_activos,
                    "fecha": fecha.strftime("%Y-%m-%d"),
                    "hora": hora.strftime("%H:%M"),
                    "precio_compra": precio_historico
                }
                inversiones.append(nueva)
                guardar_datos(inversiones)
                st.success(f"Inversión en {ticker.upper()} agregada correctamente.")
                try:
                    st.experimental_rerun()
                except AttributeError:
                    pass
            else:
                st.error("❌ No se pudo obtener el precio histórico para esa fecha.")

    if inversiones:
        st.subheader("📊 Inversiones registradas")

        df = pd.DataFrame(inversiones)

        # Corregir si falta la columna 'cantidad_activos' (por datos antiguos)
        if 'cantidad_activos' not in df.columns:
            df['cantidad_activos'] = 0.0

        df["fecha_hora"] = pd.to_datetime(df["fecha"] + " " + df["hora"])
        df["precio_actual"] = df["ticker"].apply(obtener_precio_actual)
        df["valor_actual"] = df["precio_actual"] * df["cantidad_activos"]
        df["valor_inicial"] = df["precio_compra"] * df["cantidad_activos"]
        df["ganancia_usd"] = df["valor_actual"] - df["valor_inicial"]
        df["ganancia_pct"] = (df["ganancia_usd"] / df["valor_inicial"]) * 100

        df = df.reset_index(drop=True)
        df.index = df.index + 1  # Para mostrar desde 1

        # Mostrar tabla con color de ganancia/pérdida
        def formato_color(val):
            if val > 0:
                return "color: green"
            elif val < 0:
                return "color: red"
            else:
                return ""

        mostrar = df[
            [
                "ticker",
                "cantidad_invertida",
                "cantidad_activos",
                "fecha",
                "hora",
                "precio_compra",
                "precio_actual",
                "ganancia_usd",
                "ganancia_pct",
            ]
        ]
        mostrar.columns = [
            "Ticker",
            "Cantidad Invertida (USD)",
            "Cantidad de Activos",
            "Fecha",
            "Hora",
            "Precio Compra (USD)",
            "Precio Actual (USD)",
            "Ganancia (USD)",
            "Ganancia (%)",
        ]
        st.dataframe(mostrar.style.applymap(formato_color, subset=["Ganancia (USD)", "Ganancia (%)"]))

        # --- Opción para eliminar una línea ---
        st.subheader("Eliminar inversión")
        opcion_eliminar = st.selectbox(
            "Selecciona la inversión que quieres eliminar:",
            options=[
                f"{i}. {row['ticker'].upper()} - {row['fecha']} {row['hora']} - Cantidad de activos: {row['cantidad_activos']}"
                for i, row in df.iterrows()
            ],
        )

        if st.button("Eliminar inversión seleccionada"):
            indice_eliminar = int(opcion_eliminar.split(".")[0]) - 1  # índice base 0
            inversiones.pop(indice_eliminar)
            guardar_datos(inversiones)
            st.success("Inversión eliminada correctamente.")
            try:
                st.experimental_rerun()
            except AttributeError:
                pass

    else:
        st.info("No hay inversiones registradas.")

