import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=600)
def descargar_datos(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            st.warning("No se encontraron datos para el símbolo y fechas indicados.")
        return data
    except Exception as e:
        st.error(f"Error al descargar datos de Yahoo Finance: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def descargar_datos_multiple(tickers, start, end):
    precios = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                precios[ticker] = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        except Exception as e:
            st.warning(f"Error descargando {ticker}: {e}")
    if not precios:
        st.warning("No se encontraron datos para los símbolos y fechas indicados.")
    return pd.DataFrame(precios)

def calcular_metricas_historicas(df_precios):
    retornos = df_precios.pct_change().dropna()
    retorno_medio = retornos.mean() * 252  # anualizado
    volatilidad = retornos.std() * (252 ** 0.5)
    return pd.DataFrame({'Retorno anual': retorno_medio, 'Volatilidad anual': volatilidad})

@st.cache_data(ttl=3600)
def obtener_info_fundamental(ticker):
    try:
        info = yf.Ticker(ticker).info
        campos = ['longName', 'sector', 'industry', 'marketCap', 'trailingPE', 'dividendYield']
        return {k: info.get(k, None) for k in campos}
    except Exception as e:
        st.warning(f"Error obteniendo información fundamental de {ticker}: {e}")
        return {}

import yfinance as yf

def obtener_fundamentales(ticker):
    """
    Obtiene datos fundamentales básicos del ticker usando yfinance.
    Retorna un diccionario con PE, EPS, Dividend Yield y Debt to Equity.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        pe_ratio = info.get('trailingPE', None)
        eps = info.get('trailingEps', None)
        dividend_yield = info.get('dividendYield', 0)  # Puede ser None o 0 si no paga dividendo
        debt_to_equity = info.get('debtToEquity', None)

        return {
            "PE": pe_ratio if pe_ratio else "N/A",
            "EPS": eps if eps else "N/A",
            "Dividend Yield": dividend_yield if dividend_yield else 0,
            "Debt to Equity": debt_to_equity if debt_to_equity else "N/A"
        }
    except Exception as e:
        raise RuntimeError(f"Error al obtener fundamentales: {e}")

def calcular_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcular_SMA(data, window):
    return data.rolling(window=window).mean()


def calcular_metricas(data):
    """
    Calcula métricas financieras básicas a partir de datos históricos.
    """
    data['daily_return'] = data['Close'].pct_change()

    volatilidad = data['daily_return'].std() * (252 ** 0.5) * 100  # Volatilidad anualizada
    retorno_total = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    retorno_mensual = data['daily_return'].mean() * 21 * 100  # Promedio de 21 días hábiles por mes
    sharpe_ratio = (data['daily_return'].mean() / data['daily_return'].std()) * (252 ** 0.5)

    return {
        "Volatilidad anual": volatilidad,
        "Retorno total (%)": retorno_total,
        "Retorno mensual (%)": retorno_mensual,
        "Sharpe Ratio": sharpe_ratio
    }
