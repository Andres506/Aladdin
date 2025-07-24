import yfinance as yf
import numpy as np
import pandas as pd

def descargar_datos_multiple(tickers, start, end):
    precios = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end)
        # Usa 'Adj Close' si existe, si no, usa 'Close'
        if 'Adj Close' in df.columns:
            precios[ticker] = df['Adj Close']
        elif 'Close' in df.columns:
            precios[ticker] = df['Close']
        else:
            raise ValueError(f"No se encontró columna de precios para {ticker}")
    return pd.DataFrame(precios)

def calcular_retorno_covarianza(precios):
    """
    Recibe DataFrame de precios ajustados.
    Retorna retornos mensuales (DataFrame), promedio mensual (Series) y matriz de covarianza mensual (DataFrame).
    """
    # Retornos diarios
    retornos_diarios = precios.pct_change().dropna()
    # Retornos mensuales agrupados
    retornos_mensuales = retornos_diarios.resample('M').apply(lambda x: (1 + x).prod() - 1)
    # Promedio mensual
    ret_mean = retornos_mensuales.mean()
    # Covarianza mensual
    cov_mensual = retornos_mensuales.cov()
    return retornos_mensuales, ret_mean, cov_mensual