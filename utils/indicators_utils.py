import pandas as pd

def agregar_indicadores_tecnicos(df, price_col="Adj Close"):
    """
    Añade indicadores técnicos comunes al DataFrame:
    - SMA 20
    - EMA 20
    - RSI 14
    - MACD y MACD Signal
    - Bandas de Bollinger (Middle, Upper, Lower)
    
    Parámetros:
        df: pd.DataFrame con datos históricos que incluyen columna de precios.
        price_col: str, nombre de la columna con precios (por defecto "Adj Close")
        
    Retorna:
        df modificado con nuevas columnas de indicadores técnicos.
    """

    # Validar que la columna price_col exista en df
    if price_col not in df.columns:
        raise ValueError(f"La columna '{price_col}' no existe en el DataFrame.")

    # SMA 20
    df["SMA_20"] = df[price_col].rolling(window=20).mean()

    # EMA 20
    df["EMA_20"] = df[price_col].ewm(span=20, adjust=False).mean()

    # RSI 14
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD y MACD Signal
    ema_12 = df[price_col].ewm(span=12, adjust=False).mean()
    ema_26 = df[price_col].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bandas de Bollinger
    df["BB_Middle"] = df[price_col].rolling(window=20).mean()
    df["BB_Std"] = df[price_col].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_Std"]

    return df
