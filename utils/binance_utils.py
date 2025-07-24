from binance.client import Client
import os
from dotenv import load_dotenv

load_dotenv()

def crear_cliente_binance():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("Claves API de Binance no encontradas en .env")
    return Client(api_key, api_secret)

def obtener_precio_actual(client, symbol):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return ticker["price"]

def obtener_saldos(client):
    account_info = client.get_account()
    return [b for b in account_info['balances'] if float(b['free']) > 0]

def crear_orden_compra(client, symbol, cantidad):
    orden = client.create_order(
        symbol=symbol,
        side=Client.SIDE_BUY,
        type=Client.ORDER_TYPE_MARKET,
        quantity=round(cantidad, 4)
    )
    return orden
