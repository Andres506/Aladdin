import requests
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analizar_sentimiento_textblob(texto):
    tb = TextBlob(texto)
    return tb.sentiment.polarity

def analizar_sentimiento_vader(texto):
    vs = sia.polarity_scores(texto)
    return vs["compound"]

def obtener_noticias(ticker, api_key, idiomas=["es", "en"], max_articulos=10):
    noticias = []
    for lang in idiomas:
        url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={api_key}&language={lang}&pageSize={max_articulos}"
        response = requests.get(url)
        if response.status_code == 200:
            articulos = response.json().get("articles", [])
            for art in articulos:
                texto = (art.get("title") or "") + " " + (art.get("description") or "")
                fecha_pub = art.get("publishedAt", "")[:10]
                if not texto.strip():
                    continue
                if lang == "es":
                    polaridad = analizar_sentimiento_textblob(texto)
                else:
                    polaridad = analizar_sentimiento_vader(texto)
                noticias.append({
                    "title": art.get("title"),
                    "description": art.get("description"),
                    "url": art.get("url"),
                    "fecha": fecha_pub,
                    "sentimiento": polaridad,
                    "idioma": lang
                })
    return noticias
