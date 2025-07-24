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
