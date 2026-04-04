from QUANTA_ai_oracle import get_groq_sentiment

headlines = [
    "Bitcoin surges past 70k",
    "Ethereum ETF approved by SEC",
    "Market crashes due to inflation fears",
    "Binance CEO steps down amid regulatory scrutiny"
]

result = get_groq_sentiment(headlines)
print("Result:", result)
