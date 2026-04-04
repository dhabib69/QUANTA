"""Live Groq Sentiment Pipeline Validation"""
import sys
import time
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Set proxy BEFORE any imports that use NetworkHelper
import QUANTA_network
QUANTA_network._PROXY_URL = "http://127.0.0.1:62970"
print(f"✅ Proxy set: {QUANTA_network._PROXY_URL}")

# Force reset the session so it picks up the proxy
QUANTA_network.NetworkHelper.reset_session()
print("✅ NetworkHelper session reset with proxy")

# Test 1: Can we import get_groq_sentiment?
print("\n━━━ TEST 1: Import Check ━━━")
try:
    from QUANTA_ai_oracle import get_groq_sentiment
    print(f"✅ get_groq_sentiment imported: {get_groq_sentiment}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Direct Groq API call
print("\n━━━ TEST 2: Direct Groq API Call ━━━")
test_headlines = [
    "Bitcoin surges past $90K as institutional demand accelerates",
    "Fed signals potential rate cuts in Q2 2026",
    "Ethereum whale dumps 50K ETH on Binance",
]
result = get_groq_sentiment(test_headlines)
if result:
    print(f"✅ Groq returned: score={result['score']:+.3f}")
    print(f"   Summary: {result['summary']}")
    for ph in result['per_headline']:
        print(f"   {ph['score']:+.2f} | {ph['title']}")
else:
    print("❌ Groq returned None")

# Test 3: Full SentimentEngine pipeline
print("\n━━━ TEST 3: Full SentimentEngine Pipeline ━━━")
from QUANTA_sentiment import SentimentEngine

class FakeCfg:
    fear_greed_enabled = True
    
se = SentimentEngine()

print("  Fetching RSS headlines...")
se._refresh_rss()
print(f"  Headlines in buffer: {len(se._headlines)}")

if se._headlines:
    print(f"  Top 3 headlines:")
    for h in se._headlines[:3]:
        print(f"    [{h['source']}] {h['title'][:60]}")

print("\n  Forcing Groq deep score...")
se._last_groq_call = 0  # Reset cooldown
se._groq_deep_score()

print(f"\n  Groq score: {se._groq_score}")
print(f"  Groq summary: {se._groq_summary}")

print("\n━━━ TEST 4: Telegram /sentiment Preview ━━━")
print(se.get_summary())

print("\n━━━ TEST 5: ML Feature Vector ━━━")
features = se.get_sentiment_features()
print(f"  Features: {features}")
print(f"  news_score (idx 3) = {features[3]:+.3f} {'(GROQ)' if se._groq_score is not None else '(L&M)'}")

print("\n✅ ALL TESTS COMPLETE")
