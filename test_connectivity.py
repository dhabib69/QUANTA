import os
import requests
import json
import logging
from pathlib import Path

# Try to import central config
try:
    from apis.quanta_api import (
        BINANCE_API_KEY, BINANCE_API_SECRET,
        BYBIT_API_KEY, BYBIT_API_SECRET,
        HYPERLIQUID_PRIVATE_KEY,
        TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
        AI_API_KEY, AI_BASE_URL,
        GROQ_API_KEY, GEMINI_API_KEY
    )
except ImportError:
    print("❌ Could not import apis.quanta_api. Ensure .env exists or apis/quanta_api.py is correct.")
    # Fallback to env vars for direct testing
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
    BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
    HYPERLIQUID_PRIVATE_KEY = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    AI_API_KEY = os.getenv("AI_API_KEY", "")
    AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.openai.com/v1")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def test_binance():
    print("Testing Binance Connectivity...")
    if not BINANCE_API_KEY: return "❌ Missing Key"
    # Placeholder for real ping (needs hmac signing for account info, using public ping instead)
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/ping", timeout=5)
        if r.status_code == 200: return "✅ Connected (API Reachable)"
    except Exception as e: return f"❌ Error: {e}"
    return "❌ Failed"

def test_telegram():
    print("Testing Telegram Bot...")
    if not TELEGRAM_TOKEN: return "❌ Missing Token"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return f"✅ Connected (@{data['result']['username']})"
    except Exception as e: return f"❌ Error: {e}"
    return f"❌ Failed (Status {r.status_code})"

def test_groq():
    print("Testing Groq Sentiment Engine...")
    if not GROQ_API_KEY: return "❌ Missing Key"
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200: return "✅ Connected (Key Valid)"
    except Exception as e: return f"❌ Error: {e}"
    return f"❌ Failed (Status {r.status_code})"

def test_gemini():
    print("Testing Gemini AI Oracle...")
    if not GEMINI_API_KEY: return "❌ Missing Key"
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200: return "✅ Connected (Key Valid)"
    except Exception as e: return f"❌ Error: {e}"
    return f"❌ Failed (Status {r.status_code})"

def test_zeus():
    print("Testing ZEUS.ai (GLM 5.1/4)...")
    if not AI_API_KEY: return "❌ Missing Key"
    # Zhipu AI / GLM OpenAI-compatible endpoint check
    try:
        url = AI_BASE_URL
        headers = {"Authorization": f"Bearer {AI_API_KEY}"}
        data = {
            "model": "glm-4",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5
        }
        r = requests.post(url, headers=headers, json=data, timeout=10)
        if r.status_code == 200: return "✅ Connected (GLM-4 Valid)"
        return f"❌ Failed (Status {r.status_code}: {r.text[:50]}...)"
    except Exception as e: return f"❌ Error: {e}"

def test_glm():
    print("Testing GLM (Zhipu AI / ChatGLM)...")
    if not GLM_API_KEY: return "❌ Missing Key"
    # Zhipu AI usually requires a specific JWT or SDK, but we can do a simple GET check for now if they have a status endpoint
    # For now, let's just check if it's correctly loaded in the script
    return "✅ Key Loaded (Diagnostics Pending)"

def run_all():
    print("\n" + "="*40)
    print("   QUANTA API CONNECTIVITY REPORT (LITE)")
    print("="*40)
    
    results = {}
    if BINANCE_API_KEY: results["Binance"] = test_binance()
    if TELEGRAM_TOKEN: results["Telegram"] = test_telegram()
    if GROQ_API_KEY: results["Groq"] = test_groq()
    if GEMINI_API_KEY: results["Gemini"] = test_gemini()
    if AI_API_KEY: results["ZEUS.ai"] = test_zeus()
    
    if not results:
        print("⚠️ No keys found in .env to test.")
    else:
        for api, res in results.items():
            print(f"{api:15}: {res}")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_all()
