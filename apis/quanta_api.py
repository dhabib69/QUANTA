import os
from pathlib import Path

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, it will just use system env vars
    pass

# =================================================================
# QUANTA CENTRAL API CONFIGURATION
# =================================================================
# This file centralizes all API keys and environment variables.
# For security, do NOT hardcode keys here. Use a .env file.
# =================================================================

# --- LLM & AI ORACLE CONFIG ---
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.openai.com/v1")
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "claude-3-5-sonnet-20241022")

# Specific Provider Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GLM_API_KEY = os.getenv("GLM_API_KEY", "")

# --- TELEGRAM BOT CONFIG ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- NETWORK & PROXY ---
QUANTA_PROXY_PORT = os.getenv("QUANTA_PROXY_PORT", "52681")

# --- EXCHANGE API KEYS ---
# Binance
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Bybit
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# Hyperliquid
HYPERLIQUID_PRIVATE_KEY = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")

def validate_connectivity():
    """Check if basic API configuration is present."""
    required = {
        "BINANCE_API_KEY": BINANCE_API_KEY,
        "TELEGRAM_TOKEN": TELEGRAM_TOKEN,
        "GROQ_API_KEY": GROQ_API_KEY
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"⚠️  Warning: Missing keys in .env: {', '.join(missing)}")
    else:
        print("✅ QUANTA API Configuration Loaded Successfully.")

if __name__ == "__main__":
    validate_connectivity()
