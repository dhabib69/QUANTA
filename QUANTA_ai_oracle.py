import os
try:
    from apis.quanta_api import GROQ_API_KEY, GEMINI_API_KEY
except ImportError:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
import json
import requests
import logging


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GROQ LLM SENTIMENT ENGINE (Llama 3.3 70B — Free Tier)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14,400 free calls/day | 30 RPM | ~200ms inference (LPU)
# Replaces L&M 2011 word-counting with full contextual understanding.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_groq_sentiment(headlines: list, active_coins: list = None) -> dict:
    """
    Batch-analyze crypto headlines using Groq/Llama 3.3 70B for both global + per-coin sentiment.
    
    Returns:
        {
            'score': float,          # -1.0 to +1.0 (aggregate sentiment)
            'summary': str,          # 1-sentence market narrative
            'per_headline': list,    # [{title, score}]
            'coin_sentiment': dict   # {'BTC': 0.8, 'ETH': 0.2, ...} (-1.0 to 1.0)
        }
    Returns None on failure.
    """
    api_key = GROQ_API_KEY
    if not api_key:
        logging.warning("🧠 Groq: API Key missing or empty. Falling back to L&M lexicon.")
        return None
    
    if not headlines:
        print("🧠 Groq Error: Headlines list is empty!")
        return None
        
    active_coins = active_coins or []
    # Take up to 20 coins to keep prompt reasonably sized
    coins_str = ", ".join(active_coins[:20]) if active_coins else "None specific"
    
    # Build the batch prompt — force JSON output for deterministic parsing
    headline_block = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines[:15])])
    
    prompt = (
        "You are a quantitative financial analyst specializing in crypto markets. "
        "Score each headline from -1.0 (extremely bearish) to +1.0 (extremely bullish). "
        "Also provide an aggregate sentiment score for specific coins mentioned in these headlines, "
        "based ONLY on the provided text. If a coin is not mentioned or implied, score it 0.0.\n\n"
        f"Coins to track: {coins_str}\n\n"
        f"Headlines:\n{headline_block}\n\n"
        "Return ONLY valid JSON in this exact format, no other text:\n"
        '{"scores": [0.5, -0.3, ...], "summary": "one sentence market narrative", "coin_sentiment": {"BTC": 0.5, "ETH": -0.2}}'
    )
    
    try:
        # Bypass proxy to avoid Cloudflare/Groq blocking Psiphon 3 IPs
        request_kwargs = {
            "timeout": 15.0,
            "proxies": {
                "http": None,
                "https": None
            },
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            "json": {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a financial sentiment analyst. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500,
            }
        }
        
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", **request_kwargs)
        
        if response.status_code != 200:
            logging.debug(f"Groq API error: {response.status_code} {response.text}")
            print(f"🧠 Groq API error status: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        text = data['choices'][0]['message']['content'].strip()
        
        # Parse JSON from response (handle markdown code blocks)
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
        
        result = json.loads(text)
        scores = result.get('scores', [])
        summary = result.get('summary', '')
        coin_sentiment = result.get('coin_sentiment', {})
        
        # Clean up coin sentiment dict
        clean_coin_sentiment = {}
        if isinstance(coin_sentiment, dict):
            for k, v in coin_sentiment.items():
                try:
                    score = float(v)
                    if score != 0.0:
                        clean_coin_sentiment[str(k).upper()] = max(-1.0, min(1.0, score))
                except (ValueError, TypeError):
                    pass
        
        # Build per-headline mapping
        per_headline = []
        for i, h in enumerate(headlines[:15]):
            score = float(scores[i]) if i < len(scores) else 0.0
            score = max(-1.0, min(1.0, score))  # Clamp
            per_headline.append({'title': h, 'score': score})
        
        # Aggregate score
        if per_headline:
            agg_score = sum(p['score'] for p in per_headline) / len(per_headline)
        else:
            agg_score = 0.0
        
        return {
            'score': max(-1.0, min(1.0, agg_score)),
            'summary': summary,
            'per_headline': per_headline,
            'coin_sentiment': clean_coin_sentiment
        }
        
    except json.JSONDecodeError as e:
        logging.debug("Groq: Failed to parse JSON response")
        print(f"🧠 Groq JSON Decode Error: {e} \nRaw text was: {text}")
        return None
    except Exception as e:
        logging.debug(f"Groq sentiment failed: {e}")
        print(f"🧠 Groq sentiment exception: {e}")
        return None

def get_oracle_summary(direction: str, symbol: str, headlines: list) -> str:
    """
    Queries Google Gemini 1.5 Flash to generate a 1-sentence contextual summary
    based on the latest crypto news headlines. Non-blocking with strict timeout.
    """
    api_key = GEMINI_API_KEY
    if not api_key:
        logging.debug("🧠 Gemini: API key missing.")
        return ""

    if not headlines:
        headline_text = "No recent headlines available."
    else:
        headline_text = "\n".join([f"- {h}" for h in headlines[:5]])

    prompt = (
        f"The algorithmic trading bot just triggered a high-confidence {direction} signal for {symbol}. "
        f"Here are the latest crypto headlines:\n{headline_text}\n"
        "Considering the algorithmic signal and these headlines, provide a single-sentence summary of the macro narrative and whether it aligns with the signal. Keep it very concise."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 60
        }
    }

    try:
        # Bypass proxy to avoid Cloudflare/Gemini blocking Psiphon 3 IPs
        request_kwargs = {"json": payload, "timeout": 3.5, "proxies": {"http": None, "https": None}}

        response = requests.post(url, **request_kwargs)
        if response.status_code == 200:
            data = response.json()
            try:
                summary = data['candidates'][0]['content']['parts'][0]['text']
                return summary.strip().replace('\n', ' ')
            except (KeyError, IndexError):
                return ""
        else:
            logging.debug(f"Oracle API error: {response.status_code} {response.text}")
            return ""
    except Exception as e:
        logging.debug(f"Oracle request failed: {e}")
        return ""

def ask_oracle_chat(question: str, headlines: list = None) -> str:
    """
    General purpose Oracle query for Telegram commands (e.g. /ask BTC)
    """
    api_key = GEMINI_API_KEY
    if not api_key:
        return "⚠️ Gemini API key missing."
        
    headline_text = ""
    if headlines:
        headline_text = "Here are the 10 most recent crypto news headlines pulled from RSS:\n"
        headline_text += "\n".join([f"- {h}" for h in headlines]) + "\n\n"
        
    prompt = (
        f"You are the Oracle of the QUANTA Trading Bot. The user is asking you about the crypto market.\n"
        f"{headline_text}"
        f"User question: {question}\n\n"
        f"Respond in 3 concise sentences or less. Be professional, analytical, and highly direct."
    )
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 150
        }
    }
    
    try:
        # Bypass proxy to avoid Cloudflare/Gemini blocking Psiphon 3 IPs
        request_kwargs = {"json": payload, "timeout": 8.0, "proxies": {"http": None, "https": None}}

        response = requests.post(url, **request_kwargs)
        if response.status_code == 200:
            data = response.json()
            try:
                summary = data['candidates'][0]['content']['parts'][0]['text']
                return summary.strip()
            except (KeyError, IndexError):
                return "⚠️ Failed to parse Oracle response."
        else:
            return f"⚠️ Oracle API error: {response.status_code}"
    except Exception as e:
        return f"⚠️ Oracle request timeout or failure."
