import os
import json
import requests

def test_groq():
    api_key = os.environ.get("GROQ_API_KEY", "")
    print(f"Using API Key: {api_key[:5]}...{api_key[-5:]}")
    headlines = ["Bitcoin surges past 70k", "Ethereum ETF approved by SEC"]
    headline_block = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines[:15])])
    
    prompt = (
        "You are a quantitative financial analyst specializing in crypto markets. "
        "Score each headline from -1.0 (extremely bearish) to +1.0 (extremely bullish). "
        "UNDERSTAND CONTEXT: 'crash fears overblown' is BULLISH. 'new ATH but whales dumping' is MIXED/BEARISH. "
        "'SEC delays ETF' is BEARISH. 'regulation clarity' is BULLISH.\n\n"
        f"Headlines:\n{headline_block}\n\n"
        "Return ONLY valid JSON in this exact format, no other text:\n"
        '{"scores": [0.5, -0.3], "summary": "one sentence market narrative"}'
    )
    
    try:
        request_kwargs = {
            "timeout": 15.0,
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
                "max_tokens": 300,
            }
        }
        
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", **request_kwargs)
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print("Response:", response.text)
            return
            
        print("Response:", response.text)
        
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_groq()
