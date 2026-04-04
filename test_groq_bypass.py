import requests
import json
import os

api_key = os.getenv("GROQ_API_KEY", "")

print("Test 1: Normal request (might use system proxy)")
try:
    resp1 = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "Test"}]},
        timeout=10
    )
    print(f"Normal: {resp1.status_code}")
except Exception as e:
    print(f"Normal Error: {e}")

print("\nTest 2: Trust_Env=False (ignore system proxy env vars)")
try:
    session = requests.Session()
    session.trust_env = False
    resp2 = session.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "Test"}]},
        timeout=10
    )
    print(f"Trust_Env: {resp2.status_code}")
except Exception as e:
    print(f"Trust_Env Error: {e}")

print("\nTest 3: Empty proxies dict")
try:
    resp3 = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "Test"}]},
        proxies={"http": "", "https": "", "no_proxy": "*"},
        timeout=10
    )
    print(f"Empty Proxies: {resp3.status_code}")
except Exception as e:
    print(f"Empty Proxies Error: {e}")
