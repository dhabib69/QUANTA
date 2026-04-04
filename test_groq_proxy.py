import requests
import json
import urllib3
urllib3.disable_warnings()
from quanta_proxy import ProxyManager
ProxyManager.set_proxy("http://127.0.0.1:62970")

import os
api_key = os.getenv("GROQ_API_KEY", "")
proxy_kwargs = ProxyManager.get_requests_kwargs()

print(f"Proxy config: {proxy_kwargs}")

request_kwargs = {
    "timeout": 15.0,
    "headers": {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    "json": {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": "Test"}
        ]
    }
}
request_kwargs.update(proxy_kwargs)

try:
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", **request_kwargs)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"Error: {e}")
