import os
import requests
import certifi

key = os.getenv("LLM_API_KEY", "").strip()
url = "https://api.groq.com/openai/v1/chat/completions"
payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "hi"}],
    "temperature": 0
}

s = requests.Session()
s.trust_env = False

try:
    r = s.post(
        url,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Connection": "close",
        },
        json=payload,
        timeout=30,
        verify=certifi.where(),
    )
    print("status =", r.status_code)
    print("body =", r.text[:400])
except Exception as e:
    print("error =", type(e).__name__, e)
