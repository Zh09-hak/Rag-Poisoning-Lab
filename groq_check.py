import os
import sys
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.groq.com/openai/v1"
key = os.getenv("GROQ_API_KEY", "").strip()
model = sys.argv[1].strip() if len(sys.argv) > 1 else os.getenv("GROQ_MODEL", "").strip()

def fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()[:12] if value else "missing"

print("GROQ DIAGNOSTIC")
print("key_present:", bool(key))
print("key_length:", len(key))
print("key_fingerprint:", fingerprint(key))
print("requested_model:", model or "not_set")

if not key:
    print("ERROR: GROQ_API_KEY is missing")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}

try:
    response = requests.get(f"{BASE}/models", headers=headers, timeout=30)
    print("models_status:", response.status_code)

    if response.status_code != 200:
        print(response.text)
        sys.exit(1)

    payload = response.json()
    models = sorted([m.get("id") for m in payload.get("data", []) if m.get("id")])
    print("models_count:", len(models))

    for item in models:
        print("model:", item)

    if not model:
        model = models[0] if models else ""
        print("auto_selected_model:", model)

    if not model:
        print("ERROR: no available models")
        sys.exit(1)

    if model not in models:
        print("WARNING: requested model is not in the available models list")

    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply with the single word OK"}
        ],
        "temperature": 0,
        "max_tokens": 16
    }

    chat = requests.post(f"{BASE}/chat/completions", headers=headers, json=body, timeout=60)
    print("chat_status:", chat.status_code)
    print("chat_model:", model)

    if chat.status_code == 200:
        data = chat.json()
        print("chat_reply:", data["choices"][0]["message"]["content"])
    else:
        print(chat.text)

    try:
        from langchain_groq import ChatGroq

        try:
            llm = ChatGroq(model=model, temperature=0, api_key=key)
        except TypeError:
            llm = ChatGroq(model_name=model, temperature=0, groq_api_key=key)

        result = llm.invoke("Reply with the single word OK")
        print("langchain_status: ok")
        print("langchain_reply:", result.content)
    except ImportError as e:
        print("langchain_status: missing_dependency")
        print(str(e))
    except Exception as e:
        print("langchain_status: error")
        print(type(e).__name__, str(e))

except requests.RequestException as e:
    print("network_error:", type(e).__name__, str(e))
    sys.exit(1)