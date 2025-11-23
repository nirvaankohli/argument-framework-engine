from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv
import os
import requests


ROOT_DIR = Path(__file__).parent.parent

env_path = Path(ROOT_DIR) / ".env"

load_dotenv(env_path)

api_key = os.getenv("API_KEY", "")
api_key = api_key
api_url = "https://ai.hackclub.com/proxy/v1/chat/completions"
model = "openai/gpt-5-mini"
timeout = 10

payload = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions concisely.",
        },
        {"role": "user", "content": "What is the capital of France?"},
    ],
    "temperature": 0.3,
    "max_tokens": 4000,
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

r = requests.post(url=api_url, json=payload, headers=headers, timeout=timeout)

print(r.status_code)
print(r.json()["choices"][0]["message"]["content"])