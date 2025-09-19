import requests
import json

def ollama_chat(prompt):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "qwen2:0.5b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        return f"错误: {e}"

# 测试
response = ollama_chat("你好，请介绍一下你自己")
print(response)
