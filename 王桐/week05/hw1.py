from openai import OpenAI

client = OpenAI(
    api_key="sk-a0bf1dd5942540908257f0a7b79f9a3b", 
    base_url="https://api.deepseek.com/v1",  
)

response = client.chat.completions.create(
    model="deepseek-chat",  
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好！请介绍一下你自己。"},
    ],
    stream=False,  # 流式响应
)

print(response.choices[0].message.content)
