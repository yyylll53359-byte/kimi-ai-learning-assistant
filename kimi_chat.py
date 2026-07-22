from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv


env_path = Path.home() / "AI-Secrets" / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("KIMI_API_KEY")

if not api_key:
    raise ValueError("没有读取到 KIMI_API_KEY")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1"
)


question = input("请输入你的问题：")


response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {
            "role": "user",
            "content": question
        }
    ]
)


print("AI回答：")
print(response.choices[0].message.content)