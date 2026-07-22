from openai import OpenAI
import json
import os
from dotenv import load_dotenv


load_dotenv()


client = OpenAI(
    api_key=os.getenv("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)


with open("memory.json", "r", encoding="utf-8") as f:
    user_memory = json.load(f)


messages = [
    {
        "role": "system",
        "content": f"""
你是一名耐心的AI学习导师。

用户信息：
姓名：{user_memory['name']}
专业：{user_memory['major']}
目标：{user_memory['goal']}

请根据用户背景回答问题。
"""
    }
]


while True:

    question = input("\n你：")

    if question == "退出":
        break


    messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=messages
    )


    answer = response.choices[0].message.content


    print("\nAI：")
    print(answer)


    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )