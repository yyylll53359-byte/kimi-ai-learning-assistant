import streamlit as st
from openai import OpenAI
import os
import json
from pathlib import Path
from dotenv import load_dotenv


# 外部 .env 文件的位置
env_path = Path.home() / "AI-Secrets" / ".env"

# 加载外部 .env
load_dotenv(dotenv_path=env_path)


# 读取 API Key
api_key = os.getenv("KIMI_API_KEY")

# 检查是否成功读取
if not api_key:
    st.error("没有读取到 KIMI_API_KEY，请检查 AI-Secrets 文件夹中的 .env")
    st.stop()


# 创建 Kimi 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1"
)


# =========================
# 2. 读取用户信息
# =========================

with open("memory.json", "r", encoding="utf-8") as f:
    user_memory = json.load(f)


# =========================
# 3. 创建系统Prompt
# =========================

system_prompt = f"""
你是一名耐心的AI学习导师。

用户信息：

姓名：{user_memory['name']}
专业：{user_memory['major']}
目标：{user_memory['goal']}

请根据用户背景提供个性化建议。
"""


# =========================
# 4. 页面设置
# =========================

st.title("🤖 AI学习助手")

st.write(
    f"你好 {user_memory['name']}，我是你的AI学习导师。"
)


# =========================
# 5. 加载历史聊天
# =========================

if "messages" not in st.session_state:

    try:

        with open(
            "chat_history.json",
            "r",
            encoding="utf-8"
        ) as f:

            history = json.load(f)

    except:

        history = []


    st.session_state.messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ] + history



# =========================
# 6. 清空聊天
# =========================

if st.button("🗑 清空聊天"):

    st.session_state.messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    with open(
        "chat_history.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            [],
            f,
            ensure_ascii=False,
            indent=4
        )

    st.success("聊天记录已清空")


# =========================
# 7. 显示历史消息
# =========================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.chat_message("user").write(
            message["content"]
        )


    elif message["role"] == "assistant":

        st.chat_message("assistant").write(
            message["content"]
        )



# =========================
# 8. 用户输入
# =========================

question = st.chat_input(
    "请输入你的问题..."
)


if question:


    # 保存用户问题

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # 请求Kimi

    response = client.chat.completions.create(

        model="moonshot-v1-8k",

        messages=st.session_state.messages

    )


    answer = response.choices[0].message.content



    # 保存AI回答

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # 写入长期记忆文件

    with open(
        "chat_history.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            st.session_state.messages[1:],
            f,
            ensure_ascii=False,
            indent=4
        )


    # 显示回答

    st.chat_message("assistant").write(
        answer
    )