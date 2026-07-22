import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. 读取外部 API Key
# =========================

env_path = Path.home() / "AI-Secrets" / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("KIMI_API_KEY")

if not api_key:
    raise RuntimeError("没有读取到 KIMI_API_KEY")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1"
)


# =========================
# 2. 读取 PDF
# =========================

pdf_path = Path(__file__).parent / "data" / "learning_material.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() or ""
    full_text += f"\n--- 第 {page_number} 页 ---\n"
    full_text += page_text

print(f"PDF读取成功，共 {len(reader.pages)} 页")


# =========================
# 3. 切分文本
# =========================

def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunks = split_text(full_text)

print(f"切分完成，共 {len(chunks)} 个文本块")


# =========================
# 4. 输入问题
# =========================

question = input("\n请输入问题：")


# =========================
# 5. 检索相关文本块
# =========================

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4)
)

all_texts = chunks + [question]

tfidf_matrix = vectorizer.fit_transform(all_texts)

question_vector = tfidf_matrix[-1]
chunk_vectors = tfidf_matrix[:-1]

scores = cosine_similarity(
    question_vector,
    chunk_vectors
).flatten()

top_indices = scores.argsort()[::-1][:3]

retrieved_chunks = [
    chunks[index]
    for index in top_indices
]

context = "\n\n".join(retrieved_chunks)


# =========================
# 6. 让 Kimi 根据资料回答
# =========================

response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {
            "role": "system",
            "content": (
                "你是一名文档问答助手。"
                "只能根据提供的资料回答问题。"
                "如果资料中没有答案，请明确回答："
                "当前文档中没有找到相关信息。"
            )
        },
        {
            "role": "user",
            "content": f"""
资料：
{context}

问题：
{question}

请给出简洁、准确的答案。
"""
        }
    ]
)

answer = response.choices[0].message.content


# =========================
# 7. 显示答案和来源
# =========================

print("\nAI回答：")
print(answer)

print("\n回答所依据的文本块：")

for rank, index in enumerate(top_indices, start=1):
    print(f"\n--- 来源 {rank}，相关度：{scores[index]:.3f} ---")
    print(chunks[index])