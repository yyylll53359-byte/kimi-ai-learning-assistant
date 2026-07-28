import hashlib
import io
import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="PDF 知识库助手",
    page_icon="📄",
)

st.title("📄 PDF 知识库助手")
st.write("上传一份 PDF，系统会检索相关内容，并让 Kimi 只根据文档回答。")


# =========================
# 1. RAG 参数
# =========================

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_COUNT = 3
MAX_DISTANCE = 1.25


# =========================
# 2. 加载外部 API Key
# =========================

env_path = Path.home() / "AI-Secrets" / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("KIMI_API_KEY")

if not api_key:
    st.error("没有读取到 KIMI_API_KEY，请检查 AI-Secrets 文件夹中的 .env。")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1",
)


# =========================
# 3. 加载嵌入模型和 Chroma
# =========================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_resource
def load_chroma_collection():
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    return chroma_client.get_or_create_collection(
        name="pdf_knowledge_base"
    )


embedding_model = load_embedding_model()
collection = load_chroma_collection()


# =========================
# 4. 文本处理函数
# =========================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


@st.cache_data(show_spinner=False)
def create_embeddings(text_chunks):
    vectors = embedding_model.encode(
        text_chunks,
        normalize_embeddings=True,
    )
    return vectors.tolist()


# =========================
# 5. 上传并读取 PDF
# =========================

uploaded_file = st.file_uploader(
    "上传 PDF 文件",
    type=["pdf"],
)

if uploaded_file is None:
    st.info("请先上传一份 PDF。")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_id = hashlib.sha256(file_bytes).hexdigest()

try:
    reader = PdfReader(io.BytesIO(file_bytes))
except Exception as error:
    st.error(f"PDF 读取失败：{error}")
    st.stop()


# =========================
# 6. 按页提取并切分文本
# =========================

full_text_parts = []
chunk_records = []
pages_with_text = 0

for page_number, page in enumerate(reader.pages, start=1):
    page_text = (page.extract_text() or "").strip()

    if not page_text:
        continue

    pages_with_text += 1
    full_text_parts.append(
        f"--- 第 {page_number} 页 ---\n{page_text}"
    )

    page_chunks = split_text(page_text)

    for page_chunk_number, chunk in enumerate(
        page_chunks,
        start=1,
    ):
        chunk_records.append(
            {
                "text": chunk,
                "page_number": page_number,
                "page_chunk_number": page_chunk_number,
            }
        )

full_text = "\n\n".join(full_text_parts)
chunks = [record["text"] for record in chunk_records]

if not chunks:
    st.error("没有提取到可用文字。若 PDF 是扫描件，需要先进行 OCR。")
    st.stop()

st.success(
    f"读取成功：共 {len(reader.pages)} 页，"
    f"其中 {pages_with_text} 页提取到文字，"
    f"共 {len(full_text)} 个字符，切分为 {len(chunks)} 个文本块。"
)

with st.expander("查看提取的部分文字"):
    st.write(full_text[:3000])


# =========================
# 7. 建立当前 PDF 的向量知识库
# =========================

with st.spinner("正在建立向量知识库..."):
    chunk_embeddings = create_embeddings(chunks)

    # 删除这个 PDF 以前的旧记录，防止旧数据没有页码。
    collection.delete(
        where={"file_id": file_id}
    )

    chunk_ids = []
    metadatas = []

    for global_index, record in enumerate(
        chunk_records,
        start=1,
    ):
        page_number = record["page_number"]
        page_chunk_number = record["page_chunk_number"]

        chunk_ids.append(
            f"{file_id}_page_{page_number}_chunk_{page_chunk_number}"
        )

        metadatas.append(
            {
                "file_id": file_id,
                "file_name": uploaded_file.name,
                "page_number": int(page_number),
                "page_chunk_number": int(page_chunk_number),
                "chunk_number": int(global_index),
            }
        )

    collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        embeddings=chunk_embeddings,
        metadatas=metadatas,
    )

st.info(f"知识库已建立：当前 PDF 共保存 {len(chunks)} 个向量。")


# =========================
# 8. 用户提问和语义检索
# =========================

question = st.chat_input("请输入一个与 PDF 内容有关的问题...")

if question:
    st.chat_message("user").write(question)

    query_embedding = create_embeddings([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_COUNT, len(chunks)),
        where={"file_id": file_id},
        include=["documents", "distances", "metadatas"],
    )

    retrieved_chunks = results["documents"][0]
    retrieved_distances = results["distances"][0]
    retrieved_metadatas = results["metadatas"][0]

    if not retrieved_chunks:
        st.chat_message("assistant").write(
            "当前文档中没有找到相关信息。"
        )
        st.stop()

    best_distance = retrieved_distances[0]

    # 距离越小，代表问题与文本越相似。
    if best_distance > MAX_DISTANCE:
        st.chat_message("assistant").write(
            "当前文档中没有找到相关信息。"
        )
        st.warning(
            f"最相关文本的向量距离为 {best_distance:.3f}，"
            f"超过拒答阈值 {MAX_DISTANCE:.2f}。"
        )
        st.stop()


    # =========================
    # 9. 组装带页码的上下文
    # =========================

    context_parts = []

    for source_number, (document, metadata) in enumerate(
        zip(retrieved_chunks, retrieved_metadatas),
        start=1,
    ):
        page_number = metadata.get("page_number", "未知")

        context_parts.append(
            f"资料 {source_number}（PDF 第 {page_number} 页）：\n"
            f"{document}"
        )

    context = "\n\n".join(context_parts)


    # =========================
    # 10. 调用 Kimi 生成答案
    # =========================

    with st.spinner("正在根据文档生成回答..."):
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一名严格的文档问答助手。"
                        "只能根据用户提供的资料回答问题。"
                        "如果资料能够支持答案，即使用户问题和资料说法不同，"
                        "也要根据资料组织答案。"
                        "禁止使用资料之外的信息，也不要依靠常识猜测。"
                        "如果资料不能支持答案，必须只回答："
                        "当前文档中没有找到相关信息。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"问题：\n{question}\n\n"
                        f"可用资料：\n{context}\n\n"
                        "请根据以上资料回答。"
                    ),
                },
            ],
        )

    answer = response.choices[0].message.content
    st.chat_message("assistant").write(answer)


    # =========================
    # 11. 展示答案来源和 PDF 页码
    # =========================

    st.subheader("回答依据")

    for rank, (document, distance, metadata) in enumerate(
        zip(
            retrieved_chunks,
            retrieved_distances,
            retrieved_metadatas,
        ),
        start=1,
    ):
        page_number = metadata.get("page_number", "未知")
        chunk_number = metadata.get("chunk_number", "未知")

        with st.expander(
            f"来源 {rank}｜PDF 第 {page_number} 页｜"
            f"文本块 {chunk_number}｜向量距离 {distance:.3f}"
        ):
            st.write(document)
