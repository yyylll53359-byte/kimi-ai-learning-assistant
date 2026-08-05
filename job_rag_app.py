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
    page_title="求职 JD 整理助手",
    page_icon="💼",
)

st.title("💼 求职 JD 整理助手")
st.write("上传多份岗位 JD，系统会帮你总结共同要求、岗位差异和技能关键词。")


# =========================
# 1. 参数
# =========================

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_COUNT = 5
MAX_DISTANCE = 1.25


# =========================
# 2. 读取 API Key
# =========================

env_path = Path.home() / "AI-Secrets" / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("KIMI_API_KEY")

if not api_key:
    st.error("没有读取到 KIMI_API_KEY，请检查 AI-Secrets 文件夹。")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1",
)


# =========================
# 3. 加载模型和向量数据库
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
        name="job_rag_knowledge_base"
    )


embedding_model = load_embedding_model()
collection = load_chroma_collection()


# =========================
# 4. 工具函数
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
        list(text_chunks),
        normalize_embeddings=True,
    )

    return vectors.tolist()


def split_answer(answer):
    conclusion = answer.strip()
    details = ""

    if "【详细说明】" in answer:
        conclusion, details = answer.split(
            "【详细说明】",
            maxsplit=1,
        )

    conclusion = conclusion.replace(
        "【一句话结论】",
        "",
    ).strip()

    details = details.strip()

    return conclusion, details


def parse_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_id = hashlib.sha256(file_bytes).hexdigest()

    reader = PdfReader(io.BytesIO(file_bytes))

    records = []
    full_text_parts = []
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
            records.append(
                {
                    "text": chunk,
                    "file_id": file_id,
                    "file_name": uploaded_file.name,
                    "page_number": page_number,
                    "page_chunk_number": page_chunk_number,
                }
            )

    return {
        "file_id": file_id,
        "file_name": uploaded_file.name,
        "page_count": len(reader.pages),
        "pages_with_text": pages_with_text,
        "records": records,
        "full_text": "\n\n".join(full_text_parts),
    }


# =========================
# 5. 上传多份岗位 JD
# =========================

uploaded_files = st.file_uploader(
    "上传岗位 JD PDF 文件，可一次选择多份",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("请先上传至少一份岗位 JD PDF。")
    st.stop()

if len(uploaded_files) == 1:
    st.info("当前上传了 1 份 JD。上传 2 份以上后，可以进行岗位对比。")


# =========================
# 6. 解析所有 PDF
# =========================

parsed_documents = []
all_records = []
failed_files = []

for uploaded_file in uploaded_files:
    try:
        document = parse_pdf(uploaded_file)

        if not document["records"]:
            failed_files.append(
                f"{uploaded_file.name}：没有提取到文字，可能是扫描件。"
            )
            continue

        parsed_documents.append(document)
        all_records.extend(document["records"])

    except Exception as error:
        failed_files.append(
            f"{uploaded_file.name}：读取失败，原因是 {error}"
        )


if failed_files:
    for message in failed_files:
        st.warning(message)

if not all_records:
    st.error("没有读取到任何可用文字。请上传文字版 PDF。")
    st.stop()


file_ids = sorted(
    document["file_id"]
    for document in parsed_documents
)

doc_set_id = hashlib.sha256(
    "|".join(file_ids).encode("utf-8")
).hexdigest()

chunks = tuple(
    record["text"]
    for record in all_records
)


# =========================
# 7. 建立多文档知识库
# =========================

if st.session_state.get("doc_set_id") != doc_set_id:
    with st.spinner("正在读取多份 JD 并建立知识库..."):
        chunk_embeddings = create_embeddings(chunks)

        collection.delete(
            where={
                "doc_set_id": doc_set_id
            }
        )

        chunk_ids = []
        metadatas = []

        for index, (record, embedding) in enumerate(
            zip(all_records, chunk_embeddings),
            start=1,
        ):
            chunk_ids.append(
                f"{doc_set_id}_{record['file_id']}_"
                f"page_{record['page_number']}_"
                f"chunk_{record['page_chunk_number']}"
            )

            metadatas.append(
                {
                    "doc_set_id": doc_set_id,
                    "file_id": record["file_id"],
                    "file_name": record["file_name"],
                    "page_number": int(record["page_number"]),
                    "page_chunk_number": int(
                        record["page_chunk_number"]
                    ),
                    "chunk_number": int(index),
                }
            )

        collection.upsert(
            ids=chunk_ids,
            documents=list(chunks),
            embeddings=chunk_embeddings,
            metadatas=metadatas,
        )

        st.session_state.doc_set_id = doc_set_id
        st.session_state.chunk_count = len(chunks)
        st.session_state.file_count = len(parsed_documents)


st.success(
    f"知识库已建立：共 {len(parsed_documents)} 份 JD，"
    f"保存 {len(chunks)} 个文本块。"
)

with st.expander("查看已上传的岗位文件"):
    for document in parsed_documents:
        st.write(
            f"- {document['file_name']}："
            f"{document['page_count']} 页，"
            f"{document['pages_with_text']} 页提取到文字"
        )


# =========================
# 8. 用户提问
# =========================

question = st.chat_input(
    "例如：这些岗位都要求哪些技能？"
)

if question:
    st.chat_message("user").write(question)

    query_embedding = create_embeddings(
        (question,)
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_COUNT, len(chunks)),
        where={
            "doc_set_id": doc_set_id
        },
        include=[
            "documents",
            "distances",
            "metadatas",
        ],
    )

    retrieved_chunks = results["documents"][0] or []
    retrieved_distances = results["distances"][0] or []
    retrieved_metadatas = results["metadatas"][0] or []

    if not retrieved_chunks:
        st.chat_message("assistant").write(
            "当前岗位资料中没有找到相关信息。"
        )
        st.stop()

    best_distance = retrieved_distances[0]

    if best_distance > MAX_DISTANCE:
        st.chat_message("assistant").write(
            "当前岗位资料中没有找到相关信息。"
        )

        st.warning(
            f"最相关内容的向量距离为 {best_distance:.3f}，"
            f"超过拒答阈值 {MAX_DISTANCE:.2f}。"
        )
        st.stop()


    # =========================
    # 9. 组装上下文
    # =========================

    context_parts = []

    for source_number, (
        document,
        metadata,
    ) in enumerate(
        zip(
            retrieved_chunks,
            retrieved_metadatas,
        ),
        start=1,
    ):
        file_name = metadata.get(
            "file_name",
            "未知文件",
        )

        page_number = metadata.get(
            "page_number",
            "未知页码",
        )

        context_parts.append(
            f"资料 {source_number}：\n"
            f"文件名：{file_name}\n"
            f"PDF 第 {page_number} 页\n"
            f"内容：\n{document}"
        )

    context = "\n\n".join(context_parts)


    # =========================
    # 10. 调用 Kimi
    # =========================

    with st.spinner("正在对比岗位要求..."):
        response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一名求职岗位分析助手。"
                        "只能根据用户上传的岗位资料回答。"
                        "不能使用资料之外的信息进行猜测。"
                        "如果资料无法支持答案，"
                        "请回答：当前岗位资料中没有找到相关信息。\n\n"
                        "请严格按照以下格式回答：\n"
                        "【一句话结论】\n"
                        "先用一句话总结最重要的结论。\n\n"
                        "【详细说明】\n"
                        "再用 1 到 5 点解释岗位要求、"
                        "岗位差异或技能关键词。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户问题：\n{question}\n\n"
                        f"岗位资料：\n{context}\n\n"
                        "请根据岗位资料回答。"
                    ),
                },
            ],
        )


    answer = response.choices[0].message.content
    conclusion, details = split_answer(answer)


    # =========================
    # 11. 产品化展示
    # =========================

    with st.chat_message("assistant"):
        st.markdown(
            f"**结论：** {conclusion}"
        )

        if details:
            with st.expander("查看详细说明"):
                st.markdown(details)


    # =========================
    # 12. 展示回答来源
    # =========================

    st.subheader("回答依据")

    for rank, (
        document,
        distance,
        metadata,
    ) in enumerate(
        zip(
            retrieved_chunks,
            retrieved_distances,
            retrieved_metadatas,
        ),
        start=1,
    ):
        file_name = metadata.get(
            "file_name",
            "未知文件",
        )

        page_number = metadata.get(
            "page_number",
            "未知页码",
        )

        with st.expander(
            f"来源 {rank}｜{file_name}｜"
            f"PDF 第 {page_number} 页｜"
            f"向量距离 {distance:.3f}"
        ):
            st.write(document)