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


# =========================
# 1. 页面设置
# =========================

st.set_page_config(
    page_title="PDF知识库助手",
    page_icon="📚"
)

st.title("📚 PDF知识库助手")

st.write(
    "上传一份PDF，系统会将文档保存到本地向量数据库，"
    "再通过语义检索和Kimi回答问题。"
)


# =========================
# 2. RAG参数设置
# =========================

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_COUNT = 3

# 向量距离越小，代表内容越相关
# 超过这个距离就直接拒绝回答
MAX_DISTANCE = 1.25


# =========================
# 3. 读取API Key
# =========================

env_path = (
    Path.home()
    / "AI-Secrets"
    / ".env"
)

load_dotenv(
    dotenv_path=env_path
)

api_key = os.getenv(
    "KIMI_API_KEY"
)

if not api_key:
    st.error(
        "没有读取到KIMI_API_KEY，"
        "请检查AI-Secrets中的.env。"
    )

    st.stop()


# =========================
# 4. 创建Kimi客户端
# =========================

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1"
)


# =========================
# 5. 加载Embedding模型
# =========================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )


embedding_model = load_embedding_model()


# =========================
# 6. 创建Chroma数据库
# =========================

@st.cache_resource
def load_chroma_client():
    database_path = (
        Path(__file__).parent
        / "chroma_db"
    )

    return chromadb.PersistentClient(
        path=str(database_path)
    )


chroma_client = load_chroma_client()

collection = (
    chroma_client
    .get_or_create_collection(
        name="pdf_knowledge_base"
    )
)


# =========================
# 7. 文本切分函数
# =========================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================
# 8. 生成文档向量
# =========================

@st.cache_data(
    show_spinner=False
)
def create_embeddings(
    text_chunks
):
    return embedding_model.encode(
        text_chunks,
        normalize_embeddings=True
    ).tolist()


# =========================
# 9. 上传PDF
# =========================

uploaded_file = st.file_uploader(
    "请上传一份PDF",
    type=["pdf"]
)


if uploaded_file:
    file_bytes = (
        uploaded_file.getvalue()
    )

    # 根据文件内容生成唯一编号
    file_id = hashlib.sha256(
        file_bytes
    ).hexdigest()[:16]


    try:
        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

    except Exception:
        st.error(
            "PDF读取失败，请确认文件没有损坏。"
        )

        st.stop()


    # =========================
    # 10. 提取PDF文字
    # =========================

    full_text = ""

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        page_text = (
            page.extract_text() or ""
        )

        full_text += (
            f"\n--- 第 {page_number} 页 ---\n"
        )

        full_text += page_text


    if not full_text.strip():
        st.error(
            "没有提取到文字。"
            "这份PDF可能是扫描图片，"
            "需要使用OCR。"
        )

        st.stop()


    # =========================
    # 11. 切分文档
    # =========================

    chunks = split_text(
        full_text
    )

    st.success(
        f"读取成功："
        f"共{len(reader.pages)}页，"
        f"提取{len(full_text)}个字符，"
        f"切分为{len(chunks)}个文本块。"
    )


    # =========================
    # 12. 将文档写入Chroma
    # =========================

    with st.spinner(
        "正在建立本地向量知识库..."
    ):
        chunk_embeddings = (
            create_embeddings(chunks)
        )

        chunk_ids = [
            f"{file_id}_chunk_{index}"
            for index in range(
                len(chunks)
            )
        ]

        metadatas = [
            {
                "file_id": file_id,
                "file_name": uploaded_file.name,
                "chunk_number": index + 1
            }
            for index in range(
                len(chunks)
            )
        ]

        collection.upsert(
            ids=chunk_ids,
            documents=chunks,
            embeddings=chunk_embeddings,
            metadatas=metadatas
        )


    st.info(
        f"知识库已建立："
        f"本文件共保存{len(chunks)}个向量。"
    )


    with st.expander(
        "查看提取的部分文字"
    ):
        st.write(
            full_text[:1000]
        )


    # =========================
    # 13. 用户提问
    # =========================

    question = st.chat_input(
        "请输入关于这份PDF的问题..."
    )


    if question:
        st.chat_message(
            "user"
        ).write(
            question
        )


        # =========================
        # 14. 查询Chroma
        # =========================

        with st.spinner(
            "正在查询向量数据库..."
        ):
            question_embedding = (
                embedding_model.encode(
                    question,
                    normalize_embeddings=True
                ).tolist()
            )

            result_count = min(
                TOP_COUNT,
                len(chunks)
            )

            results = collection.query(
                query_embeddings=[
                    question_embedding
                ],
                n_results=result_count,
                where={
                    "file_id": file_id
                },
                include=[
                    "documents",
                    "distances",
                    "metadatas"
                ]
            )


        retrieved_chunks = (
            results["documents"][0]
        )

        distances = (
            results["distances"][0]
        )

        result_metadatas = (
            results["metadatas"][0]
        )


        if not retrieved_chunks:
            st.chat_message(
                "assistant"
            ).write(
                "当前文档中没有找到相关信息。"
            )

            st.stop()


        # =========================
        # 15. 检索距离保护
        # =========================

        best_distance = float(
            distances[0]
        )

        if best_distance > MAX_DISTANCE:
            st.chat_message(
                "assistant"
            ).write(
                "当前文档中没有找到相关信息。"
            )

            st.info(
                f"最佳检索距离为 "
                f"{best_distance:.3f}，"
                f"超过拒答阈值 "
                f"{MAX_DISTANCE}。"
            )

            st.stop()


        # =========================
        # 16. 整理检索资料
        # =========================

        context_parts = []

        for number, document in enumerate(
            retrieved_chunks,
            start=1
        ):
            context_parts.append(
                f"资料{number}：\n{document}"
            )

        context = "\n\n".join(
            context_parts
        )


        # =========================
        # 17. 调用Kimi
        # =========================

        with st.spinner(
            "正在根据文档生成回答..."
        ):
            try:
                response = (
                    client
                    .chat
                    .completions
                    .create(
                        model="moonshot-v1-8k",

                        messages=[
                            {
                                "role": "system",

                                "content": (
                                    "你是一名文档问答助手。"
                                    "只能根据提供的资料回答问题。"
                                    "禁止使用资料之外的信息。"
                                    "禁止根据常识进行猜测。"
                                    "只有资料明确写出答案时，"
                                    "才可以回答。"
                                    "如果资料没有明确答案，"
                                    "必须回答："
                                    "当前文档中没有找到相关信息。"
                                )
                            },

                            {
                                "role": "user",

                                "content": f"""
检索到的资料：

{context}

用户问题：

{question}

请先判断资料中是否明确包含答案。

如果包含，请给出简洁、准确的回答。

如果不包含，只回答：
当前文档中没有找到相关信息。
"""
                            }
                        ]
                    )
                )

            except Exception as error:
                st.error(
                    f"调用Kimi失败：{error}"
                )

                st.stop()


        answer = (
            response
            .choices[0]
            .message
            .content
        )


        # =========================
        # 18. 显示答案
        # =========================

        st.chat_message(
            "assistant"
        ).write(
            answer
        )


        # =========================
        # 19. 显示回答依据
        # =========================

        st.subheader(
            "回答依据"
        )

        for rank, (
            document,
            distance,
            metadata
        ) in enumerate(
            zip(
                retrieved_chunks,
                distances,
                result_metadatas
            ),
            start=1
        ):
            chunk_number = metadata[
                "chunk_number"
            ]

            with st.expander(
                f"来源{rank}｜"
                f"文本块{chunk_number}｜"
                f"向量距离{distance:.3f}"
            ):
                st.write(
                    document
                )