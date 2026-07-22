from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# =========================
# 1. 加载向量模型
# =========================

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================
# 2. 创建本地向量数据库
# =========================

db_path = Path(__file__).parent / "chroma_db"

client = chromadb.PersistentClient(
    path=str(db_path)
)


# =========================
# 3. 创建集合
# =========================

collection = client.get_or_create_collection(
    name="rag_test"
)


# =========================
# 4. 准备测试资料
# =========================

texts = [
    "小林每天学习四小时，第一小时阅读代码，第二和第三小时开发项目，第四小时测试和复盘。",
    "RAG会先从知识库检索相关资料，再让大模型根据资料回答。",
    "Agent可以根据任务调用天气、搜索、邮件和数据库等工具。"
]

ids = [
    "document_1",
    "document_2",
    "document_3"
]


# =========================
# 5. 生成向量并写入数据库
# =========================

embeddings = model.encode(
    texts,
    normalize_embeddings=True
).tolist()

collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings
)

print(
    f"写入成功，数据库中共有 "
    f"{collection.count()} 条数据"
)


# =========================
# 6. 查询向量数据库
# =========================

question = "每天的时间怎么安排？"

question_embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

results = collection.query(
    query_embeddings=[
        question_embedding
    ],
    n_results=3,
    include=[
        "documents",
        "distances"
    ]
)


# =========================
# 7. 显示查询结果
# =========================

print("\n问题：", question)
print("\n查询结果：")

documents = results["documents"][0]
distances = results["distances"][0]

for rank, (document, distance) in enumerate(
    zip(documents, distances),
    start=1
):
    print(
        f"\n第 {rank} 名｜"
        f"距离：{distance:.3f}"
    )

    print(document)