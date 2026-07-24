import json
import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# =========================
# 1. 路径和参数
# =========================

PROJECT_PATH = Path(__file__).parent

QUESTIONS_PATH = (
    PROJECT_PATH
    / "eval_questions.json"
)

RESULTS_PATH = (
    PROJECT_PATH
    / "eval_results.json"
)

DATABASE_PATH = (
    PROJECT_PATH
    / "chroma_db"
)

TOP_COUNT = 3
MAX_DISTANCE = 1.25

REFUSAL_TEXT = (
    "当前文档中没有找到相关信息"
)


# =========================
# 2. 读取API Key
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
    print(
        "错误：没有读取到KIMI_API_KEY。"
    )

    raise SystemExit


client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1"
)


# =========================
# 3. 读取测试题
# =========================

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as file:
    test_questions = json.load(file)


print(
    f"成功读取 {len(test_questions)} 道测试题。"
)


# =========================
# 4. 加载模型和数据库
# =========================

print("正在加载Embedding模型...")

embedding_model = SentenceTransformer(
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


chroma_client = chromadb.PersistentClient(
    path=str(DATABASE_PATH)
)

collection = chroma_client.get_collection(
    name="pdf_knowledge_base"
)


chunk_count = collection.count()

print(
    f"知识库连接成功，共有 "
    f"{chunk_count} 个文本块。"
)


if chunk_count == 0:
    print(
        "知识库是空的，请先运行rag_app.py上传PDF。"
    )

    raise SystemExit


# =========================
# 5. 检索资料函数
# =========================

def retrieve_documents(question):
    question_embedding = (
        embedding_model.encode(
            question,
            normalize_embeddings=True
        ).tolist()
    )

    result_count = min(
        TOP_COUNT,
        chunk_count
    )

    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=result_count,
        include=[
            "documents",
            "distances",
            "metadatas"
        ]
    )

    documents = (
        results["documents"][0]
    )

    distances = (
        results["distances"][0]
    )

    return documents, distances


# =========================
# 6. 调用Kimi函数
# =========================

def ask_kimi(
    question,
    documents
):
    context_parts = []

    for number, document in enumerate(
        documents,
        start=1
    ):
        context_parts.append(
            f"资料{number}：\n"
            f"{document}"
        )

    context = "\n\n".join(
        context_parts
    )

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
                        "只能根据提供的资料回答问题，"
                        "不要使用资料之外的常识。"
                        "只要资料中存在能够支持答案的信息，"
                        "就应该正常回答。"
                        "即使用户问题的说法与原文不完全相同，"
                        "也要理解意思后回答。"
                        "只有当资料完全没有相关信息时，"
                        "才回答："
                        "当前文档中没有找到相关信息。"
                    )
                },

                {
                    "role": "user",

                    "content": f"""
下面是系统检索到的资料：

{context}

用户问题：

{question}

请完成以下判断：

1. 如果资料能够支持答案，请直接给出简洁答案。
2. 不要求问题文字与原文完全相同，要理解它们的意思。
3. 不得补充资料之外的信息。
4. 如果资料完全无法支持答案，只回答：
当前文档中没有找到相关信息。
"""
                }
            ]
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# =========================
# 7. 初始化评测统计
# =========================

evaluation_results = []

answerable_count = 0
unanswerable_count = 0

retrieval_hit_count = 0
answer_pass_count = 0
refusal_pass_count = 0

distance_rejection_count = 0
api_call_count = 0

total_time = 0


# =========================
# 8. 逐题评测
# =========================

for item in test_questions:
    question_id = item["id"]
    question = item["question"]
    should_answer = item["should_answer"]
    keywords = item["keywords"]

    print("\n" + "=" * 60)

    print(
        f"第 {question_id} 题："
        f"{question}"
    )


    # =========================
    # 8.1 检索资料
    # =========================

    documents, distances = (
        retrieve_documents(
            question
        )
    )


    if not documents:
        print(
            "知识库没有返回任何资料。"
        )

        continue


    best_distance = float(
        distances[0]
    )

    retrieved_text = "\n\n".join(
        documents
    )


    print(
        f"最佳向量距离："
        f"{best_distance:.3f}"
    )


    # =========================
    # 8.2 检查检索命中
    # =========================

    found_in_context = [
        keyword
        for keyword in keywords
        if keyword in retrieved_text
    ]

    retrieval_pass = False


    if should_answer:
        answerable_count += 1

        if found_in_context:
            retrieval_pass = True
            retrieval_hit_count += 1

    else:
        unanswerable_count += 1


    # =========================
    # 8.3 距离保护
    # =========================

    rejected_by_distance = (
        best_distance
        > MAX_DISTANCE
    )


    if rejected_by_distance:
        distance_rejection_count += 1

        answer = (
            "当前文档中没有找到相关信息。"
        )

        elapsed_time = 0.0
        error_message = None

        print(
            f"距离超过 {MAX_DISTANCE}，"
            f"Python直接拒答。"
        )


    else:
        # =========================
        # 8.4 调用Kimi
        # =========================

        start_time = time.perf_counter()

        try:
            api_call_count += 1

            answer = ask_kimi(
                question,
                documents
            )

            error_message = None

        except Exception as error:
            answer = ""
            error_message = str(error)


        elapsed_time = (
            time.perf_counter()
            - start_time
        )


    total_time += elapsed_time


    # =========================
    # 8.5 自动批改
    # =========================

    answer_pass = False
    refusal_pass = False
    found_in_answer = []


    if error_message:
        print(
            f"API调用失败："
            f"{error_message}"
        )


    elif should_answer:
        found_in_answer = [
            keyword
            for keyword in keywords
            if keyword in answer
        ]

        if found_in_answer:
            answer_pass = True
            answer_pass_count += 1

        print(
            f"AI回答：{answer}"
        )

        print(
            "答案检查："
            + (
                "通过 ✅"
                if answer_pass
                else "未通过 ❌"
            )
        )


    else:
        if REFUSAL_TEXT in answer:
            refusal_pass = True
            answer_pass = True

            refusal_pass_count += 1
            answer_pass_count += 1

        print(
            f"AI回答：{answer}"
        )

        print(
            "拒答检查："
            + (
                "通过 ✅"
                if refusal_pass
                else "未通过 ❌"
            )
        )


    print(
        f"回答时间："
        f"{elapsed_time:.2f}秒"
    )


    # =========================
    # 8.6 保存本题结果
    # =========================

    evaluation_results.append(
        {
            "id": question_id,

            "question": question,

            "should_answer": (
                should_answer
            ),

            "keywords": keywords,

            "retrieval_pass": (
                retrieval_pass
            ),

            "found_in_context": (
                found_in_context
            ),

            "top_distances": [
                round(
                    float(distance),
                    4
                )
                for distance in distances
            ],

            "best_distance": round(
                best_distance,
                4
            ),

            "max_distance": (
                MAX_DISTANCE
            ),

            "rejected_by_distance": (
                rejected_by_distance
            ),

            "answer": answer,

            "found_in_answer": (
                found_in_answer
            ),

            "answer_pass": (
                answer_pass
            ),

            "refusal_pass": (
                refusal_pass
            ),

            "response_time_seconds": round(
                elapsed_time,
                2
            ),

            "error": error_message
        }
    )


# =========================
# 9. 计算最终成绩
# =========================

total_questions = len(
    test_questions
)


if answerable_count > 0:
    retrieval_rate = (
        retrieval_hit_count
        / answerable_count
        * 100
    )

else:
    retrieval_rate = 0


if total_questions > 0:
    answer_rate = (
        answer_pass_count
        / total_questions
        * 100
    )

    average_time = (
        total_time
        / total_questions
    )

else:
    answer_rate = 0
    average_time = 0


if unanswerable_count > 0:
    refusal_rate = (
        refusal_pass_count
        / unanswerable_count
        * 100
    )

else:
    refusal_rate = 0


# =========================
# 10. 保存评测报告
# =========================

summary = {
    "total_questions": (
        total_questions
    ),

    "answerable_questions": (
        answerable_count
    ),

    "unanswerable_questions": (
        unanswerable_count
    ),

    "retrieval_hit_rate": round(
        retrieval_rate,
        1
    ),

    "answer_pass_rate": round(
        answer_rate,
        1
    ),

    "refusal_pass_rate": round(
        refusal_rate,
        1
    ),

    "average_response_time_seconds": round(
        average_time,
        2
    ),

    "distance_threshold": (
        MAX_DISTANCE
    ),

    "distance_rejection_count": (
        distance_rejection_count
    ),

    "api_call_count": (
        api_call_count
    )
}


report = {
    "summary": summary,
    "details": evaluation_results
}


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        report,
        file,
        ensure_ascii=False,
        indent=4
    )


# =========================
# 11. 显示最终结果
# =========================

print("\n" + "=" * 60)

print("RAG评测完成")

print(
    f"检索命中率："
    f"{retrieval_rate:.1f}%"
)

print(
    f"答案通过率："
    f"{answer_rate:.1f}%"
)

print(
    f"无答案拒答率："
    f"{refusal_rate:.1f}%"
)

print(
    f"平均响应时间："
    f"{average_time:.2f}秒"
)

print(
    f"距离直接拒答："
    f"{distance_rejection_count}题"
)

print(
    f"实际调用Kimi："
    f"{api_call_count}次"
)

print(
    "详细结果已保存到："
    "eval_results.json"
)