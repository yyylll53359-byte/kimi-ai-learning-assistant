from sentence_transformers import SentenceTransformer, util


# 加载支持中文的语义向量模型
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


question = "每天的学习时间怎样安排？"


texts = [
    "小林每天可以学习四小时，第一小时阅读代码，第二和第三小时开发项目，第四小时测试和复盘。",
    "RAG会先检索相关文档，再让大模型根据资料回答。",
    "Agent可以调用天气、搜索、邮件等外部工具。"
]


# 把问题和文本转换成向量
question_embedding = model.encode(
    question,
    convert_to_tensor=True
)

text_embeddings = model.encode(
    texts,
    convert_to_tensor=True
)


# 计算语义相似度
scores = util.cos_sim(
    question_embedding,
    text_embeddings
)[0]


print("问题：", question)

for text, score in zip(texts, scores):
    print(f"\n相似度：{score.item():.3f}")
    print(text)