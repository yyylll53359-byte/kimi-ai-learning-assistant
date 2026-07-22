# Kimi AI Learning Assistant

基于 Python、Streamlit、Kimi API、Sentence Transformers 和 Chroma 构建的 AI 学习助手与 PDF 知识库问答项目。

## 项目简介

本项目包含两个主要功能：

1. AI 学习助手

- Kimi 大模型问答
- Streamlit Web 交互
- 多轮对话上下文
- 用户画像管理
- 本地聊天记录持久化
- 清空聊天状态

2. PDF 知识库助手

- 上传并解析文字型 PDF
- 文本切分与重叠处理
- 多语言 Embedding 语义向量
- Chroma 本地向量数据库
- 相似内容检索
- 基于文档上下文生成回答
- 展示回答依据
- 文档没有答案时拒绝编造

## RAG 工作流程

```text
用户上传 PDF
      ↓
提取 PDF 文字
      ↓
切分为多个文本块
      ↓
生成 Embedding 向量
      ↓
保存到 Chroma
      ↓
将用户问题转换为向量
      ↓
检索最相关的三个文本块
      ↓
把问题和相关资料发送给 Kimi
      ↓
生成答案并展示来源
```

## 技术栈

- Python 3.11
- Streamlit
- Kimi API
- OpenAI Python SDK
- Sentence Transformers
- paraphrase-multilingual-MiniLM-L12-v2
- Chroma
- pypdf
- JSON
- python-dotenv

## 项目结构

```text
AI-Learning
├── app.py                 # AI学习助手网页
├── rag_app.py             # PDF知识库助手网页
├── kimi_chat.py           # 基础命令行聊天
├── kimi_chat_v2.py        # 多轮命令行聊天
├── rag_demo.py            # RAG基础流程演示
├── embedding_test.py      # Embedding语义检索测试
├── chroma_test.py         # Chroma向量数据库测试
├── memory.json            # 示例用户画像
├── requirements.txt       # Python依赖
├── .env.example           # 环境变量示例
└── .gitignore             # 隐私文件忽略规则
```

以下内容只保存在本地，不上传 GitHub：

```text
.env
data/
chroma_db/
chat_history.json
```

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

首次运行语义检索时，需要下载多语言 Embedding 模型，可能需要等待几分钟。

## 配置 API Key

为了避免泄露密钥，本项目从用户目录外部读取 `.env`。

Windows 示例位置：

```text
C:\Users\你的用户名\AI-Secrets\.env
```

macOS 或 Linux 示例位置：

```text
~/AI-Secrets/.env
```

`.env` 内容：

```text
KIMI_API_KEY=your_api_key_here
```

请勿将真实 API Key 上传到 GitHub。

## 运行 AI 学习助手

```bash
streamlit run app.py
```

## 运行 PDF 知识库助手

```bash
streamlit run rag_app.py
```

打开网页后：

1. 上传一份可以选择文字的 PDF。
2. 等待系统建立本地向量知识库。
3. 输入与文档有关的问题。
4. 查看 AI 回答及对应来源片段。

扫描图片型 PDF 暂不支持，需要额外接入 OCR。

## 项目亮点

- 从想法到可运行原型独立完成完整链路。
- 使用用户画像和系统提示词实现个性化回答。
- 使用 Session State 管理多轮对话上下文。
- 使用 JSON 实现聊天历史持久化。
- 使用多语言 Embedding 提升中文语义检索效果。
- 使用 Chroma 实现向量的本地持久化和查询。
- 展示回答来源，帮助用户判断答案是否有依据。
- 对无答案问题进行约束，降低模型幻觉。
- 将 API Key、用户文档和向量数据库排除在公开仓库之外。

## 后续计划

- 增加固定问题集和 RAG 效果评估
- 增加页码级来源定位
- 增加多文件知识库
- 增加重新排序 Reranker
- 增加 OCR 扫描文档识别
- 增加 Agent 工具调用能力

## 项目性质

个人 AI 应用学习与实践项目。