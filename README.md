# Kimi AI Learning Assistant

基于 Python、Streamlit、Kimi API、Sentence Transformers 和 Chroma 构建的 AI 学习助手与 PDF 知识库问答项目。

项目仓库：

https://github.com/yyylll53359-byte/kimi-ai-learning-assistant

## 项目简介

本项目包含两个主要功能。

### 1. AI学习助手

- Kimi大模型问答
- Streamlit Web交互
- 多轮对话上下文
- 用户画像管理
- 本地聊天记录持久化
- 清空聊天状态

### 2. PDF知识库助手

- 上传并解析文字型PDF
- 按PDF页面提取和切分文字
- 文本块重叠处理
- 多语言Embedding语义向量
- Chroma本地向量数据库
- Top 3语义检索
- 基于文档上下文生成回答
- 展示PDF页码、文本块和向量距离
- 使用距离阈值拦截低相关问题
- 文档没有答案时拒绝编造
- 使用固定测试题自动评测RAG效果

## RAG工作流程

```text
用户上传PDF
      ↓
pypdf逐页提取文字
      ↓
每一页单独切分文本
      ↓
相邻文本块重叠50字
      ↓
生成Embedding语义向量
      ↓
将页码等metadata保存到Chroma
      ↓
将用户问题转换为向量
      ↓
检索最相关的3个文本块
      ↓
检查最佳向量距离
      ↓
距离超过1.25：直接拒答
距离不超过1.25：交给Kimi
      ↓
生成答案并展示PDF页码来源
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
├── eval_rag.py            # RAG自动评测程序
├── eval_questions.json    # 固定评测问题集
├── eval_results.json      # 自动生成的评测结果
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

以下内容只保存在本地，不上传GitHub：

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

首次运行语义检索时，需要下载多语言Embedding模型，可能需要等待几分钟。

## 配置API Key

为了避免泄露密钥，本项目从项目外部读取`.env`。

Windows示例位置：

```text
C:\Users\你的用户名\AI-Secrets\.env
```

macOS或Linux示例位置：

```text
~/AI-Secrets/.env
```

`.env`内容：

```text
KIMI_API_KEY=your_api_key_here
```

请勿将真实API Key上传到GitHub。

## 运行AI学习助手

```bash
python -m streamlit run app.py
```

## 运行PDF知识库助手

```bash
python -m streamlit run rag_app.py
```

打开网页后：

1. 上传一份可以选择文字的PDF。
2. 等待系统建立本地向量知识库。
3. 输入与文档有关的问题。
4. 查看AI回答及对应的PDF页码、文本块和向量距离。
5. 当检索距离过大时，系统会拒绝回答。

扫描图片型PDF暂不支持，需要额外接入OCR。

## 页码级来源溯源

为了让用户能够回到PDF原文核对答案，系统将PDF改为逐页读取和页内切分。

每个文本块写入Chroma时，会同时保存：

```text
file_id
file_name
page_number
page_chunk_number
chunk_number
```

检索完成后，页面会显示类似：

```text
来源1｜PDF第2页｜文本块6｜向量距离1.121
```

重新上传同一份PDF时，程序会先根据`file_id`删除旧向量记录，再写入带页码的新记录，避免缺少页码的历史数据污染查询结果。

## 运行RAG自动评测

先使用`rag_app.py`上传测试PDF并建立本地知识库，然后运行：

```bash
python eval_rag.py
```

评测程序会自动完成：

1. 读取10道固定测试题。
2. 将每道题转换为语义向量。
3. 从Chroma查询最相关的3个文本块。
4. 检查检索是否命中正确资料。
5. 对距离过大的问题直接拒答。
6. 对距离合格的问题调用Kimi。
7. 检查答案关键词和无答案拒答情况。
8. 记录每道题的响应时间。
9. 将详细结果保存到`eval_results.json`。

## RAG评测结果

当前测试集包含：

- 6道文档中有答案的问题
- 4道文档中没有答案的问题

### 优化前

```text
检索命中率：100.0%
答案通过率：90.0%
无答案拒答率：75.0%
平均响应时间：1.28秒
实际调用Kimi：10次
```

### 优化措施

根据评测结果增加两层防幻觉保护：

```text
第一层：最佳向量距离超过1.25时，Python直接拒答
第二层：距离合格后，要求Kimi只能根据检索资料回答
```

同时调整提示词，避免系统对有资料支持的问题过度拒答。

### 优化后

```text
检索命中率：100.0%
答案通过率：100.0%
无答案拒答率：100.0%
平均响应时间：0.76秒
距离直接拒答：4题
实际调用Kimi：6次
```

距离保护使4道无答案题不再调用Kimi，在当前测试集中提高了拒答准确率，并减少了API调用次数。

以上结果仅代表当前PDF和10道固定测试题，不代表系统在所有文档上都能永久达到100%。

`1.25`是根据当前测试数据选择的初始阈值。更换文档、Embedding模型或距离算法后，需要重新评测并调整。

## Day 6页码溯源验收

使用3页测试PDF重新建立知识库后，共生成9个带页码的文本向量。

### 第2页测试

```text
问题：RAG的五个核心步骤是什么？
结果：回答正确
来源：前两条来源均为PDF第2页
```

### 第3页测试

```text
问题：一个能运行的Demo至少需要关注哪四个指标？
结果：回答正确
来源：前两条来源均为PDF第3页
最佳向量距离：0.456
```

### 无答案拒答测试

```text
问题：小林最喜欢吃什么？
结果：当前文档中没有找到相关信息
最佳向量距离：1.794
拒答阈值：1.25
```

以上测试证明：页码能够随文本块写入向量数据库，并在查询后正确返回；同时，第5天增加的距离拒答功能仍然有效。

## 项目亮点

- 独立完成从PDF解析到网页问答的完整RAG链路。
- 使用多语言Embedding实现中文语义检索。
- 使用Chroma实现向量的本地持久化和查询。
- 实现逐页解析、页内切分和页码级来源定位。
- 展示PDF页码、文本块和向量距离，方便核查答案依据。
- 建立包含有答案题和无答案题的固定评测集。
- 自动统计检索命中率、答案通过率、拒答率和响应时间。
- 根据评测结果定位幻觉问题并调整产品策略。
- 使用距离阈值在调用大模型前拦截低相关问题。
- 在提高准确率的同时减少Kimi API调用次数。
- 将API Key、用户文档和本地向量数据库排除在公开仓库之外。

## 当前局限

- 当前评测集只有10道题，规模较小。
- 关键词评分只能进行基础检查，不能完全代表语义正确。
- 距离阈值需要根据不同文档重新评测。
- 目前主要支持单份PDF。
- 暂不支持扫描图片型PDF。
- 页码溯源暂时不能定位到页面中的具体段落或坐标。
- 尚未加入Reranker重新排序。
- 尚未部署为公开访问的在线产品。

## 后续计划

- 扩充RAG评测问题集
- 增加多文件知识库
- 增加Reranker重新排序
- 增加OCR扫描文档识别
- 增加用户反馈和日志统计
- 增加Agent工具调用能力
- 增加人工评分或大模型评分
- 部署为可公开访问的演示项目

## 项目性质

个人AI应用学习与实践项目。
      