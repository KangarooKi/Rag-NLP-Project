# NLP课程答疑系统（基于RAG技术）

一个基于**检索增强生成（RAG）**技术的自然语言处理课程答疑智能体，支持**基线RAG**和**知识图谱增强的GraphRAG**两种问答模式，帮助学生快速获取课程相关知识点。

---

## 核心功能

- **基线RAG问答**：通过向量数据库检索相关文本片段，结合大语言模型生成精准答案  
- **GraphRAG问答**：利用知识图谱挖掘概念间关系，提供更具关联性的深度回答  
- **自动构建知识图谱**：从课程材料中提取实体与关系，构建NLP领域知识网络  
- **Web交互界面**：基于 Gradio 提供直观可视化交互，支持对话历史记录追溯  

---

## 技术架构

### 检索层
- 文本分块：采用 `RecursiveCharacterTextSplitter` 实现自适应分块（大小200，重叠50）  
- 向量存储：使用 **Chroma** 作为向量数据库存储文本嵌入  
- 嵌入模型：基于 `intfloat/e5-base-v2` 生成文本向量  

### 生成层
- 大语言模型：集成智谱 AI 的 `glm-4-flashx` 模型  
- 提示工程：针对课程答疑场景优化的提示词模板  

### 知识图谱层
- 实体关系提取：从文本中识别 NLP 领域实体（概念 / 模型 / 工具等）  
- 存储引擎：使用 **Neo4j** 图数据库存储实体与关系  
- 关系类型：支持 `DependsOn` / `IsA` / `UsedFor` / `ProposedBy` / `Contains` 等关系  

### 交互层
- 基于 **Gradio** 构建 Web 界面  
- 支持多标签页切换两种 RAG 模式  

---

## 如何运行

### 1️.环境准备

```bash
# 克隆仓库
git clone https://github.com/KangarooKi/Rag-NLP-Project.git
cd Rag-NLP-Project
```

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2️.配置环境

修改 `config.py` 文件：

- 设置数据存储路径 `DATA_PATH`（建议使用英文路径）  
- 填入智谱 AI API 密钥 `LLM_API_KEY`（从智谱 AI 开放平台获取）  
- 配置 Neo4j 数据库连接信息（需提前安装 Neo4j）

### 3️.准备课程材料

将 **PDF** 或 **TXT** 格式的课程讲义、课件放入 `DATA_PATH` 目录下。

### 4️.运行系统

```bash
python app.py
```

运行后访问终端输出的本地地址（默认 [http://localhost:7860/](http://localhost:7860/)）打开 Web NLP课程答疑系统界面；此外，你可以通过访问适配的neo4j数据库查看生成的可视化知识图谱，具体安装与查看步骤详见 https://blog.csdn.net/qq_60492174/article/details/144369861。  

> 首次使用 GraphRAG 模式需点击 “构建知识图谱” 按钮（耗时较长，取决于材料数量）。

---

## 目录结构

```plaintext
Rag-NLP-Project/
├── app.py               # Gradio 交互界面
├── config.py            # 配置文件（路径、密钥、参数）
├── nlp_rag_core.py      # RAG 核心逻辑实现
├── requirements.txt     # 依赖清单
├── data/                # 数据目录（需自行创建）
│   ├── 课程材料.pdf/txt  # 输入的课程文档
│   ├── embedding_model/  # 嵌入模型（自动下载）
│   └── nlp_rag_db/      # 向量数据库（自动生成）
└── .gitignore           # Git 忽略文件
```

---

## ⚠️ 注意事项

- 嵌入模型和向量数据库体积较大，已通过 `.gitignore` 排除，需本地生成。  
- 知识图谱构建首次运行耗时较长，后续无需重复构建。  
- 敏感信息（API 密钥、数据库密码）建议使用环境变量管理（可结合 `python-dotenv`）。  
- 若需更换大语言模型，可修改 `nlp_rag_core.py` 中的 `_load_llm` 方法。  

---

## 扩展方向

- 支持更多文档类型（Markdown、PPT 等）  
- 增加模型选择功能（支持多 LLM 切换）  
- 优化知识图谱实体链接和关系抽取精度  
- 加入问答评价反馈机制  
- 实现文档自动更新与增量索引  
- 增加知识图谱生成速度
- 增加语料库广度
---

## 致谢

本项目参考并借鉴了微软 GraphRAG 框架的设计思想，  
感谢智谱 AI 提供的大语言模型支持，以及所有参与课程项目的贡献者。
