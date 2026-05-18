# 💊 药品知识库 AI 助手

> 基于 RAG（检索增强生成）技术，以 10 万份药品说明书为知识库，提供专业药品信息查询服务。

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-1.2-orange)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.14-purple)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [核心模块说明](#核心模块说明)
- [API 文档](#api-文档)
- [配置说明](#配置说明)
- [免责声明](#免责声明)

---

## 功能特性

- **按药品名称查询** — 输入商品名或通用名，精准召回药品详细说明书
- **按症状推荐用药** — 描述身体症状，智能匹配适应症，推荐候选药品
- **复杂问题综合查询** — 处理"高血压患者能用什么感冒药"等多意图混合问题
- **流式实时输出** — SSE 技术逐 token 推送，打字机效果实时显示回答
- **多轮对话记忆** — 会话内上下文连贯，支持追问和澄清
- **完全离线部署** — Embedding 模型本地运行，无需连接 HuggingFace，适合内网环境

---

## 系统架构

```
用户浏览器
    │  HTTP / SSE (Server-Sent Events)
    ▼
FastAPI Web 服务
    │
    ▼
MedicineAgent（LangGraph ReAct Agent + Qwen LLM）
    │  按意图路由至对应工具
    ├── search_medicine_by_name        ──→  product      集合（名称检索）
    ├── search_medicine_by_symptom     ──→  indications  集合（症状检索）
    └── search_medicine_by_comprehensive ─→ comprehensive集合（综合检索）
                   │
                   ▼
         MedicineIndexManager（llama-index）
           ├── BGE-M3 Embedding（本地 GPU）
           ├── ChromaDB 持久化向量数据库
           └── 后处理器链
                 ├── SimilarityPostprocessor（相似度阈值过滤）
                 ├── SentenceTransformerRerank（可选精排）
                 └── LongContextReorder（缓解"迷失中间"）
```

### 三集合检索策略

| 集合 | 向量化内容 | 检索增强技术 |
|------|-----------|------------|
| `medicine_product` | 药品名称、商品名、拼音、分类 | 元数据精确/包含匹配快速路径（绕过向量相似度） |
| `medicine_indications` | 适应症全文、疾病关键词、药理摘要 | QueryFusionRetriever + RRF（4路多查询融合） |
| `medicine_comprehensive` | 名称+适应症+用法+禁忌+不良反应+注意事项 | HyDEQueryTransform（假设文档向量化） |

---

## 技术栈

### 后端

| 组件 | 技术 | 版本 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 0.135 / 0.41 |
| Agent 框架 | LangChain + LangGraph | 1.2 |
| LLM 接入 | LangChain-OpenAI（兼容 Qwen API） | 1.1 |
| 向量索引 | llama-index-core | 0.14 |
| Embedding 模型 | BAAI/bge-m3（HuggingFace） | — |
| 向量数据库 | ChromaDB（本地持久化，cosine 空间） | 1.5 |
| 深度学习框架 | PyTorch（CUDA 12.9） | 2.8 |
| 大语言模型 | Qwen3.5-plus（阿里云 DashScope） | — |
| 数据清洗 | BeautifulSoup4 | 4.14 |

### 前端

| 组件 | 技术 |
|------|------|
| 界面框架 | 原生 HTML5 + CSS3 + Vanilla JS |
| Markdown 渲染 | marked.js v9 |
| 代码高亮 | highlight.js v11 |
| 会话持久化 | localStorage |
| 流式消费 | Fetch API + ReadableStream + AbortController |

---

## 快速开始

### 前置要求

- Python 3.12.3
- conda
- NVIDIA GPU（推荐显存 ≥ 8GB，支持 CUDA 12.x）
- 已下载 BGE-M3 模型至本地（默认路径 `D:/Embeddings_Model`）
- 已构建 ChromaDB 向量索引（默认路径 `D:/Embeddings_Data/medicine_llamaindex/storage`）

### 安装依赖

```bash
conda create -n lang python=3.12.3
conda activate lang
pip install -r requirements.txt
```

### 配置（.env 注入）

敏感配置（API Key、服务端口等）通过项目根目录的 `.env` 文件注入，`config/config.py` 启动时自动加载（`python-dotenv`），**禁止硬编码到代码**。

复制以下内容到项目根目录的 `.env` 并填入自己的密钥：

```dotenv
# ─── Qwen（通义千问） ─────────────────────────────────
QWEN_API_KEY=your-dashscope-api-key
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL_NAME=qwen3.5-plus
QWEN_TEMPERATURE=0.2
QWEN_MAX_TOKENS=10000
QWEN_TIMEOUT=60

# ─── FastAPI 服务 ─────────────────────────────────────
FASTAPI_HOST=127.0.0.1
FASTAPI_PORT=8000

# ─── 离线资源路径（可选，缺省走 config.py 中的默认值） ─
# MEDICINE_BASE_DIR=D:/Embeddings_Data/medicine_llamaindex
# EMBEDDING_MODELS_ROOT=D:/Embeddings_Model
# EMBEDDING_DEVICE=cuda
# EMBEDDING_TARGET_DEVICES=cuda:0,cuda:1
# RERANKER_ENABLED=false
```

`.env` 已在 `.gitignore` 中，不会被提交。其他相对稳定的离线资源路径（向量库 / 模型缓存）默认值在 `config/config.py` 中，按需通过环境变量覆盖。

### 构建向量索引（首次使用）

```bash
conda activate lang
cd /path/to/project
python build_index.py
# 或使用脚本入口（PowerShell）
pwsh ./scripts/build.ps1
```

### 启动服务

```bash
conda activate lang
cd /path/to/project
uvicorn app_fastapi:app --host 127.0.0.1 --port 8000 --reload
# 或使用脚本入口（PowerShell，host/port 走 .env 默认值）
pwsh ./scripts/run.ps1
```

访问 [http://127.0.0.1:8000](http://127.0.0.1:8000) 即可使用。

---

## 项目结构

```
medicine-ai-assistant/
├── app_fastapi.py          # FastAPI 服务入口，SSE 流式对话端点
├── build_index.py          # 向量索引构建入口（process_data → build_indexes）
├── .env                    # 敏感配置（API Key / 端口），git 忽略
├── config/
│   └── config.py           # 全局配置（自动 load_dotenv，提供默认值）
├── modules/
│   ├── agents.py           # LangChain ReAct Agent、三个 Tool、流式输出
│   ├── index_manager.py    # llama-index 向量索引构建、加载、检索管理
│   ├── chroma_store.py     # ChromaDB 适配器（DEFAULT/MMR 查询、MetadataFilters）
│   └── data_loader.py      # 药品 JSON 数据加载、清洗、三类 Document 构建
├── scripts/
│   ├── run.ps1             # 启动 FastAPI（PowerShell）
│   └── build.ps1           # 构建向量索引（PowerShell）
├── static/
│   ├── index.html          # 前端 SPA 主页
│   ├── css/style.css       # 界面样式
│   └── js/app.js           # 前端逻辑（SSE 消费、Markdown 渲染、会话管理）
└── README.md
```

---

## 核心模块说明

### Agent 决策逻辑

Agent 根据用户输入的意图，自动选择对应检索工具：

| 用户输入示例 | 触发工具 |
|-------------|---------|
| "阿莫西林的用法用量" | `search_medicine_by_name` |
| "感冒头痛发烧推荐什么药" | `search_medicine_by_symptom` |
| "高血压患者感冒了可以吃什么药" | `search_medicine_by_comprehensive` |
| "哪些感冒药含有嗜睡成分" | `search_medicine_by_comprehensive` |

### 数据字段覆盖

每条药品记录包含以下字段（存储于 ChromaDB metadata）：

`标题` · `通用名称` · `商品名称` · `批准文号` · `药品分类` · `药品性质` · `生产企业` · `相关疾病` · `适应症` · `主要成份` · `规格` · `用法用量` · `不良反应` · `禁忌` · `注意事项` · `孕妇及哺乳期妇女用药` · `儿童用药` · `老人用药` · `药物相互作用` · `药理毒理` · `贮藏` · `有效期`

### 增量索引更新

向量索引支持增量写入，新增药品数据时无需重建全量索引：

```bash
# 将新数据追加到 all_data.json 后，重新运行
python build_index.py
# 已处理的文档 ID 通过 processed_ids.json 追踪，自动跳过
```

---

## API 文档

### `POST /api/chat`

流式对话接口（Server-Sent Events）。

**请求体**

```json
{
  "message": "阿莫西林的用法用量是什么？",
  "thread_id": ""
}
```

> `thread_id` 为空时自动创建新会话；传入已有 ID 则延续该会话的对话历史。

**响应流（text/event-stream）**

```
data: {"thread_id": "uuid-xxx", "type": "meta"}

data: {"token": "阿莫西林", "type": "token"}
data: {"token": "（", "type": "token"}
...
data: {"token": "。", "type": "token"}

data: [DONE]
```

| 事件类型 | 说明 |
|---------|------|
| `meta` | 首帧，携带本次会话的 `thread_id` |
| `token` | LLM 输出的文本片段，前端逐步拼接渲染 |
| `error` | 发生错误时推送，携带错误信息 |
| `[DONE]` | 流结束标志 |

### `POST /api/session`

创建新会话，返回唯一 `thread_id`。

```json
// 响应
{ "thread_id": "550e8400-e29b-41d4-a716-446655440000" }
```

### `GET /health`

健康检查。

```json
{
  "status": "ok",
  "agent_loaded": true,
  "index_loaded": true
}
```

---

## 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Embedding 模型名称 |
| `EMBEDDING_DEVICE` | `cuda` | 推理设备（cuda / cpu） |
| `EMBEDDING_BATCH_SIZE` | `128` | 向量化批大小 |
| `RERANKER_ENABLED` | `False` | 是否启用二阶段精排 |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | 精排模型（需单独下载） |
| `QWEN_MODEL_NAME` | `qwen3.5-plus` | 使用的千问模型 |
| `QWEN_TEMPERATURE` | `0.2` | LLM 温度（越低越稳定） |
| `QWEN_MAX_TOKENS` | `10000` | 最大输出 token 数 |
| `FASTAPI_HOST` | `127.0.0.1` | 服务监听地址 |
| `FASTAPI_PORT` | `8000` | 服务监听端口 |

所有配置项均可通过 `.env` 覆盖；空缺时使用 `config/config.py` 中的默认值。

**启用精排（Reranker）：** 在 `.env` 中设置

```dotenv
RERANKER_ENABLED=true
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
```

---

## 免责声明

> 本系统仅供参考，不构成任何医疗建议。药品信息来源于说明书文本，可能存在版本滞后。
> **请在医生或药师的指导下用药。**

---
