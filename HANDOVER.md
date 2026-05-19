# 项目交接文档 — 药品知识库 AI 助手

> 交接日期：2026-05-18
> 交接范围：本日（2026-05-18）完成的项目熟悉与配置规范化优化
> 仓库根：`D:\OneDrive\lang\medicine-ai-assistant`
> 当前 HEAD：`a0b9897`
> 主分支：`master`

---

## 一、项目定位

基于 **RAG（检索增强生成）** 的药品知识库 AI 助手。以约 10 万份药品说明书为知识库，通过 FastAPI + LangGraph ReAct Agent + Qwen LLM + llama-index/ChromaDB（BGE-M3 向量化）提供：

- 按药品名称查询说明书
- 按症状/疾病推荐用药
- 复杂多意图查询（如"高血压患者能吃什么感冒药"）
- SSE 流式 token 推送、多轮会话记忆、完全离线 Embedding

---

## 二、架构总览

```
浏览器 (SSE)
  ↓
app_fastapi.py  /api/chat
  ↓
MedicineAgent  (LangGraph create_agent + InMemorySaver)
  ├ tool: search_medicine_by_name        → product 集合      (元数据精确/$contains 快速路径)
  ├ tool: search_medicine_by_symptom     → indications 集合  (QueryFusionRetriever + RRF, 4 路融合)
  └ tool: search_medicine_by_comprehensive → comprehensive 集合 (HyDEQueryTransform)
                                              ↓
            MedicineIndexManager (llama-index + ChromaVectorStore)
            后处理链：SimilarityPostprocessor → 可选 Rerank → LongContextReorder
```

### 三集合设计要点

| 集合 | 向量化内容 | 检索增强 |
|------|-----------|---------|
| `medicine_product` | 名称 + 拼音 + 分类（极短文本） | 元数据 $eq → $contains 两阶段，命中则跳过向量检索 |
| `medicine_indications` | 适应症全文 + 拆词后的相关疾病 + 药理摘要前 250 字 | QueryFusionRetriever 生成 3 变体 + 原 query，RRF 融合；跳过余弦阈值过滤 |
| `medicine_comprehensive` | 名称 + 适应症 + 用法 + 禁忌 + 不良反应 + 注意事项 | HyDEQueryTransform 生成假设说明书段落取平均向量 |

---

## 三、关键文件索引

| 文件 | 行数 | 职责 |
|------|------|------|
| `app_fastapi.py` | 190 | FastAPI 入口、SSE 流式端点、生命周期预加载 |
| `build_index.py` | 79 | 向量索引构建入口（**今日新增**） |
| `config/config.py` | 102 | 全局配置，启动时自动 `load_dotenv()`（**今日重写**） |
| `modules/agents.py` | 309 | 3 个 `@tool` + `MedicineAgent`（同步 chat + `astream_events` 流式） |
| `modules/index_manager.py` | 540 | BGE-M3 加载、三集合 build/load/search、HyDE/QueryFusion 装配、增量索引 |
| `modules/chroma_store.py` | 199 | chromadb 1.3 + llama-index 0.14 适配，MetadataFilters、MMR |
| `modules/data_loader.py` | 292 | JSON 清洗、为每条药品生成 3 类 Document |
| `scripts/run.ps1` | 38 | 启动 FastAPI（**今日新增**） |
| `scripts/build.ps1` | 22 | 构建索引（**今日新增**） |
| `static/index.html, css/, js/` | — | 原生 HTML/JS SPA |
| `data/all_medicine.json` | — | 462 MB，git LFS 管理 |

---

## 四、本日工作内容（按时间顺序）

### 4.1 项目熟悉（只读探索）

通读了下列文件，建立完整的架构理解：
- `README.md` / `app_fastapi.py` / `config/config.py`
- `modules/{agents,index_manager,chroma_store,data_loader}.py`
- `static/index.html`
- `.env` / `.gitignore` / `requirements.txt`

### 4.2 问题诊断

发现 5 处问题：

1. **`config/config.py` QWEN_API_KEY 硬编码** — 违反全局 CLAUDE.md "禁止硬编码 API Key" 规范；用户本地虽有改动但未提交（git 历史里仍是占位符，密钥未泄露至远端）
2. **`config/config.py` 重复定义 `BASE_DIR`**（第 6 行和第 81 行，后者覆盖前者用于 STATIC_DIR 计算）— 语义混乱
3. **README 与代码不一致**：README 写 `conda create -n medicine`，但 `app_fastapi.py:14` 注释与全局 CLAUDE.md 均使用 `lang`
4. **README 反复提及 `python build_index.py` 但根目录不存在该文件** — `MedicineIndexManager.build_indexes()` 已实现但缺入口
5. **缺少 `scripts/` 与单元测试目录** — 不符合全局 CLAUDE.md 的 AI 应用模板

### 4.3 优化执行（commit a0b9897）

| 文件 | 操作 | 说明 |
|------|------|------|
| `config/config.py` | 重写 | `load_dotenv()` + `os.getenv()` 注入 Qwen/FastAPI/Embedding/Reranker 配置；`QWEN_API_KEY` 缺失时抛 `RuntimeError`；引入 `PROJECT_ROOT` 替代被覆盖的 `BASE_DIR` |
| `build_index.py` | 新建 | argparse 入口，串联 `MedicineDataLoader.process_data()` → `MedicineIndexManager.build_indexes()`；**关键修复**：`data_loader` 返回 `{"product"/"indications"/"comprehensive": ...}` 而 `build_indexes()` 期望 `COLLECTIONS` 长 key（`"medicine_product"` 等），脚本内做了映射 |
| `scripts/run.ps1` | 新建 | 启动 FastAPI，强制 `PYTHONUTF8=1`，默认 `D:/soft/Miniconda/envs/lang/python.exe`，支持 `-NoReload` / `-Port` / `-BindHost` 参数 |
| `scripts/build.ps1` | 新建 | 构建索引，支持 `-Data` 覆盖默认 JSON 路径 |
| `README.md` | 更新 | conda 环境 `medicine` → `lang`；删除"编辑 config.py 填 key"段，改为 `.env` 配置示例；项目结构补 `build_index.py` 与 `scripts/`；启用 Reranker 说明改写为 `.env` 配置 |
| `.gitignore` | 更新 | 增加 `.claude/`（Claude Code 工作目录） |
| `.env` | 清理 | 删除 law 项目残留字段（`GEMINI_*` / `STATUTE_*` / `OPINION_*` / `LLM_PROXY` / `OPINIONS_JSONL` / `ENABLE_*` / `ACTIVE_LLM_PROVIDER`）；`FASTAPI_PORT` 从 8020 改回 8000 与 README 默认值一致；保留用户调优过的 `QWEN_TEMPERATURE=0.3` / `QWEN_MAX_TOKENS=8000` |

### 4.4 验证

```text
[验证 1] from config.config import ... → QWEN_API_KEY loaded: True, PORT: 8000
[验证 2] python build_index.py --help → argparse 输出正常
```

---

## 五、当前状态

### Git 历史
```
a0b9897 (HEAD -> master) refactor(config): dotenv 注入敏感配置，补入口脚本与文档同步
9c12b44 add all_medicine.json with lfs
99fc259 enable git lfs for medicine data
1183c5f feat: 初始提交 — 药品知识库 AI 助手
```

工作区干净，已提交，未推送至远端。

### 配置生效路径

启动时 `config/config.py` → `load_dotenv(.env)` → `os.getenv()` 读取以下变量（缺失走代码内默认值）：

| 变量 | 默认值 | 当前 .env 值 |
|------|--------|------------|
| `QWEN_API_KEY` | 无（必需） | `***（已脱敏，见本地 .env）` |
| `QWEN_API_BASE` | `https://dashscope.aliyuncs.com/...` | 同默认 |
| `QWEN_MODEL_NAME` | `qwen3.5-plus` | 同默认 |
| `QWEN_TEMPERATURE` | `0.2` | `0.3` |
| `QWEN_MAX_TOKENS` | `10000` | `8000` |
| `QWEN_TIMEOUT` | `60` | `60` |
| `FASTAPI_HOST` | `127.0.0.1` | `127.0.0.1` |
| `FASTAPI_PORT` | `8000` | `8000` |
| `MEDICINE_BASE_DIR` | `D:\Embeddings_Data\medicine_llamaindex` | 走默认 |
| `EMBEDDING_MODELS_ROOT` | `D:/Embeddings_Model` | 走默认 |
| `EMBEDDING_DEVICE` | `cuda` | 走默认 |
| `EMBEDDING_PARALLEL_PROCESS` | `true` | 走默认 |
| `EMBEDDING_TARGET_DEVICES` | `cuda:0` | 走默认 |
| `RERANKER_ENABLED` | `false` | 走默认 |

### 离线资源（不在仓库内）

- BGE-M3 模型：`D:/Embeddings_Model/`
- ChromaDB 向量库：`D:/Embeddings_Data/medicine_llamaindex/storage/{product,indications,comprehensive}/`
- 原始数据：`data/all_medicine.json`（git LFS）

---

## 六、如何继续工作

### 环境准备
```powershell
conda activate lang        # python 3.12.3，依赖见 requirements.txt
```

### 启动服务
```powershell
pwsh ./scripts/run.ps1                     # 默认 reload 模式
pwsh ./scripts/run.ps1 -NoReload           # 关闭 reload
pwsh ./scripts/run.ps1 -Port 8030          # 覆盖端口
```

或直接：
```powershell
uvicorn app_fastapi:app --host 127.0.0.1 --port 8000 --reload
```

访问 <http://127.0.0.1:8000>。

### 重建/增量索引
```powershell
pwsh ./scripts/build.ps1                   # 走默认 .env 中的 MEDICINE_BASE_DIR
pwsh ./scripts/build.ps1 -Data D:/.../other.json
```

增量行为：`MedicineIndexManager._build_single_index` 通过 `processed_ids.json` 追踪已向量化文档 id，重跑只处理新增。

### 健康检查
```bash
curl http://127.0.0.1:8000/health
```

---

## 七、已知风险与待办

### 风险

1. **密钥仍以明文存储于本地 `.env`** — 已 gitignore，但建议生产部署改用密钥管理器（如 vault / aliyun KMS）
2. **路径硬编码 Windows 风格**（`D:/Embeddings_Model` 等）— 非 Windows 环境需通过 `.env` 完全覆盖
3. **`EMBEDDING_PARALLEL_PROCESS=true` 双卡并行**仅在 `EMBEDDING_TARGET_DEVICES` 列表 > 1 时实际启用；当前默认单卡 `cuda:0`

### 待办（按优先级）

| 优先级 | 项 | 描述 |
|--------|-----|------|
| 中 | 单元测试 | 全局 CLAUDE.md 要求业务逻辑覆盖率 ≥ 80%；当前 `tests/` 不存在。可从 `parse_nodes`、`_exact_product_search`、`MedicineDataLoader.process_data` 三处入手，LLM 调用走 mock |
| 中 | logging 落盘 | 当前所有模块用 `logging.basicConfig(level=INFO)` 输出到 stderr；全局规范要求落盘到 `logs/`，建议统一 `logging_config.py` |
| 低 | CLAUDE.md 项目级 | 当前仅依赖用户级全局 CLAUDE.md；可在项目根加一份精简的项目级 CLAUDE.md 沉淀 BGE-M3 离线模式、三集合阈值调参经验 |
| 低 | API Key 轮换 | 当前 Qwen key 已硬编码进过本地 config.py（虽未推远端），建议去阿里云控制台轮换一次 |
| 低 | `data_loader` 与 `index_manager` key 接口对齐 | `process_data` 返回短 key，`build_indexes` 期望长 key，目前靠 `build_index.py` 内部映射粘合。长期可在 `data_loader.process_data()` 直接返回长 key，删除映射层 |

---

## 八、关键架构决策（沉淀给后续维护者）

以下是项目中**非显然的设计选择**，看代码不容易看出来：

1. **三集合而非单集合 + 元数据过滤**：因为 BGE-M3 dense embedding 对超短文本（如 3~4 字药品名）匹配能力不足，名称必须单独向量化避免被长说明书稀释
2. **product 集合的元数据快速路径**：`$eq` 命中后**直接 return**，**完全跳过向量检索** — 这是绕过 dense embedding 短文本弱点的关键
3. **indications 集合跳过余弦阈值过滤**：QueryFusionRetriever 的 RRF 分数典型值 ~0.016，远小于余弦相似度阈值 0.4，强行用阈值会过滤光所有结果
4. **HyDE 仅用于 comprehensive 集合**：name/symptom 的查询本身已经比较具体，HyDE 反而引入噪声；只有"高血压能吃什么感冒药"这种问题与说明书向量空间存在系统性偏差时才需要 HyDE
5. **双卡并行向量化必须直接调 `SentenceTransformer.encode_multi_process`**：`HuggingFaceEmbedding.get_text_embedding_batch(**kwargs)` 会把 `parallel_process/target_devices` 吞掉，调用上层 API 不生效（见 `index_manager.py` 注释）
6. **BGE-M3 必须手动指定 `query_instruction`**：llama-index 的 `BGE_MODELS` 列表不含 bge-m3，自动检测返回空字符串，会显著降低检索精度
7. **chromadb `$or` 不支持 `$contains`**：所以 `_exact_product_search` 的包含匹配只能逐字段串行尝试，命中即 break

---

## 九、联系/答疑

如对今日变更有疑问，可参考：
- Git: `git show a0b9897`
- 验证命令：`python -c "from config.config import QWEN_API_KEY; print(bool(QWEN_API_KEY))"`
- 全局规范：`D:\OneDrive\ClaudeCode\.claude\CLAUDE.md`
