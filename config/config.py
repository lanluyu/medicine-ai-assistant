# coding:utf-8
import os

# 项目根目录
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = r"D:\Embeddings_Data\medicine_llamaindex"

# 原始数据路径
DATA_PATH = os.path.join(BASE_DIR, "data", "all_medicine.json")

# ─── 向量存储（ChromaDB）──────────────────────────────────────────────
STORAGE_ROOT              = os.path.join(BASE_DIR, "storage")
STORAGE_PRODUCT_DIR       = os.path.join(STORAGE_ROOT, "product")
STORAGE_INDICATIONS_DIR   = os.path.join(STORAGE_ROOT, "indications")
STORAGE_COMPREHENSIVE_DIR = os.path.join(STORAGE_ROOT, "comprehensive")

COLLECTION_PRODUCT       = "medicine_product"
COLLECTION_INDICATIONS   = "medicine_indications"
COLLECTION_COMPREHENSIVE = "medicine_comprehensive"

# 三集合配置：检索数量与相似度阈值
# product 集合已有元数据精确匹配快速路径，向量检索作为 fallback，阈值适当放宽
COLLECTIONS = {
    COLLECTION_PRODUCT: {
        "dir": STORAGE_PRODUCT_DIR,
        "score_threshold": 0.40,
        "top_k": 50,
    },
    COLLECTION_INDICATIONS: {
        "dir": STORAGE_INDICATIONS_DIR,
        "score_threshold": 0.40,
        "top_k": 50,
    },
    COLLECTION_COMPREHENSIVE: {
        "dir": STORAGE_COMPREHENSIVE_DIR,
        "score_threshold": 0.45,
        "top_k": 50,
    },
}

# ─── Embedding 模型配置 ───────────────────────────────────────────────
# 模型文件统一存放于 D:\Embeddings_Model（HuggingFace Hub 格式）
EMBEDDING_MODELS_ROOT = "D:/Embeddings_Model"
EMBEDDING_MODEL_NAME  = "BAAI/bge-m3"

# [优化 1.2] BGE-M3 不在 llama-index utils.py 的 BGE_MODELS 列表中，
# 需手动指定 query_instruction，否则 get_query_instruct_for_model_name 返回 ""
# document 端保持空字符串（BGE-M3 官方建议 document 不加前缀）
EMBEDDING_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关药品信息："
EMBEDDING_TEXT_INSTRUCTION  = ""

EMBEDDING_DEVICE     = "cuda"
EMBEDDING_BATCH_SIZE = 128
EMBEDDING_MAX_LENGTH = 512

# [优化 1.1] 双卡并行向量化（构建索引时生效）
EMBEDDING_PARALLEL_PROCESS = True
EMBEDDING_TARGET_DEVICES   = ["cuda:0"]
# EMBEDDING_TARGET_DEVICES   = ["cuda:0", "cuda:1"]

# ─── Reranker 配置（二阶段精排）──────────────────────────────────────
# 如需启用 CrossEncoder 精排，将模型下载到 D:\Embeddings_Model 后设为 True
# 推荐模型：BAAI/bge-reranker-v2-m3（与 bge-m3 同系列，中文友好）
RERANKER_ENABLED    = False
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_MODELS_ROOT = EMBEDDING_MODELS_ROOT
RERANKER_TOP_N      = 6
RERANKER_DEVICE     = "cuda"

# ─── Qwen 大语言模型配置 ─────────────────────────────────────────────
QWEN_API_KEY     = "替换你自己的 KEY"
QWEN_API_BASE    = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL_NAME  = "qwen3.5-plus"
QWEN_TEMPERATURE = 0.2
QWEN_MAX_TOKENS  = 10000
QWEN_TIMEOUT     = 60

# ─── FastAPI 服务配置 ─────────────────────────────────────────────────
FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8000
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR   = os.path.join(BASE_DIR, "static")
