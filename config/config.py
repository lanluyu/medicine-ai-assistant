# coding:utf-8
"""
全局配置 — 路径、Embedding、Reranker、LLM、FastAPI

敏感配置（API Key / 服务端口等）通过 .env 注入，
其他相对稳定的离线资源路径直接写在本文件，便于版本管理与团队一致。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（用于定位 static/ 等仓库内资源）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载项目根目录下的 .env（不覆盖已存在的环境变量）
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_env(key: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(
            f"环境变量 {key} 未配置，请在项目根目录创建 .env 并设置该值"
        )
    return val or ""


# ─── 向量索引根目录（离线构建产物，独立于代码仓库）────────────────────
BASE_DIR = _get_env(
    "MEDICINE_BASE_DIR",
    default=r"D:\Embeddings_Data\medicine_llamaindex",
)

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
EMBEDDING_MODELS_ROOT = _get_env(
    "EMBEDDING_MODELS_ROOT",
    default="D:/Embeddings_Model",
)
EMBEDDING_MODEL_NAME  = "BAAI/bge-m3"

# BGE-M3 不在 llama-index utils.py 的 BGE_MODELS 列表中，
# 需手动指定 query_instruction，否则 get_query_instruct_for_model_name 返回 ""
# document 端保持空字符串（BGE-M3 官方建议 document 不加前缀）
EMBEDDING_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关药品信息："
EMBEDDING_TEXT_INSTRUCTION  = ""

EMBEDDING_DEVICE     = _get_env("EMBEDDING_DEVICE", default="cuda")
EMBEDDING_BATCH_SIZE = int(_get_env("EMBEDDING_BATCH_SIZE", default="128"))
EMBEDDING_MAX_LENGTH = int(_get_env("EMBEDDING_MAX_LENGTH", default="512"))

# 双卡并行向量化（构建索引时生效）
EMBEDDING_PARALLEL_PROCESS = _get_env("EMBEDDING_PARALLEL_PROCESS", default="true").lower() == "true"
EMBEDDING_TARGET_DEVICES   = [
    d.strip() for d in _get_env("EMBEDDING_TARGET_DEVICES", default="cuda:0").split(",") if d.strip()
]

# ─── Reranker 配置（二阶段精排）──────────────────────────────────────
# 如需启用 CrossEncoder 精排，将模型下载到 D:\Embeddings_Model 后设为 True
# 推荐模型：BAAI/bge-reranker-v2-m3（与 bge-m3 同系列，中文友好）
RERANKER_ENABLED     = _get_env("RERANKER_ENABLED", default="false").lower() == "true"
RERANKER_MODEL_NAME  = _get_env("RERANKER_MODEL_NAME", default="BAAI/bge-reranker-v2-m3")
RERANKER_MODELS_ROOT = EMBEDDING_MODELS_ROOT
RERANKER_TOP_N       = int(_get_env("RERANKER_TOP_N", default="6"))
RERANKER_DEVICE      = _get_env("RERANKER_DEVICE", default="cuda")

# ─── Qwen 大语言模型配置（必须从 .env 注入，禁止硬编码）─────────────
QWEN_API_KEY     = _get_env("QWEN_API_KEY", required=True)
QWEN_API_BASE    = _get_env(
    "QWEN_API_BASE",
    default="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL_NAME  = _get_env("QWEN_MODEL_NAME", default="qwen3.5-plus")
QWEN_TEMPERATURE = float(_get_env("QWEN_TEMPERATURE", default="0.2"))
QWEN_MAX_TOKENS  = int(_get_env("QWEN_MAX_TOKENS", default="10000"))
QWEN_TIMEOUT     = int(_get_env("QWEN_TIMEOUT", default="60"))

# ─── FastAPI 服务配置 ─────────────────────────────────────────────────
FASTAPI_HOST = _get_env("FASTAPI_HOST", default="127.0.0.1")
FASTAPI_PORT = int(_get_env("FASTAPI_PORT", default="8000"))

# 仓库内静态资源目录
STATIC_DIR = str(PROJECT_ROOT / "static")
