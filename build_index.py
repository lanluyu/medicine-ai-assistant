# coding:utf-8
"""
向量索引构建入口

用法：
  conda activate lang
  python build_index.py [--data <path>]

流程：
  1. MedicineDataLoader 读取并清洗 all_medicine.json
  2. process_data() 为每条药品生成 3 类 Document（product / indications / comprehensive）
  3. MedicineIndexManager.build_indexes() 批量向量化并写入对应 ChromaDB 集合
     已处理文档通过 processed_ids.json 追踪，支持增量更新
"""
import argparse
import logging
import os
import sys
import time

# 离线模式：未配置缓存时 HuggingFaceEmbedding 不会尝试访问网络
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from config.config import (
    DATA_PATH,
    COLLECTION_PRODUCT,
    COLLECTION_INDICATIONS,
    COLLECTION_COMPREHENSIVE,
)
from core.logging_config import setup_logging
from modules.data_loader import MedicineDataLoader
from modules.index_manager import MedicineIndexManager

setup_logging(app_name="build-index")
logger = logging.getLogger("build_index")


# data_loader 返回的短 key → COLLECTIONS 中使用的长 key
_KEY_MAP = {
    "product":       COLLECTION_PRODUCT,
    "indications":   COLLECTION_INDICATIONS,
    "comprehensive": COLLECTION_COMPREHENSIVE,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="构建药品知识库向量索引")
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help=f"原始药品 JSON 路径（默认: {DATA_PATH}）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        logger.error(f"数据文件不存在: {args.data}")
        return 1

    logger.info("=" * 60)
    logger.info(f"开始构建索引 | 数据源: {args.data}")
    t0 = time.time()

    loader = MedicineDataLoader(args.data)
    loader.load_data()
    short_keyed = loader.process_data()
    documents_dict = {_KEY_MAP[k]: v for k, v in short_keyed.items()}

    manager = MedicineIndexManager()
    manager.build_indexes(documents_dict)

    elapsed = time.time() - t0
    logger.info(f"索引构建完成，总耗时 {elapsed:.1f}s")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
