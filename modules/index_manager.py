# coding:utf-8
"""
LlamaIndex 向量索引管理模块

职责：
  1. 初始化 llama-index HuggingFaceEmbedding（BGE-M3）
  2. 将 Document 批量向量化并写入三个 ChromaDB 集合
  3. 加载已有索引，提供 VectorIndexRetriever 供 Agent 工具调用

优化内容：
  [优化 1.1] 双卡并行向量化（构建索引时 parallel_process=True + target_devices）
  [优化 1.2] 手动指定 query_instruction（bge-m3 不在 BGE_MODELS 列表）
  [优化 2.1] SimilarityPostprocessor 预创建（按集合缓存，不在每次 search 时 new）
  [优化 2.2] LongContextReorder 单例（全局共享，不在每次 search 时 new）
  [优化 2.3] 可选 SentenceTransformerRerank（需下载 bge-reranker-v2-m3）
  [优化 3.1] _build_single_index 增量更新（processed_ids.json 追踪已处理文档）
  [优化 3.2] indications 集合使用 QueryFusionRetriever（多角度查询 + RRF 融合）
  [优化 3.3] comprehensive 集合使用 HyDE（假设文档向量化，弥合问题/文档向量不对齐）
  [优化 3.4] _exact_product_search 补全 $contains 包含匹配 fallback
"""
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import SecretStr

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import Document, TextNode, QueryBundle, NodeWithScore
from llama_index.core.postprocessor import SimilarityPostprocessor, LongContextReorder
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from modules.chroma_store import ChromaVectorStore, get_or_create_chroma_collection
from config.config import (
    EMBEDDING_MODELS_ROOT,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_QUERY_INSTRUCTION,
    EMBEDDING_TEXT_INSTRUCTION,
    EMBEDDING_PARALLEL_PROCESS,
    EMBEDDING_TARGET_DEVICES,
    RERANKER_ENABLED,
    RERANKER_MODEL_NAME,
    RERANKER_MODELS_ROOT,
    RERANKER_TOP_N,
    RERANKER_DEVICE,
    COLLECTIONS,
    COLLECTION_PRODUCT,
    COLLECTION_INDICATIONS,
    COLLECTION_COMPREHENSIVE,
    QWEN_API_KEY,
    QWEN_API_BASE,
    QWEN_MODEL_NAME,
    QWEN_TEMPERATURE,
    QWEN_TIMEOUT,
    QWEN_MAX_TOKENS
)

logger = logging.getLogger(__name__)


class MedicineIndexManager:
    """
    药品向量索引管理器（llama-index + ChromaDB 双层架构）

    build_indexes()  → 一次性向量化 + 写入（build_index.py 调用）
    load_indexes()   → 加载已有索引（app_fastapi.py / main.py 调用）
    get_retriever()  → 返回 VectorIndexRetriever（agents.py 工具调用）
    search()         → 封装检索 + 后处理（阈值 + 重排）
    """

    def __init__(self):
        self._indexes: Dict[str, VectorStoreIndex] = {}
        self._reranker = None
        self._llm = None
        self._init_embed_model()
        self._init_llm()           # [优化 3.2/3.3] 须在 _init_postprocessors 之前
        self._init_postprocessors()

    # ─── [优化 1.2] 初始化 Embedding 模型 ────────────────────────────
    def _init_embed_model(self):
        """
        加载 BGE-M3，手动指定 query_instruction。

        llama-index utils.py 的 BGE_MODELS 列表不含 bge-m3，
        导致 get_query_instruct_for_model_name 返回空字符串。
        手动传入 query_instruction 修复此问题，使检索精度与预期一致。

        local_files_only=True：模型已在本地时完全禁用 HuggingFace 网络请求，
        避免启动时因无法连接 huggingface.co 而超时报错。
        """
        import os
        import torch

        local_files_only = os.path.exists(EMBEDDING_MODELS_ROOT)
        if local_files_only:
            logger.info(f"使用本地缓存（local_files_only=True）: {EMBEDDING_MODELS_ROOT}")
        else:
            logger.warning(f"本地缓存不存在，将尝试从 HuggingFace 下载: {EMBEDDING_MODEL_NAME}")

        # model_kwargs 传给底层 SentenceTransformer，进一步确保不访问网络
        model_kwargs: dict = {}
        if local_files_only:
            model_kwargs["local_files_only"] = True
        # bfloat16 降低显存（RTX 5060Ti 支持）
        if EMBEDDING_DEVICE == "cuda" and torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
            logger.info("启用 BFloat16 精度（降低显存）")

        logger.info(f"加载 Embedding 模型: {EMBEDDING_MODEL_NAME}")
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder=EMBEDDING_MODELS_ROOT,
            device=EMBEDDING_DEVICE,
            embed_batch_size=EMBEDDING_BATCH_SIZE,
            max_length=EMBEDDING_MAX_LENGTH,
            normalize=True,
            # [优化 1.2] 手动指定 query/text 指令
            query_instruction=EMBEDDING_QUERY_INSTRUCTION,
            text_instruction=EMBEDDING_TEXT_INSTRUCTION,
            # 本地离线模式
            local_files_only=local_files_only,
            model_kwargs=model_kwargs
        )
        Settings.embed_model = embed_model
        logger.info(f"Embedding 模型加载完成，设备: {EMBEDDING_DEVICE}")

    # ─── [优化 3.2/3.3] 初始化 LLM（QueryFusion 变体生成 + HyDE 假设文档）──
    def _init_llm(self):
        """
        初始化 ChatOpenAI 兼容的 LLM，用于：
          - QueryFusionRetriever：生成多角度变体查询（indications 集合）
          - HyDEQueryTransform：生成假设药品说明书段落（comprehensive 集合）

        使用 llama_index.llms.openai.OpenAI 连接 OpenAI 兼容的千问 API。
        max_tokens=512 足够生成短查询/短段落，避免不必要的 token 消耗。
        """
        try:
            # from llama_index.llms.openai import OpenAI as LlamaOpenAI
            # self._llm = LlamaOpenAI(
            #     model=QWEN_MODEL_NAME,
            #     api_key=QWEN_API_KEY,
            #     api_base=QWEN_API_BASE,
            #     temperature=0.3,
            #     max_tokens=512,
            #     timeout=float(QWEN_TIMEOUT),
            # )
            
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                api_key=SecretStr(QWEN_API_KEY),
                base_url=QWEN_API_BASE,
                model=QWEN_MODEL_NAME,
                temperature=QWEN_TEMPERATURE,
                max_tokens=QWEN_MAX_TOKENS,
                timeout=QWEN_TIMEOUT,
            )
            Settings.llm = self._llm
            logger.info(f"LLM 初始化完成（{QWEN_MODEL_NAME}）")
        except Exception as e:
            logger.warning(f"LLM 初始化失败，QueryFusion/HyDE 将不可用: {e}")
            self._llm = None

    # ─── [优化 2.x] 初始化后处理器 ───────────────────────────────────
    def _init_postprocessors(self):
        """
        初始化检索后处理器：
          [优化 2.1] _sim_filters — 按集合预创建 SimilarityPostprocessor（不在每次 search 时 new）
          [优化 2.2] _reorder    — LongContextReorder 全局单例
          [优化 2.3] _reranker   — 可选 SentenceTransformerRerank
          [优化 3.3] _hyde       — HyDEQueryTransform（comprehensive 集合）
        """
        # [优化 2.1] 按集合预创建阈值过滤器
        self._sim_filters: Dict[str, SimilarityPostprocessor] = {
            name: SimilarityPostprocessor(similarity_cutoff=cfg["score_threshold"])
            for name, cfg in COLLECTIONS.items()
        }
        # [优化 2.2] LongContextReorder 全局单例
        self._reorder = LongContextReorder()

        # [优化 2.3] 可选 Reranker
        if RERANKER_ENABLED:
            try:
                from llama_index.core.postprocessor import SentenceTransformerRerank
                self._reranker = SentenceTransformerRerank(
                    model=RERANKER_MODEL_NAME,
                    top_n=RERANKER_TOP_N,
                    device=RERANKER_DEVICE,
                    keep_retrieval_score=True,
                )
                logger.info(f"Reranker 已启用: {RERANKER_MODEL_NAME}")
            except Exception as e:
                logger.warning(f"Reranker 初始化失败，跳过: {e}")
                self._reranker = None
        else:
            logger.info("Reranker 未启用（RERANKER_ENABLED=False）")

        # [优化 3.3] HyDE for comprehensive 集合
        if self._llm is not None:
            self._hyde = HyDEQueryTransform(llm=self._llm, include_original=True)
            logger.info("HyDE QueryTransform 已初始化（用于 comprehensive 集合）")
        else:
            self._hyde = None
            logger.info("HyDE 未启用（LLM 未初始化）")

    # ─── 构建索引（一次性，耗时较长）──────────────────────────────────
    def build_indexes(self, documents_dict: Dict[str, List[Document]]) -> bool:
        """
        批量向量化并写入 ChromaDB。

        [优化 1.1] 构建时使用 parallel_process=True + target_devices 双卡并行。
        """
        for coll_name, docs in documents_dict.items():
            cfg = COLLECTIONS[coll_name]
            self._build_single_index(coll_name, cfg["dir"], docs)
        return True

    # ─── [优化 3.1] 增量更新辅助方法 ─────────────────────────────────
    @staticmethod
    def _load_processed_ids(persist_dir: str) -> set:
        """加载已处理文档 ID 集合（processed_ids.json）。"""
        ids_file = Path(persist_dir) / "processed_ids.json"
        if ids_file.exists():
            with open(ids_file, encoding="utf-8") as f:
                return set(json.load(f))
        return set()

    @staticmethod
    def _save_processed_ids(persist_dir: str, ids: set) -> None:
        """持久化已处理文档 ID 集合。"""
        ids_file = Path(persist_dir) / "processed_ids.json"
        with open(ids_file, "w", encoding="utf-8") as f:
            json.dump(list(ids), f, ensure_ascii=False)

    def _build_single_index(
        self,
        collection_name: str,
        persist_dir: str,
        documents: List[Document],
    ) -> None:
        """
        构建单个集合的向量索引。

        [优化 3.1] 增量更新：通过 processed_ids.json 追踪已向量化的文档 ID，
        仅对新增/未处理的文档进行向量化，已处理的直接跳过。
        相比原来 count > 0 即跳过整个集合，本方案支持数据库新增药品后的增量写入。
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"构建集合: {collection_name}  文档数: {len(documents)}")
        logger.info(f"存储目录: {persist_dir}")

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        chroma_coll = get_or_create_chroma_collection(persist_dir, collection_name)
        vector_store = ChromaVectorStore(collection=chroma_coll)

        # [优化 3.1] 过滤已处理文档，只保留新增文档
        processed_ids = self._load_processed_ids(persist_dir)
        new_documents = [doc for doc in documents if doc.id_ not in processed_ids]

        if not new_documents:
            logger.info(
                f"集合 [{collection_name}] 全部 {len(documents)} 条文档已处理，无需更新"
            )
            self._indexes[collection_name] = VectorStoreIndex.from_vector_store(vector_store)
            return

        logger.info(
            f"增量更新: 新增 {len(new_documents)} 条 / 已跳过 {len(processed_ids)} 条"
        )
        texts = [doc.text for doc in new_documents]
        logger.info(f"开始批量向量化 {len(texts)} 条文档...")
        t0 = time.time()

        # [优化 1.1] 双卡并行向量化
        #
        # ⚠️ 注意：get_text_embedding_batch(**kwargs) 中的 parallel_process/target_devices
        # 会被基类的 **kwargs 签名吞掉，实际完全不生效。
        # 正确做法：直接调用底层 SentenceTransformer.encode_multi_process，
        # 对全量文本一次性开多进程池，避免每个 128 条 mini-batch 都重建进程池的开销。
        if EMBEDDING_PARALLEL_PROCESS and len(EMBEDDING_TARGET_DEVICES) > 1:
            logger.info(f"双卡并行模式: {EMBEDDING_TARGET_DEVICES}")
            # 取 HuggingFaceEmbedding 底层的 SentenceTransformer 实例
            st_model = Settings.embed_model._model
            pool = st_model.start_multi_process_pool(
                target_devices=EMBEDDING_TARGET_DEVICES
            )
            try:
                raw = st_model.encode_multi_process(
                    texts,
                    pool=pool,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    prompt_name="text",          # 对应 text_instruction（BGE-M3 为 ""）
                    normalize_embeddings=True,
                    show_progress_bar=True,
                )
            finally:
                st_model.stop_multi_process_pool(pool=pool)
            embeddings = raw.tolist()
        else:
            embeddings = Settings.embed_model.get_text_embedding_batch(
                texts, show_progress=True
            )
        logger.info(f"向量化完成，耗时 {time.time()-t0:.1f}s")

        # 构建 TextNode 列表（携带 embedding）
        nodes: List[TextNode] = []
        for doc, emb in zip(new_documents, embeddings):
            node = TextNode(
                text=doc.text,
                metadata=doc.metadata,
                id_=doc.id_,
                embedding=emb,
            )
            nodes.append(node)

        # 写入 Chroma（增量追加）
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=[], storage_context=storage_ctx)
        index.insert_nodes(nodes)

        # [优化 3.1] 更新并持久化已处理 ID
        processed_ids.update(doc.id_ for doc in new_documents)
        self._save_processed_ids(persist_dir, processed_ids)

        self._indexes[collection_name] = index
        logger.info(f"集合 [{collection_name}] 写入 {len(nodes)} 条，累计已处理 {len(processed_ids)} 条")

    # ─── 加载已有索引 ─────────────────────────────────────────────────
    def load_indexes(self) -> None:
        """从 ChromaDB 加载已有索引，不重新向量化。"""
        for coll_name, cfg in COLLECTIONS.items():
            persist_dir = cfg["dir"]
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            chroma_coll = get_or_create_chroma_collection(persist_dir, coll_name)
            count = chroma_coll.count()
            if count == 0:
                logger.warning(f"集合 [{coll_name}] 为空，请先运行 build_index.py")
            vector_store = ChromaVectorStore(collection=chroma_coll)
            self._indexes[coll_name] = VectorStoreIndex.from_vector_store(vector_store)
            logger.info(f"集合 [{coll_name}] 加载完成，文档数: {count}")

    # ─── 获取检索器 ──────────────────────────────────────────────────
    # 中文变体查询生成 Prompt（替换 QueryFusionRetriever 的默认英文 Prompt）
    _CHINESE_QUERY_GEN_PROMPT = (
        "你是一位专业的药品信息检索助手。"
        "请基于以下查询，生成 {num_queries} 个语义不同但主题相关的中文搜索查询，"
        "每行一个，用于从药品知识库中检索相关信息：\n"
        "原始查询：{query}\n"
        "生成的查询：\n"
    )

    def get_retriever(
        self, collection_name: str, top_k: Optional[int] = None
    ) -> Union[VectorIndexRetriever, QueryFusionRetriever]:
        """
        返回指定集合的检索器。

        [优化 3.2] indications 集合返回 QueryFusionRetriever：
          - 自动生成 3 个语义变体查询（共 4 路）
          - RRF 算法融合多路结果，提升症状查询召回率
          - 使用中文 Prompt，保证生成查询与药品语境匹配
        其他集合返回标准 VectorIndexRetriever。
        """
        if collection_name not in self._indexes:
            raise KeyError(f"集合 [{collection_name}] 未加载，请先调用 load_indexes()")
        cfg = COLLECTIONS[collection_name]
        k = top_k or cfg["top_k"]

        base_retriever = VectorIndexRetriever(
            index=self._indexes[collection_name],
            similarity_top_k=k,
        )

        # [优化 3.2] indications 集合：多查询融合（仅在 LLM 可用时生效）
        if collection_name == COLLECTION_INDICATIONS and self._llm is not None:
            return QueryFusionRetriever(
                retrievers=[base_retriever],
                llm=self._llm,
                query_gen_prompt=self._CHINESE_QUERY_GEN_PROMPT,
                mode=FUSION_MODES.RECIPROCAL_RANK,
                num_queries=4,       # 生成 3 个变体 + 原始查询，共 4 路
                similarity_top_k=k,
                use_async=False,     # 同步模式，与项目现有同步架构一致
                verbose=False,
            )

        return base_retriever

    # ─── 元数据精确匹配（product 集合快速路径）────────────────────────
    def _exact_product_search(self, query: str) -> List[Any]:
        """
        在 product 集合中，通过元数据精确/包含匹配快速查找药品。

        动机：BGE-M3 dense embedding 对超短品牌名（如"新康泰克"）精确匹配能力
        不足——品牌名只有 3~4 个汉字，文档向量被长通用名稀释后余弦相似度偏低。
        通过 ChromaDB 元数据直接匹配可绕过向量相似度限制，精确召回目标药品。

        [优化 3.4] 匹配策略（按优先级）：
          1. 精确等值匹配：$or 同时匹配商品名称 / 通用名称 / 标题
          2. 包含匹配（fallback）：逐字段尝试 $contains，任一命中即返回
             仅在精确匹配无结果且 query 长度 >= 2 时触发
        """
        if COLLECTION_PRODUCT not in self._indexes:
            return []

        try:
            store = self._indexes[COLLECTION_PRODUCT]._vector_store
            coll  = store._collection

            # ── 阶段 1：精确等值匹配（$eq）──────────────────────────
            results = coll.get(
                where={"$or": [
                    {"商品名称": {"$eq": query}},
                    {"通用名称": {"$eq": query}},
                    {"标题":    {"$eq": query}},
                ]},
                include=["documents", "metadatas"],
            )

            # ── 阶段 2：包含匹配 fallback（$contains）───────────────
            # [优化 3.4] 原代码缺失此步骤：注释声明有包含匹配但实现中未做
            # Chroma 不支持 $or + $contains 组合，逐字段串行尝试
            if not results["ids"] and len(query) >= 2:
                for field in ["商品名称", "通用名称", "标题"]:
                    try:
                        results = coll.get(
                            where={field: {"$contains": query}},
                            include=["documents", "metadatas"],
                        )
                        if results["ids"]:
                            logger.info(
                                f"包含匹配命中（字段={field!r}，query={query!r}）"
                            )
                            break
                    except Exception:
                        continue

            if not results["ids"]:
                return []

            nodes = []
            for nid, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            ):
                node = TextNode(text=doc or "", metadata=meta or {}, id_=nid)
                nodes.append(NodeWithScore(node=node, score=1.0))

            logger.info(f"元数据匹配命中 {len(nodes)} 条（query={query!r}），跳过向量检索")
            return nodes

        except Exception as e:
            logger.warning(f"元数据精确匹配失败，降级为向量检索: {e}")
            return []

    # ─── 封装检索 + 后处理 ──────────────────────────────────────────
    def search(
        self,
        collection_name: str,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Any]:
        """
        执行向量检索并经过后处理器链：
          1. [product 集合] 元数据精确/包含匹配快速路径（_exact_product_search）
          2. [comprehensive 集合] HyDE 变换 QueryBundle（弥合问题/说明书向量不对齐）
          3. 向量检索（indications→QueryFusionRetriever，其他→VectorIndexRetriever）
          4. SimilarityPostprocessor — 阈值过滤（indications 跳过，RRF 分数不兼容余弦阈值）
          5. SentenceTransformerRerank（可选）— 精排
          6. LongContextReorder — 重排缓解"迷失中间"
        """
        # ── [快速路径] product 集合先尝试精确/包含元数据匹配 ─────────
        if collection_name == COLLECTION_PRODUCT:
            exact_nodes = self._exact_product_search(query)
            if exact_nodes:
                # [优化 2.2] 精确命中后仍做 LongContextReorder（结果 > 3 时）
                if len(exact_nodes) > 3:
                    exact_nodes = self._reorder.postprocess_nodes(exact_nodes)
                return exact_nodes

        retriever = self.get_retriever(collection_name, top_k)
        cfg = COLLECTIONS[collection_name]
        threshold = score_threshold if score_threshold is not None else cfg["score_threshold"]

        # ── [优化 3.3] comprehensive 集合：HyDE 变换 QueryBundle ──────
        # HyDEQueryTransform 让 LLM 生成"假设药品说明书段落"，
        # VectorIndexRetriever._retrieve 会用 query_bundle.embedding_strs
        # 调用 get_agg_embedding_from_queries() 取平均向量（见 retriever.py:109）
        query_bundle: Union[str, QueryBundle] = query
        if collection_name == COLLECTION_COMPREHENSIVE and self._hyde is not None:
            try:
                query_bundle = self._hyde.run(QueryBundle(query))
                logger.info("HyDE 变换已应用（comprehensive 集合）")
            except Exception as e:
                logger.warning(f"HyDE 变换失败，降级为原始查询: {e}")
                query_bundle = query

        logger.info(f"检索 [{collection_name}] | query={query!r} | threshold={threshold}")
        t0 = time.time()
        nodes = retriever.retrieve(query_bundle)
        logger.info(f"检索耗时 {time.time()-t0:.3f}s，原始结果 {len(nodes)} 条")

        # ── [优化 2.1] 阈值过滤（使用预创建的实例）──────────────────
        # indications 使用 QueryFusionRetriever + RRF 模式，RRF 分数远小于余弦相似度
        # （典型值 ~0.016），不能用余弦阈值过滤，跳过此步依赖 top_k 控制结果数
        if collection_name != COLLECTION_INDICATIONS:
            nodes = self._sim_filters[collection_name].postprocess_nodes(nodes)
            logger.info(f"阈值过滤后剩余 {len(nodes)} 条")

        if not nodes:
            return nodes

        # [优化 2.3] 可选精排
        if self._reranker is not None:
            nodes = self._reranker.postprocess_nodes(
                nodes, query_bundle=QueryBundle(query)
            )
            logger.info(f"Reranker 精排后剩余 {len(nodes)} 条")

        # [优化 2.2] LongContextReorder（使用单例，仅当结果 > 3 时有意义）
        if len(nodes) > 3:
            nodes = self._reorder.postprocess_nodes(nodes)

        return nodes

    def is_loaded(self) -> bool:
        return len(self._indexes) == 3

    def get_collection_count(self, collection_name: str) -> int:
        if collection_name in self._indexes:
            store = self._indexes[collection_name]._vector_store
            if hasattr(store, "count"):
                return store.count()
        return -1
