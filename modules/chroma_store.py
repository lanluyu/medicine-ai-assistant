# coding:utf-8
"""
自定义 ChromaVectorStore — 适配 chromadb 1.3.0 + llama-index 0.14.9

优化内容（相对初版）：
  [优化 3.1] 支持 MetadataFilters（12 种 FilterOperator → Chroma where 子句）
  [优化 3.2] 支持 MMR 查询模式（调用 llama-index 内置 get_top_k_mmr_embeddings）
"""
import logging
from typing import Any, List, Optional

import chromadb
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

logger = logging.getLogger(__name__)


# FilterOperator → chromadb where 操作符映射
_OP_MAP = {
    FilterOperator.EQ:  "$eq",
    FilterOperator.NE:  "$ne",
    FilterOperator.GT:  "$gt",
    FilterOperator.GTE: "$gte",
    FilterOperator.LT:  "$lt",
    FilterOperator.LTE: "$lte",
    FilterOperator.IN:  "$in",
    FilterOperator.NIN: "$nin",
    # TEXT_MATCH 用 $contains 近似（Chroma 1.3 支持）
    FilterOperator.TEXT_MATCH:             "$contains",
    FilterOperator.TEXT_MATCH_INSENSITIVE: "$contains",
    FilterOperator.CONTAINS:              "$contains",
}


class ChromaVectorStore(BasePydanticVectorStore):
    """ChromaDB 向量存储 llama-index 适配器"""

    stores_text:   bool = True
    flat_metadata: bool = True

    class Config:
        arbitrary_types_allowed = True

    _collection: Any

    def __init__(self, collection: Any):
        super().__init__()
        object.__setattr__(self, "_collection", collection)

    @property
    def client(self) -> Any:
        return self._collection

    # ─── 写入 ─────────────────────────────────────────────────────────
    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        batch_size = 500
        ids_all: List[str] = []
        for start in range(0, len(nodes), batch_size):
            batch      = nodes[start: start + batch_size]
            ids        = [n.node_id for n in batch]
            texts      = [n.get_content(metadata_mode="none") for n in batch]
            embeddings = [n.embedding for n in batch]
            metadatas  = [
                (lambda m: m if m else {"_": "1"})(
                    {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                     for k, v in n.metadata.items() if v is not None}
                )
                for n in batch
            ]
            self._collection.add(
                ids=ids, embeddings=embeddings,
                documents=texts, metadatas=metadatas,
            )
            ids_all.extend(ids)
            logger.info(f"写入进度: {start + len(batch)}/{len(nodes)}")
        return ids_all

    # ─── 查询（分发到 DEFAULT / MMR）─────────────────────────────────
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.mode == VectorStoreQueryMode.MMR:
            return self._query_mmr(query)
        return self._query_default(query)

    # ─── DEFAULT 查询 ─────────────────────────────────────────────────
    def _query_default(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        where = self._build_where(query.filters) if query.filters else None
        kwargs: dict = dict(
            query_embeddings=[query.query_embedding],
            n_results=min(query.similarity_top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        return self._parse_results(results)

    # [优化 3.2] MMR 查询 ──────────────────────────────────────────────
    def _query_mmr(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """最大边际相关度检索：在相关性与多样性之间取得平衡。"""
        try:
            from llama_index.core.indices.query.embedding_utils import (
                get_top_k_mmr_embeddings,
            )
        except ImportError:
            logger.warning("get_top_k_mmr_embeddings 不可用，降级为 DEFAULT")
            return self._query_default(query)

        mmr_threshold = getattr(query, "mmr_threshold", 0.6) or 0.6
        fetch_k = min(query.similarity_top_k * 5, self._collection.count() or 1)

        results = self._collection.query(
            query_embeddings=[query.query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        candidate_embeddings = results.get("embeddings", [[]])[0]
        if not candidate_embeddings:
            return self._parse_results(results)

        indices, _ = get_top_k_mmr_embeddings(
            query_embedding=query.query_embedding,
            embeddings=candidate_embeddings,
            similarity_top_k=query.similarity_top_k,
            mmr_threshold=mmr_threshold,
        )

        nodes, sims, ids = [], [], []
        for idx in indices:
            nid  = results["ids"][0][idx]
            doc  = results["documents"][0][idx] or ""
            meta = results["metadatas"][0][idx] or {}
            dist = results["distances"][0][idx]
            node = TextNode(text=doc, metadata=meta, id_=nid)
            nodes.append(node)
            sims.append(1.0 - float(dist))
            ids.append(nid)
        return VectorStoreQueryResult(nodes=nodes, similarities=sims, ids=ids)

    # ─── 结果解析 ─────────────────────────────────────────────────────
    @staticmethod
    def _parse_results(results: dict) -> VectorStoreQueryResult:
        nodes, sims, ids = [], [], []
        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        node_ids  = results.get("ids",       [[]])[0]
        for doc, meta, dist, nid in zip(docs, metas, distances, node_ids):
            # chromadb cosine: distance = 1 - cosine_similarity → similarity = 1 - distance
            nodes.append(TextNode(text=doc or "", metadata=meta or {}, id_=nid))
            sims.append(1.0 - float(dist))
            ids.append(nid)
        return VectorStoreQueryResult(nodes=nodes, similarities=sims, ids=ids)

    # [优化 3.1] MetadataFilters → Chroma where 子句 ───────────────────
    @staticmethod
    def _build_where(filters: MetadataFilters) -> Optional[dict]:
        if not filters or not filters.filters:
            return None

        conditions = []
        for f in filters.filters:
            chroma_op = _OP_MAP.get(f.operator, "$eq")
            conditions.append({f.key: {chroma_op: f.value}})

        if len(conditions) == 1:
            return conditions[0]

        logic = "$and" if filters.condition == FilterCondition.AND else "$or"
        return {logic: conditions}

    # ─── 删除 ─────────────────────────────────────────────────────────
    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        self._collection.delete(ids=[ref_doc_id])

    def count(self) -> int:
        return self._collection.count()


def get_or_create_chroma_collection(persist_dir: str, collection_name: str) -> Any:
    """创建或加载 ChromaDB 持久化集合（cosine 相似度空间）"""
    client     = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Chroma [{collection_name}] 已加载，文档数: {collection.count()}")
    return collection
