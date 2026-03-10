# coding:utf-8
"""
药品说明书 JSON 数据加载与文档构建模块

设计要点：
  针对 3 种不同查询场景，为每条药品记录生成 3 种文本，
  向量化不同粒度的内容，以最大化检索精度：

  1. product（药品名称检索）：
     只包含各种名称（通用名、商品名、拼音），文本极短。
     用户按名称查找时，查询向量与此文本余弦相似度最高，不被其他内容稀释。

  2. indications（症状/适应症检索）：
     包含适应症全文、相关疾病关键词列表、药理摘要。
     语义丰富，BGE-M3 可将"流鼻涕"与"流涕"等表述映射到相近向量空间。

  3. comprehensive（综合查询）：
     包含名称 + 适应症 + 用法用量 + 禁忌 + 不良反应。
     应对"高血压可以吃感冒药吗"这类混合意图查询。
"""
import re
import json
import html
import hashlib
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from llama_index.core.schema import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MedicineDataLoader:
    """药品数据加载与 LlamaIndex Document 构建"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.raw_data: List[Dict[str, Any]] = []
        self._seen_ids: set = set()  # 全局去重，避免三个集合各自维护

    # ─── 数据加载 ──────────────────────────────────────────────────
    def load_data(self) -> List[Dict[str, Any]]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        with open(self.data_path, encoding="utf-8") as f:
            self.raw_data = json.load(f)
        logger.info(f"加载 {len(self.raw_data)} 条药品数据")
        return self.raw_data

    # ─── 文本清洗 ──────────────────────────────────────────────────
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = html.unescape(text)
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        return text.strip()

    # ─── 唯一 ID 生成 ───────────────────────────────────────────────
    @staticmethod
    def generate_id(item: Dict[str, Any]) -> str:
        approval = (item.get("批准文号") or "").strip()
        if approval:
            return f"wenhao_{approval}"
        name = (item.get("商品名称") or item.get("通用名称") or "unknown").strip()
        h = hashlib.md5(json.dumps(item, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:8]
        return f"hash_{name}_{h}"

    # ─── 精简 metadata（仅保留 LLM 回答时必要字段，避免 Chroma 超限）──
    @staticmethod
    def build_metadata(item: Dict[str, Any]) -> Dict[str, str]:
        """
        Chroma 要求 metadata value 为 str/int/float/bool。
        只保留关键字段，减少存储压力；LLM 需要的完整信息在 page_content 里。
        """
        fields = [
            "标题", "通用名称", "商品名称", "批准文号", "药品分类", "药品性质",
            "生产企业", "相关疾病", "适应症", "主要成份", "规格",
            "用法用量", "不良反应", "禁忌", "注意事项",
            "孕妇及哺乳期妇女用药", "儿童用药", "老人用药",
            "药物相互作用", "药理毒理", "贮藏", "有效期",
        ]
        meta = {}
        for f in fields:
            val = item.get(f, "")
            if val:
                # 截断超长字段（注意事项/药理毒理可能很长）
                meta[f] = str(val)[:1000]
        return meta

    # ─── Document 1：药品名称集合 ───────────────────────────────────
    def create_product_doc(self, item: Dict[str, Any]) -> Optional[Document]:
        """
        向量化内容：只包含名称相关字段。
        目的：用户输入"新康泰克"或"复方盐酸伪麻黄碱"时能精准命中。
        """
        parts = []
        seen_names: set = set()
        for key in ["标题", "通用名称", "商品名称"]:
            v = (item.get(key) or "").strip()
            # 去重：标题与通用名称相同时只加一次，避免通用名在向量中权重翻倍
            if v and v not in seen_names:
                seen_names.add(v)
                parts.append(v)
        # 拼音作为辅助（提升拼音输入查找率）
        pinyin = (item.get("汉语拼音") or "").strip()
        if pinyin:
            parts.append(pinyin)
        # 药品分类/性质作为补充上下文
        for key in ["药品分类", "药品性质"]:
            v = (item.get(key) or "").strip()
            if v:
                parts.append(v)

        text = self.clean_text("\n".join(parts))
        if not text:
            return None

        meta = self.build_metadata(item)
        doc_id = self.generate_id(item)
        return Document(
            text=text,
            metadata=meta,
            id_=doc_id,
            # 让 llama-index 向量化时只用 text，不把 metadata 内容混入向量
            excluded_embed_metadata_keys=list(meta.keys()),
        )

    # ─── Document 2：适应症/症状集合 ────────────────────────────────
    def create_indications_doc(self, item: Dict[str, Any]) -> Optional[Document]:
        """
        向量化内容：症状、疾病关键词 + 适应症全文 + 药理摘要。
        目的：用户描述"鼻塞流涕"或"发烧头痛"时，语义匹配到对应药品。

        关键设计：
          - 将 相关疾病 拆成空格分隔关键词列表（提升关键词命中率）
          - 适应症全文（BGE-M3 语义理解主力）
          - 药理摘要截断前200字（提供机制层语义，如"收缩毛细血管"→"鼻塞"）
        """
        parts = []

        # 药品名称（作为锚点，帮助模型关联药名与症状）
        title = (item.get("标题") or "").strip()
        trade = (item.get("商品名称") or "").strip()
        if title and trade and title != trade:
            parts.append(f"药品：{title}（{trade}）")
        elif title:
            parts.append(f"药品：{title}")

        # 相关疾病——拆分为空格列表，提升关键词匹配
        diseases = (item.get("相关疾病") or "").strip().rstrip(",，")
        if diseases:
            keywords = " ".join(re.split(r"[,，、\s]+", diseases))
            parts.append(f"主治疾病：{keywords}")

        # 适应症全文（最重要）
        indications = (item.get("适应症") or "").strip()
        if indications:
            parts.append(f"适应症：{self.clean_text(indications)}")

        # 主要成分
        ingredients = (item.get("主要成份") or "").strip()
        if ingredients:
            parts.append(f"主要成分：{ingredients}")

        # 药理摘要（截断，保留症状相关机制描述）
        pharma = (item.get("药理毒理") or "").strip()
        if pharma:
            parts.append(f"药理：{self.clean_text(pharma[:250])}")

        text = self.clean_text("\n".join(parts))
        if not text:
            return None

        meta = self.build_metadata(item)
        doc_id = self.generate_id(item)
        return Document(
            text=text,
            metadata=meta,
            id_=doc_id,
            excluded_embed_metadata_keys=list(meta.keys()),
        )

    # ─── Document 3：综合查询集合 ────────────────────────────────────
    def create_comprehensive_doc(self, item: Dict[str, Any]) -> Optional[Document]:
        """
        向量化内容：名称 + 适应症 + 用法用量 + 禁忌 + 不良反应。
        目的：处理复杂问题，如"高血压患者能用什么感冒药"、"哪些药有嗜睡副作用"。
        """
        parts = []

        title   = (item.get("标题") or "").strip()
        generic = (item.get("通用名称") or "").strip()
        trade   = (item.get("商品名称") or "").strip()

        parts.append(f"药品名称：{title}")
        if generic and generic != title:
            parts.append(f"通用名称：{generic}")
        if trade and trade != title:
            parts.append(f"商品名称：{trade}")

        indications = (item.get("适应症") or "").strip()
        if indications:
            parts.append(f"适应症：{self.clean_text(indications)}")

        diseases = (item.get("相关疾病") or "").strip().rstrip(",，")
        if diseases:
            parts.append(f"相关疾病：{diseases}")

        dosage = (item.get("用法用量") or "").strip()
        if dosage:
            parts.append(f"用法用量：{self.clean_text(dosage)}")

        adverse = (item.get("不良反应") or "").strip()
        if adverse:
            parts.append(f"不良反应：{self.clean_text(adverse)}")

        contraindications = (item.get("禁忌") or "").strip()
        if contraindications:
            parts.append(f"禁忌：{self.clean_text(contraindications)}")

        # 注意事项截取前 400 字（常含特殊人群说明，非常关键）
        notes = (item.get("注意事项") or "").strip()
        if notes:
            parts.append(f"注意事项：{self.clean_text(notes[:400])}")

        ingredients = (item.get("主要成份") or "").strip()
        if ingredients:
            parts.append(f"主要成分：{ingredients}")

        text = self.clean_text("\n".join(parts))
        if not text:
            return None

        meta = self.build_metadata(item)
        doc_id = self.generate_id(item)
        return Document(
            text=text,
            metadata=meta,
            id_=doc_id,
            excluded_embed_metadata_keys=list(meta.keys()),
        )

    # ─── 批量处理入口 ────────────────────────────────────────────────
    def process_data(self) -> Dict[str, List[Document]]:
        if not self.raw_data:
            self.load_data()

        product_docs: List[Document]       = []
        indications_docs: List[Document]   = []
        comprehensive_docs: List[Document] = []

        seen_product: set       = set()
        seen_indications: set   = set()
        seen_comprehensive: set = set()

        for i, item in enumerate(self.raw_data):
            if i % 5000 == 0:
                logger.info(f"处理进度: {i}/{len(self.raw_data)}")

            # --- product ---
            doc = self.create_product_doc(item)
            if doc and doc.id_ not in seen_product:
                seen_product.add(doc.id_)
                product_docs.append(doc)

            # --- indications ---
            doc = self.create_indications_doc(item)
            if doc and doc.id_ not in seen_indications:
                seen_indications.add(doc.id_)
                indications_docs.append(doc)

            # --- comprehensive ---
            doc = self.create_comprehensive_doc(item)
            if doc and doc.id_ not in seen_comprehensive:
                seen_comprehensive.add(doc.id_)
                comprehensive_docs.append(doc)

        logger.info(f"文档构建完成 → product: {len(product_docs)}, "
                    f"indications: {len(indications_docs)}, "
                    f"comprehensive: {len(comprehensive_docs)}")

        return {
            "product":       product_docs,
            "indications":   indications_docs,
            "comprehensive": comprehensive_docs,
        }
