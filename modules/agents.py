# coding:utf-8
"""
LangChain Agent 模块

设计：
  - 3 个 @tool 工具，分别对应三个向量集合
  - 每个工具调用 MedicineIndexManager.search() 获取 llama-index NodeWithScore 列表
  - parse_nodes() 将 NodeWithScore 格式化为 LLM 可读字符串
  - MedicineAgent 用 create_agent（LangGraph ReAct）驱动 Qwen 大模型

优化内容：
  [优化 4.1] InMemorySaver checkpointer — 多轮对话记忆（thread_id 隔离）
  [优化 4.2] astream_events 流式输出 — FastAPI SSE 实时推送 token
"""
import httpx
import logging
import uuid
from typing import AsyncIterator, Optional
from pydantic import BaseModel, Field, SecretStr

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver

from modules.index_manager import MedicineIndexManager
from config.config import (
    QWEN_API_KEY,
    QWEN_API_BASE,
    QWEN_MODEL_NAME,
    QWEN_TEMPERATURE,
    QWEN_MAX_TOKENS,
    QWEN_TIMEOUT,
    COLLECTION_PRODUCT,
    COLLECTION_INDICATIONS,
    COLLECTION_COMPREHENSIVE,
)

# 业务模块不配置日志格式，由入口 core.logging_config.setup_logging() 统一管理
logger = logging.getLogger(__name__)

# ─── 全局单例（延迟初始化）────────────────────────────────────────────
_index_manager: Optional[MedicineIndexManager] = None


def get_index_manager() -> MedicineIndexManager:
    global _index_manager
    if _index_manager is None:
        logger.info("初始化 MedicineIndexManager（首次加载）...")
        _index_manager = MedicineIndexManager()
        _index_manager.load_indexes()
    return _index_manager


# ─── 结果格式化（NodeWithScore → str）────────────────────────────────
def parse_nodes(nodes_with_score, max_results: int = 6) -> str:
    """将 llama-index NodeWithScore 列表去重并格式化为 LLM 可读字符串。"""
    seen_names: dict = {}

    for node_ws in nodes_with_score:
        node  = node_ws.node
        score = node_ws.score or 0.0
        meta  = node.metadata or {}

        name = meta.get("通用名称") or meta.get("标题") or meta.get("商品名称") or ""
        if not name:
            continue
        if name not in seen_names or seen_names[name]["score"] < score:
            seen_names[name] = {"score": score, "meta": meta}

    sorted_items = sorted(seen_names.values(), key=lambda x: x["score"], reverse=True)[:max_results]

    if not sorted_items:
        return "未找到相关药品信息。"

    results = []
    for i, item in enumerate(sorted_items, 1):
        meta  = item["meta"]
        score = item["score"]
        block = [f"【药品 {i}】（相似度: {score:.3f}）"]
        block.append(f"  药品名称: {meta.get('标题','')}")
        if meta.get("商品名称"):
            block.append(f"  商品名称: {meta['商品名称']}")
        if meta.get("通用名称"):
            block.append(f"  通用名称: {meta['通用名称']}")
        if meta.get("药品性质"):
            block.append(f"  药品性质: {meta['药品性质']}")
        if meta.get("适应症"):
            block.append(f"  适应症: {meta['适应症'][:200]}")
        if meta.get("用法用量"):
            block.append(f"  用法用量: {meta['用法用量']}")
        if meta.get("不良反应"):
            block.append(f"  不良反应: {meta['不良反应'][:150]}")
        if meta.get("禁忌"):
            block.append(f"  禁忌: {meta['禁忌'][:150]}")
        if meta.get("注意事项"):
            block.append(f"  注意事项: {meta['注意事项'][:200]}")
        if meta.get("主要成份"):
            block.append(f"  主要成分: {meta['主要成份']}")
        if meta.get("生产企业"):
            block.append(f"  生产企业: {meta['生产企业']}")
        results.append("\n".join(block))

    return "\n\n".join(results)


# ─── [优化 4.1] Tool 参数 Schema（Pydantic）──────────────────────────
class NameInput(BaseModel):
    medicine_name: str = Field(description="药品名称（商品名或通用名），如：新康泰克、阿莫西林")

class SymptomInput(BaseModel):
    symptom: str = Field(description="症状或疾病名称，如：鼻塞流涕、发烧头痛、胃痛腹泻")

class ComprehensiveInput(BaseModel):
    query_text: str = Field(description="综合查询文本，包含症状、药品、注意事项等复杂问题")


# ─── LangChain Tools ─────────────────────────────────────────────────

@tool(args_schema=NameInput)
def search_medicine_by_name(medicine_name: str) -> str:
    """
    根据药品名称（商品名或通用名）检索药品详细信息。
    仅当用户明确提到具体药品名称（如：新康泰克、阿莫西林、感冒灵）时使用此工具。
    如果用户描述的是症状或疾病，请勿使用此工具。
    """
    logger.info(f"[Tool] search_medicine_by_name: {medicine_name!r}")
    mgr = get_index_manager()
    nodes = mgr.search(collection_name=COLLECTION_PRODUCT, query=medicine_name)
    return parse_nodes(nodes, max_results=5)


@tool(args_schema=SymptomInput)
def search_medicine_by_symptom(symptom: str) -> str:
    """
    根据症状、疾病名称或适应症检索推荐药品。
    适用于用户询问"得了某种病吃什么药"或描述身体不适症状的场景。
    例如：鼻塞流涕、发烧头痛、胃痛腹泻、失眠等。
    """
    logger.info(f"[Tool] search_medicine_by_symptom: {symptom!r}")
    mgr = get_index_manager()
    nodes = mgr.search(collection_name=COLLECTION_INDICATIONS, query=symptom)
    return parse_nodes(nodes, max_results=6)


@tool(args_schema=ComprehensiveInput)
def search_medicine_by_comprehensive(query_text: str) -> str:
    """
    综合查询：同时结合药品名称、症状、用药注意事项、禁忌进行检索。
    适用于复杂问题，例如：
      - "高血压患者感冒了可以吃什么药？"
      - "哪些感冒药含有嗜睡成分？"
      - "儿童发烧推荐哪些退烧药？"
    """
    logger.info(f"[Tool] search_medicine_by_comprehensive: {query_text!r}")
    mgr = get_index_manager()
    nodes = mgr.search(collection_name=COLLECTION_COMPREHENSIVE, query=query_text)
    return parse_nodes(nodes, max_results=6)


TOOLS = [
    search_medicine_by_name,
    search_medicine_by_symptom,
    search_medicine_by_comprehensive,
]


# ─── Token 追踪回调 ───────────────────────────────────────────────────
class TokenUsageTracker(BaseCallbackHandler):
    def __init__(self):
        self.call_count = 0
        self.total_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.call_count += 1
        logger.info(f"[LLM] 第 {self.call_count} 次调用")

    def on_llm_end(self, response, **kwargs):
        try:
            info = response.model_dump()
            gen = info.get("generations", [[]])[0][0] or {}
            msg = gen.get("message", {})
            usage = msg.get("response_metadata", {}).get("token_usage", {})
            if usage:
                total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                self.total_tokens += total
                logger.info(
                    f"[LLM] tokens: prompt={usage.get('prompt_tokens',0)}, "
                    f"completion={usage.get('completion_tokens',0)}, total={total}"
                )
        except Exception:
            pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        logger.info(f"[Tool] 调用: {serialized.get('name','')} | 输入: {str(input_str)[:80]}")

    def on_tool_end(self, output, **kwargs):
        logger.info(f"[Tool] 输出: {str(output)[:120]}")


# ─── MedicineAgent ────────────────────────────────────────────────────
class MedicineAgent:
    """
    药品知识库 Agent

    llama-index 负责：向量化（HuggingFaceEmbedding）+ 向量查询
    LangChain 负责：Agent 逻辑（create_agent / ReAct）+ LLM 对话（Qwen）

    [优化 4.1] InMemorySaver checkpointer 实现多轮记忆，每个 session 用 thread_id 隔离。
    [优化 4.2] astream_events 支持 SSE 流式输出，前端实时显示 token。
    """

    _SYSTEM_PROMPT = (
        "你是一个专业的药品信息助手，基于药品说明书知识库回答用户问题。\n"
        "工具选择策略：\n"
        "  - 用户提到具体药品名称 → 使用 search_medicine_by_name\n"
        "  - 用户描述症状/疾病 → 使用 search_medicine_by_symptom\n"
        "  - 用户问题复杂（涉及禁忌/注意事项/特殊人群）→ 使用 search_medicine_by_comprehensive\n"
        "回答要求：提供准确、完整的药品信息，并提醒用户用药前咨询医生或药师。\n"
        "免责声明：本助手仅供参考，不构成医疗建议。"
    )

    def __init__(self):
        logger.info("初始化 MedicineAgent...")
        if not QWEN_API_KEY:
            raise ValueError("QWEN_API_KEY 未配置")

        # 预加载索引
        get_index_manager()

        self.tracker = TokenUsageTracker()
        # [优化 4.1] InMemorySaver — 进程内多轮记忆
        self._checkpointer = InMemorySaver()
        self._build_qwen_agent()

    def _build_qwen_agent(self):
        # 设置 Qwen 客户端
        logging.info('===== Set Qwen ChatOpenAI =====')
        chat_client = ChatOpenAI(
            api_key=SecretStr(QWEN_API_KEY),
            base_url=QWEN_API_BASE,
            model=QWEN_MODEL_NAME,
            temperature=QWEN_TEMPERATURE,
            max_tokens=QWEN_MAX_TOKENS,
            timeout=QWEN_TIMEOUT,
            callbacks=[self.tracker],
            http_client=httpx.Client(trust_env=False),
        )

        # [优化 4.1] 传入 checkpointer 使 Agent 支持多轮记忆
        self.graph = create_agent(
            model=chat_client,
            tools=TOOLS,
            system_prompt=self._SYSTEM_PROMPT,
            checkpointer=self._checkpointer,
        )
        logger.info("Qwen Agent 构建完成（含 InMemorySaver checkpointer）")

    def new_session(self) -> str:
        """生成新的会话 ID"""
        return str(uuid.uuid4())

    # ─── 同步对话（main.py CLI 用）─────────────────────────────────
    def chat(self, message: str, thread_id: str) -> str:
        """
        同步对话，返回最终回复文本。
        thread_id 唯一标识一个会话，checkpointer 自动维护该会话的消息历史。
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(
            {"messages": [("user", message)]},
            config=config,
        )
        # 取最后一条 AIMessage
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
                return msg.content
        return "抱歉，未能生成回复。"

    # ─── [优化 4.2] 异步流式对话（FastAPI SSE 用）──────────────────
    async def astream_tokens(
        self, message: str, thread_id: str
    ) -> AsyncIterator[str]:
        """
        异步流式输出 token，供 FastAPI SSE 端点使用。

        astream_events v2 事件流：
          - on_chat_model_stream → 携带 AIMessageChunk，提取 content 推送给前端
          - on_tool_start / on_tool_end → 可选地推送工具调用状态
        """
        config = {"configurable": {"thread_id": thread_id}}
        async for event in self.graph.astream_events(
            {"messages": [("user", message)]},
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")
            # LLM token 流
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
            # 工具调用提示（可选，发送元信息让前端显示"正在查询..."）
            elif kind == "on_tool_start":
                tool_name = event.get("name", "")
                yield f"\n\n> 正在调用工具：{tool_name}...\n\n"
