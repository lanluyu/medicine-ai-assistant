# coding:utf-8
"""
FastAPI 服务入口 — 药品知识库 AI 助手

端点：
  GET  /              → 静态首页（static/index.html）
  GET  /health        → 健康检查
  POST /api/chat      → SSE 流式对话
  POST /api/session   → 创建新会话，返回 thread_id
  GET  /api/session/{thread_id}/history → 获取会话历史

启动方式：
  conda activate lang
  cd D:/Awork/medicine/medicine_llamaindex
  uvicorn app_fastapi:app --host 127.0.0.1 --port 8000 --reload
  
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.config import FASTAPI_HOST, FASTAPI_PORT, STATIC_DIR
from modules.agents import MedicineAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── 全局 Agent 单例 ──────────────────────────────────────────────────
_agent: Optional[MedicineAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预加载模型和索引（避免首次请求超时）"""
    global _agent
    logger.info("=" * 60)
    logger.info("正在预加载 MedicineAgent（Embedding 模型 + 向量索引）...")
    try:
        _agent = MedicineAgent()
        logger.info("MedicineAgent 预加载完成")
    except Exception as e:
        logger.error(f"MedicineAgent 预加载失败: {e}")
        # 不阻止服务启动，请求时会返回 503
    logger.info("=" * 60)
    yield
    logger.info("服务关闭")


# ─── FastAPI 应用 ─────────────────────────────────────────────────────
app = FastAPI(
    title="药品知识库 AI 助手",
    description="基于 llama-index + LangChain + Qwen 的药品说明书 RAG 系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件（CSS / JS / 图片等）
import os
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── 请求/响应模型 ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    thread_id: str = ""          # 空字符串 = 新会话


class SessionResponse(BaseModel):
    thread_id: str


# ─── 端点 ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "ok",
        "agent_loaded": _agent is not None,
        "index_loaded": _agent.graph is not None if _agent else False,
    }


@app.post("/api/session", response_model=SessionResponse)
async def create_session():
    """创建新会话，返回唯一 thread_id"""
    if _agent is None:
        raise HTTPException(status_code=503, detail="服务正在初始化，请稍后重试")
    return SessionResponse(thread_id=_agent.new_session())


@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """
    SSE 流式对话端点。

    - Content-Type: text/event-stream
    - 每个 data: 行携带一个 token 片段（JSON 编码）
    - data: [DONE] 表示流结束
    - data: {"error": "..."} 表示发生错误

    前端用法（fetch + ReadableStream）：
      const resp = await fetch('/api/chat', {method:'POST', body: JSON.stringify({...})})
      const reader = resp.body.getReader()
      // 循环读取 chunk 并追加到消息框
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="服务正在初始化，请稍后重试")

    message   = request.message.strip()
    thread_id = request.thread_id.strip() or _agent.new_session()

    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    logger.info(f"[SSE] thread={thread_id} | message={message[:60]!r}")

    async def event_generator():
        # 先推送 thread_id，让前端知道本次使用的会话 ID
        yield f"data: {json.dumps({'thread_id': thread_id, 'type': 'meta'}, ensure_ascii=False)}\n\n"

        try:
            async for token in _agent.astream_tokens(message, thread_id):
                payload = json.dumps({"token": token, "type": "token"}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                # 每 token 发送后让出控制权，确保实时推送
                await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"[SSE] 流式生成错误: {e}", exc_info=True)
            error_payload = json.dumps({"error": str(e), "type": "error"}, ensure_ascii=False)
            yield f"data: {error_payload}\n\n"

        # 结束标志
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # 禁止 Nginx 缓冲，确保实时推送
        },
    )


@app.get("/")
async def serve_index():
    """返回前端首页"""
    from fastapi.responses import FileResponse
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse({"message": "前端文件未找到，请检查 static/ 目录"}, status_code=404)
    return FileResponse(index_path)


# ─── 开发模式直接运行 ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_fastapi:app",
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        reload=False,   # 生产环境关闭 reload
        log_level="info",
    )
