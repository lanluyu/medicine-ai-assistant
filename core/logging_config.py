# coding:utf-8
"""
统一日志配置 — 药品知识库 AI 助手

设计要点：
  1. 入口脚本（app_fastapi / build_index）启动时调用一次 `setup_logging()`
  2. 业务模块只用 `logging.getLogger(__name__)`，禁止再调 `basicConfig`
  3. 日志双路输出：滚动文件 `logs/<app_name>-YYYY-MM-DD.log` + stderr
  4. 重复调用幂等（uvicorn --reload 会重复 import 入口模块）
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Iterable, Optional

# ─── 默认参数 ──────────────────────────────────────────────────────────
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_BACKUP_DAYS = 14  # 日志保留 14 天

# 标记入口在本进程内是否已配置过，避免 reload 时重复挂 handler
_CONFIGURED = False


def setup_logging(
    app_name: str = "medicine",
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    backup_days: int = _DEFAULT_BACKUP_DAYS,
    quiet_loggers: Optional[Iterable[str]] = None,
) -> Path:
    """配置全局 root logger。

    Args:
        app_name: 日志文件名前缀（如 "medicine-api" / "build-index"）
        log_dir: 日志目录，相对路径基于当前工作目录
        level: 日志级别（默认 INFO）
        backup_days: 滚动文件保留天数（默认 14 天）
        quiet_loggers: 需要降级到 WARNING 的第三方 logger 名（如 "httpx" / "urllib3"）

    Returns:
        实际写入的日志目录绝对路径
    """
    global _CONFIGURED

    log_dir_path = Path(log_dir).resolve()
    log_dir_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # 幂等：已配置则只更新级别，不重复挂 handler
    if _CONFIGURED:
        return log_dir_path

    # 清空 root 上可能由其它库（如 uvicorn / basicConfig）挂的 handler
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)

    file_handler = TimedRotatingFileHandler(
        filename=log_dir_path / f"{app_name}.log",
        when="midnight",
        interval=1,
        backupCount=backup_days,
        encoding="utf-8",
        utc=False,
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    # 降噪第三方库
    for name in quiet_loggers or ("httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
    root.info("日志系统就绪: app=%s, dir=%s, level=%s", app_name, log_dir_path, logging.getLevelName(level))
    return log_dir_path
