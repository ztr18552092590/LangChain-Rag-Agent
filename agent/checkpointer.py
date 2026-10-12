import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

_saver: SqliteSaver | None = None


def get_checkpointer() -> SqliteSaver:
    """进程内单例，连接 data/checkpoints.sqlite，与 FAQ 的 chroma_db 分开。"""
    global _saver
    if _saver is None:
        db_path = get_abs_path("data/checkpoints.sqlite")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        _saver = SqliteSaver(conn)
        _saver.setup()
        logger.info(f"[checkpointer]已启用 SQLite 会话记忆: {db_path}")
    return _saver
