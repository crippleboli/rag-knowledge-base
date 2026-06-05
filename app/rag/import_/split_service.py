import re
from pathlib import Path
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger,PROJECT_ROOT,step_log


def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切块核心节点（RAG 最关键步骤）
    功能：加载增强后的 Markdown 内容 → 按标题智能切块 → 优化块大小 → 备份切块结果 → 写入状态
    输出：将分块后的文本列表存入 state，供后续向量化、入库使用
    """
    return state