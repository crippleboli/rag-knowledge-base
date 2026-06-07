import json
import copy
from typing import  TypedDict
from app.shared.runtime.logger import logger

class ImportGraphState(TypedDict):

    # 任务状态
    task_id : str               # 每次调用流程的标识

    # 文件状态判断
    is_md_read_enabled : bool
    is_pdf_read_enabled : bool

    # 地址路径内容
    local_file_path : str       # 源文件地址 存储要解析的文件地址:pdf/md
    local_dir : str             # 存储 pdf->md 生成的md文件
    md_path : str               # md 输入文件路径 专门存储md地址
    pdf_path : str              # pdf 输入文件路径 专门存储pdf地址
    file_title : str            # 存储文件名 无后缀  stem

    # 文本和切块内容
    md_content: str             # Markdown 文本内容 用于切片
    item_name : str             # 一个文档对应的主体
    chunks : list               # 存储切块内容
    embeddings_content : list   # 存储带有向量的切块内容


# 模板
default_state:ImportGraphState = {
    'task_id': '',
    'is_md_read_enabled': False,
    'is_pdf_read_enabled': False,
    'local_file_path': '',
    'local_dir': '',
    'md_path': '',
    'pdf_path': '',
    'file_title': '',
    'md_content': '',
    'item_name': '',
    'chunks': [],
    'embeddings_content': [],
}

def create_default_state(**overriders) -> ImportGraphState:
    """创建初始状态，并支持传入特定参数进行覆盖"""
    copy_state = copy.deepcopy(default_state)
    copy_state.update(overriders)
    return copy_state

def get_default_state() -> ImportGraphState:
    """获取纯净的默认初始状态副本"""
    return copy.deepcopy(default_state)


if __name__ == '__main__':
    state = create_default_state(task_id="task_007")
    logger.info(f"测试复制方法: \n {json.dumps(state, ensure_ascii=False, indent=4)}")

    state1 = get_default_state()
    logger.info(f"测试复制方法: \n {json.dumps(state1, ensure_ascii=False, indent=4)}")
