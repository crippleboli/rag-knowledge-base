import json
import copy
from typing import  TypedDict
from app.shared.runtime.logger import logger

class ImportGraphState(TypedDict):

    # 任务状态
    task_id : str

    # 文件状态判断
    is_md_read_enabled : bool
    is_pdf_read_enabled : bool

    # 地址路径内容
    local_file_path : str
    local_dir : str
    md_path : str
    pdf_path : str
    file_title : str

    # 文本和切块内容
    md_content: str
    item_name : str
    chunks : list
    embeddings_content : list


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
