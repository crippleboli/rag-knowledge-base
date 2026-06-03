from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger,step_log
from pathlib import Path

step_log('resolve_input_file')
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    """
    入口识别服务：
    1. 校验 local_file_path
    2. 识别文件类型（PDF / Markdown）
    3. 回写 is_pdf_read_enabled / is_md_read_enabled
    4. 回写 pdf_path / md_path / file_title
    """
    # 1. 使用get方法获取本地文件路径
    local_file_path = state.get("local_file_path")
    # 2. 校验local_file_path是否为空 否则直接抛出异常并结束
    if not local_file_path:
        logger.error(f'节点:resolve_input_file,文件路径为空,直接终止当前导入流程')
        raise ValueError('传入的local_file_path的参数为空,没有文件无法继续业务！')
    # 3.识别文件类型并设置对应的状态和路由开关
    if local_file_path.endswith('.md'): # md类型
        state["is_pdf_read_enabled"] = False
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path
    elif local_file_path.endswith('.pdf'):
        state["is_pdf_read_enabled"] = True
        state["is_md_read_enabled"] = False
        state["pdf_path"] = local_file_path
    else:
        # 除了md/pdf 其他类型终止
        logger.warning(f"传入的文件：{local_file_path}类型无法处理，当前项目只支持 md/ pdf类型，直接跳转到END节点！")
        return state
    # 4. 提取文件不带后缀的标题
    state['file_title'] = Path(local_file_path).stem
    return state