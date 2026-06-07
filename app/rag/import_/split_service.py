import json
import re
from pathlib import Path
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_MAX_SIZE, CHUNK_SIZE , CHUNK_OVERLAP
from app.shared.runtime.logger import logger,PROJECT_ROOT,step_log



@step_log('load_markdown_content')
def load_markdown_content(state: ImportGraphState) -> tuple[str, str,Path]:
    """
        从状态字典中安全加载 Markdown 内容和文档标题
        1. 优先从 state 中直接读取
        2. 缺失时自动从文件读取兜底
        3. 统一换行符格式，保证文本干净
    :param state:   状态字典
    :return:        元组: md_content,file_title,Path(md_path)
    """
    # ========================= 状态中读取数据 =================================
    #  md内容 文件标题 md文件路径
    md_content = state.get('md_content')
    file_title = state.get('file_title')
    md_path = state.get('md_path')

    # ========================= 处理md_content缺失场景 =========================
    if not md_content:
        logger.warning('没有从state读取到md_content内容，尝试使用md_path再次读取！')

        # 确保文件路径存在
        if md_path:
            md_content = Path(md_path).read_text(encoding='utf-8')
            state['md_content'] = md_content        # 读取后记得回填 方便后续使用

        # 再次校验 md_content 是否有内容
        if not md_content:
            raise ValueError('md_content无数据,尝试根据md_path读取后依旧无数据,流程终止！')

    # ======================= 处理file_title缺失场景 ============================
    if not file_title:
        file_title = Path(md_path).stem if md_path else 'default'
        state['file_title'] = file_title        # 回填
    # ========================= 统一格式 ======================================
    # 把所有平台的换行格式标准化成 \n
    md_content = md_content.replace('\r\n','\n').replace('\r','\n')

    return md_content,file_title,Path(md_path)    # 返回处理好的文本内容



@step_log('split_by_titles')
def split_by_titles(md_content: str,file_title:str) -> list[dict]:
    """
        按 Markdown 1-6 级标题进行【语义化文档切块】
        1. 自动识别标题，保证段落语义完整
        2. 跳过代码块内部的内容，不把 ``` 内的内容误判为标题
        3. 每个块包含：内容、当前标题、文档标题，方便后续检索
    :param md_content: Markdown 文本内容
    :param file_title: 文档名称（用于溯源）
    :return: 切块列表:  [{content, title, file_title}, ... ,... ]
    """
    reg = re.compile(r"^\s*#{1,6}\s.+") # 正则匹配标题 1-6级标题 且空格后有内容
    lines= md_content.split('\n')       # 按换行符切割 逐行处理
    chunks : list[dict] = []            # 存储最终切块结果

    current_title = None                # 当前拼接的快标题
    current_title_lines :list[str] = [] # 当前块的所有行内容
    is_code_block = False               # 标志位 记录是否在代码块内部
    chunk_size = 0                      # 切块数量


    # 逐行遍历md_content
    for raw_line in lines:
        line = raw_line.strip()         # 去除首尾 换行 缩进 空格
        if not line:                    # 空行
            logger.warning('本行为空行,跳过本次处理!')
            continue

        # ====================== 判断代码块 ===================================
        if line.startswith('```') or line.startswith('~~~'):
            is_code_block = not is_code_block   # 进入和出去都取反
            current_title_lines.append(line)
            continue

        # ====================== 识别标题与切分 ===================================
        if reg.match(line) and not is_code_block:       # 是标题 & 不是代码块
            # 结算保存上一个块
            if current_title and len(current_title_lines) > 1:  # >1:过滤只有标题行的块
                chunks.append({
                    'content':'\n'.join(current_title_lines),   # 块内容
                    'title': current_title,                     # 块标题
                    'file_title': file_title                    # 文档名  后续溯源使用
                })
                chunk_size += 1

            # 结算完记录新的
            current_title = line        # 本轮循环为检测到的新标题
            current_title_lines = [current_title]
        else:
            current_title_lines.append(line)        # 普通行接着追加

    # =================== 保存最后一个块 =================================
    if current_title and len(current_title_lines) > 1:
        chunks.append({
            'content':'\n'.join(current_title_lines), 'title': current_title, 'file_title': file_title
        })
        chunk_size += 1
    # =================== 全文无标题 =================================
    if chunk_size == 0:
        chunks.append({
            'content': md_content,  # 全部
            'title': 'default',
            'file_title': file_title
        })
    logger.info(f'完成文档语义切割,共切分为:{chunk_size}块,切块内容:{chunks}')
    return chunks




@step_log("_split_long_section")
def _split_long_section(section: dict[str, Any], max_length: int = CHUNK_MAX_SIZE) -> list[dict[str, Any]]:
    """
    内部工具函数：拆分【过长的文本块】，保证单个chunk不超过最大长度限制
        1. 检查内容长度，不长则直接返回
        2. 标题单独保留，只拆分正文内容
        3. 使用语义化拆分器，按段落、句子拆分，保证语义完整
    :param section: 待拆分的切块（包含title、content等）     [{content, title, file_title}, ... ,... ]
    :param max_length: 单个块最大字符长度
    :return: 拆分后的子块列表
    """
    # 获取正文内容
    content = section.get('content','') or ''   # 处理content为None时  替换为''

    # 1. 格式清除 title
    title = section.get('title')
    body = content
    if content.startswith(title): # 标题下的第一个带标题的块
        body = content[len(title):].lstrip()      # 切片去除开头 方便后续统一处理  去除左侧开头空白字符


    # 2. 定义块的固定前缀title 和 块的有效长度
    prefix = title + '\n'
    available_length = max_length - len(prefix)     # 固定开头title + 内容 < max_length   最大块长度需要考虑加入的标识title

    # 3. LangChain官方递归拆分器
    splitter=RecursiveCharacterTextSplitter(
        chunk_size= available_length,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    sub_sections = []

    # 4. 遍历标题下的经过spitter切分后的正文片段  分配title
    for index,chunk_text in enumerate(splitter.split_text(body),start=1):
        text = chunk_text.strip()
        # 空内容
        if not text: continue
        # 为每个片段添加所属的title
        full_text = (prefix + text).strip()
        # 构造章节内子块
        sub_sections.append({
            "title": f"{title}-{index}" if title else f"chunk-{index}",  # 子章节标题：原章节标题-序号
            "content": full_text,                                        # 完整内容
            "parent_title": title,                                       # 父标题 原章节标题（用于溯源）
            "part": index,                                               # 序号（同一章节标题下的第N部分）
            "file_title": section.get("file_title"),                     # 文档标题
        })

    logger.info(f"已经完成{title}对应块进行短切! 切后块数为:{len(sub_sections)} , 数据预览: {sub_sections}")
    # 返回拆分完成的所有子块
    return sub_sections

# 章节内子块理解:
"""
"title": "## 第二章-1"  包含章节标题 和 idx
"content": "## 第二章\nxxx正文..."   只包含章节标题  保留title 便于embed时识别顺序关系
"parent_title": "## 第二章"         只包含章节标题    
"""



@step_log("_merge_short_chunks")
def _merge_short_chunks(final_chunks:list[dict],max_length:int = CHUNK_MAX_SIZE,min_length:int=CHUNK_SIZE) -> list[dict]:
    """
       合并条件: 同一个章节标题,小于600,进行合并,合并后不能大于1000
    :param final_chunks: 合并结果
    :param max_length:  合并后不得超过
    :param min_length: 章节下子块的最小长度  小于时需要与后续合并
    :return:
    """
    # 1. 声明合并后的结果列表
    final_merge_chunks = []
    # 2. 记录第一个chunk的位置
    start_chunk = None      # 当前合并中心
    # 3. 循环子块合并处理  循环自动遍历候选合并块  合并中心块手动调整
    for next_chunk in final_chunks:
        # 首轮初始化 合并中心
        if not start_chunk:
            start_chunk = next_chunk
            continue

        # 合并条件:  合并中心够小 & 与候选合并块是同一个章节标题
        is_lt_chunk_size = len(start_chunk.get('content')) < min_length
        is_same_parent_title =start_chunk.get("parent_title") and start_chunk.get("parent_title") == next_chunk.get("parent_title") # 合并中心块非空 & 合并中心块与候选合并块章节标题相同 避免两者同为None的情况
        # 合并条件都满足
        if is_lt_chunk_size and is_same_parent_title:
            # 两者合并后只需一个章节标题 去除候选块的章节标题
            next_content_to_title = next_chunk.get('content')[len(next_chunk.get('parent_title')) + 2:]         # 切片去除 章节标题和换行符
            start_content = start_chunk.get('content')
            # 合并
            merged_content = start_content + '\n' +next_content_to_title
            # 长度校验 不能合并后又过长
            if len(merged_content) <= max_length:
                start_chunk['content'] = merged_content
                logger.info(f"父标题:{start_chunk['parent_title']}, start: {start_chunk['title']}  next: {next_chunk['title']} 完成合并")
            else:
                final_merge_chunks.append(start_chunk)  # 合并后过长时 直接弃用合并内容 只在最终结果中保存合并中心 并移动合并中心至下一个
                start_chunk = next_chunk
                continue
        else:   # 合并条件不满足:  合并中心大不需要合并 | 与候选合并块不是同一个章节标题 无法合并
            final_merge_chunks.append(start_chunk)
            start_chunk = next_chunk
    # 循环结束
    if start_chunk is not None:
        final_merge_chunks.append(start_chunk)
    return final_merge_chunks



@step_log("refine_chunks")
def refine_chunks(chunks: list[dict],max_len: int = CHUNK_MAX_SIZE,min_len: int = CHUNK_SIZE) -> list[dict]:
    """
        进行长切 / 短合 / 补全属性
    :param chunks:   由split_by_titles（按 # 标题切）函数切分后的 章节级 chunks
    :param max_len: 触发长切参数
    :param min_len: 触发短合参数
    :return: chunk
    """
    # 接收最终结果
    final_chunks = []

    # 1. 循环判断
    for chunk in chunks:
        if len(chunk['content']) > max_len:  # 拆分过长章节标题块
            final_chunks.extend(_split_long_section(chunk, max_len))    # extend: sub-chunk 平铺进 final_chunk 防止嵌套
        else:
            final_chunks.append(chunk)
    # 2. 过短合并
    final_merge_chunks = _merge_short_chunks(final_chunks)
    # 3. 优化属性存在
    for chunk in final_merge_chunks:
        if "parent_title" not in chunk:
            chunk['parent_title'] = chunk['title']
        if "part" not in chunk:
            chunk['part'] = 1
    # 4. 返回处理后结果
    return final_merge_chunks




@step_log("backup_chunks_json")
def backup_chunks_json(final_chunks:list[dict], md_path_obj:Path):
    """
        数据备份 字典 -> 文件名.json
    :param final_chunks:
    :param stem:
    :return:
    """
    # 获取文件对象
    json_path_obj = md_path_obj.parent / f"{md_path_obj.stem}.json"
    # 写出内容即可 .json -> 字符串
    json_path_obj.write_text(json.dumps(final_chunks,indent=4,ensure_ascii=False), encoding="utf-8")
    logger.info(f"数据完成备份,备份的位置:{str(json_path_obj)}")




@step_log("split_document")
def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切块核心节点
    功能：加载增强后的 Markdown 内容 → 按标题智能切块 → 优化块大小 → 备份切块结果 → 写入状态
    输出：将分块后的文本列表存入 state，供后续向量化、入库使用
    """
    # 1. 从状态中加载【增强后的Markdown内容】和【文档标题】
    md_content, file_title , md_path_obj = load_markdown_content(state)
    # 2. 按 Markdown 标题（#、##、###）进行【智能语义切块】（保持段落完整性）
    chunks = split_by_titles(md_content, file_title)
    # 3. 精细切割(涉及长切和短切处理,返回最终处理的chunks)
    final_chunks = refine_chunks(chunks)
    # 4. 备份final_chunks内容
    backup_chunks_json(final_chunks,md_path_obj)
    # 5. 修改state状态 chunks
    state['chunks'] = final_chunks
    return state