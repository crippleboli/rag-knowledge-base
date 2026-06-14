import re

from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import add_done_task,add_running_task,push_to_session
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger
from app.infra.persistence.history_repository import history_repository
from app.infra.llm.providers import llm_provider
import time
import sys


def check_state_has_answer(state):
    """
    检测是否有answer!
      有返回对应的字符串
    :param state:
    :return:
    """
    answer = state.get("answer")
    if not answer:
        logger.info(f"没有answer,证明有明确的item_names正常返回结果!!")
        return False
    # 我们就的给前端返回数据
    # is_stream = True -> 打字机模型 -> state (final)
    # is_stream = False -> answer -> state
    is_stream = state.get("is_stream", False)

    if is_stream:
        # 流式返回
        # answer "A B C D E F G "
        for ch in answer:
            push_to_session(
                state.get("session_id") , SSEEvent.DELTA , {"delta":ch}
            )
            #time.sleep(0.06)
    return True


def get_data_and_validates(state):
    """
      获取数据并且校验
    :param state:
    :return:
    """
    reranked_docs = state.get("reranked_docs",[])
    item_names = state.get("item_names",[])
    rewritten_query = state.get("rewritten_query")

    if len(reranked_docs) == 0 or len(item_names) == 0 or not rewritten_query:
        logger.info(f"没有reranker_docs,item_names,rewritten_query,请检查参数!!")
        raise ValueError("没有reranker_docs,item_names,rewritten_query,请检查参数!!")

    history = history_repository.list_recent(state.get("session_id"),limit=10)

    return reranked_docs,history,item_names,rewritten_query


def load_prompt_text(reranker_docs, history, item_names, rewritten_query) -> str:
    """
    加载提示词文件! 拼接提示词!
    :param reranker_docs: -> context
    :param history: -> 聊天记录
    :param item_names: ->  关联主体
    :param rewritten_query: -> 问题
    :return:
    """
    # 拼接context  reranker_docs [{title,text,type,url[取图片],score}]
    # 标题: title , 来源: 向量库 / 网络搜索 , reranker模型评分: score \n
    # 内容: xxx
    # \n\n
    context =  ""
    for doc in reranker_docs:
        context += (f"标题: {doc['title']} 来源: {'网络搜索' if doc['type'] == 'web' else '向量库'} , "
                    f"reranker模型评分: {doc['score']} \n"
                    f"内容: {doc['text']}\n\n")
    # history 拼接
    history_text = ""
    final_message_list = [item for item in history if
                          item.get("item_names") and len(item.get('item_names')) > 0]
    if final_message_list and len(final_message_list) > 0:
        # item -> 聊天记录 _id role text rewritten_query ts item_names image_urls
        for index, item in enumerate(final_message_list, start=1):
            history_text += (f"序号:{index},类型:{'提问' if item['role'] == 'user' else '回答'},"
                             f"内容:{item['rewritten_query'] if item['role'] == 'user' else item['text']},"
                             f"关联主体:{','.join(item['item_names'])}\n")
    else:
        history_text = "没有对话记录!"

    # item_names关联
    item_names_text = ",".join(item_names)

    # 加载提示词模版
    prompt_text = load_prompt("answer_out",context = context,history=history_text,
                              item_names=item_names_text,question = rewritten_query)

    return prompt_text


def call_llm_generate(answer_prompt_text, state):
    """
    调用模型生成答案 文本答案
    :param answer_prompt_text:
    :param state:
    :return:
    """
    final_answer = ""
    # 1. 获取模型对象
    llm_client = llm_provider.chat()
    # 2. 判断是否是流式调用
    is_stream = state.get("is_stream", False)
    if is_stream:
        # 一段一段文本返回
        stream = llm_client.stream(answer_prompt_text)
        for chunk in stream:
            # 当前段
            current_content = chunk.content
            push_to_session(
                state.get("session_id") , SSEEvent.DELTA , {"delta":current_content}
            )
            final_answer += current_content
    else:
        response = llm_client.invoke(answer_prompt_text)
        final_answer = response.content

    state['answer'] = final_answer


def extract_image_urls(reranker_docs, state):
    """
     提取图片 url text 装到列表! 放到state
    :param reranker_docs:
    :param state:
    :return:
    """
    # 1.定义一个正则
    # 2.定义存储数据的列表
    image_urls: list[str] = []
    # 匹配 markdown 图片正则
    reg = re.compile(r"\!\[.*?\]\((.*?)\)")
    # 3.循环 -> url / text
    for doc in reranker_docs:
        url = doc.get("url","")
        text = doc.get("text","")
        # 提取url
        if url and url.endswith((".jpg",".png",".gif",".jpeg",".svg")):
            image_urls.append(url)
        # 提取text
        for image_url in reg.findall(text):
            if image_url not in image_urls:
                image_urls.append(image_url)
    # 4.给state赋值
    state['image_urls'] = image_urls
    return state


def save_history_message(state):
    history_repository.save_message(
        session_id=state.get("session_id"),
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names",[]),
        image_urls=state.get("image_urls",[])
    )


def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """
    答案生成服务：
    1. 检查前置答案（如有追问或拒绝回答，直接输出）
    2. 构建 Prompt（用户问题 + 历史对话 + TopK 文档）
    3. 调用 LLM 生成最终答案（支持流式推送）
    4. 从引用文档中提取图片 URL
    5. 写入 MongoDB 历史记录
    6. 回写 answer 和 image_urls
    """
    # 1. 判断是否有answer内容并且返回对应的状态
    has_answer = check_state_has_answer(state)
    # 2. 如果没有结果,才调用模型进行答案生成
    if not has_answer:
        # 3. 没有结果,获取并且校验参数
        reranker_docs,history,item_names,rewritten_query = get_data_and_validates(state)
        # 4. 拼接提示词的上下文,加载外部的提示词文件
        answer_prompt_text = load_prompt_text(reranker_docs,history,item_names,rewritten_query)

        # 5. 调用模型生成(文本)答案
        call_llm_generate(answer_prompt_text, state)

        # 6. 提取图片列表 -> state[image_urls] = []
        extract_image_urls(reranker_docs,state)

    save_history_message(state)
    return state