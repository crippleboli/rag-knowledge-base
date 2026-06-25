from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log, logger
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS, RERANK_SUMMARY_CHAR_RATIO, RERANK_MIN_SUMMARY_CHARS, \
    RERANK_MAX_TOPK, RERANK_MIN_TOPK, RERANK_GAP_RATIO, RERANK_GAP_ABS

@step_log("get_rewritten_query_and_validate")
def get_rewritten_query_and_validate(state):
    """
    获取和校验核心参数
    :param state:
    :return:
    """
    # 获取参数
    rrf_chunks = state.get("rrf_chunks",[])
    web_search_docs = state.get("web_search_docs",[])
    rewritten_query = state.get("rewritten_query")

    # 1. 校验本地 RRF 融合结果
    if len(rrf_chunks) == 0:
        logger.error("关键参数错误: rrf_chunks (本地融合结果) 为空，业务无法继续进行")
        raise ValueError("关键参数错误: rrf_chunks 为空")
    # 2. 校验外网搜索结果
    if len(web_search_docs) == 0:
        logger.error("关键参数错误: web_search_docs (外网搜索结果) 为空，业务无法继续进行")
        raise ValueError("关键参数错误: web_search_docs 为空")
    # 3. 校验重写问题
    if not rewritten_query:
        logger.error("关键参数错误: rewritten_query (重写问题) 为空，业务无法继续进行")
        raise ValueError("关键参数错误: rewritten_query 为空")

    return rrf_chunks,web_search_docs,rewritten_query

@step_log("deal_chunk_list")
def deal_chunk_list(rrf_chunks, web_search_docs):
    """
    整理对齐数据结构 方便后续使用
        rrf_chunks ->  chunk_id title parent_title part file_title content type url item_name
        web_search_docs -> snippet title  url
    融合目标:
        title
        text
        type        数据来源:        milvus / web
        url         相关资源链接:     milvus - none / web  url (资源的网页 | 资源关联的图)
        score       reranker分数:   milvus 排名的分 / web 没有分 统一初始化为 0
    :param rrf_chunks:
    :param web_search_docs:
    :return:
    """
    final_chunk_list = []
    # 1. 循环rrf_chunks
    for chunk in rrf_chunks:
        final_chunk_list.append({
            "title": chunk.get("title"),
            "text": chunk.get("content"),
            "type": "milvus",           # milvus 检索获取
            "url": None,
            "score": 0.0
        })
    # 2. 循环web_search_docs
    for doc in web_search_docs:
        final_chunk_list.append({
            "title": doc.get("title"),
            "text": doc.get("snippet"), # 网页正文摘要
            "type": "web",              # web 检索获取
            "url": doc.get("url"),
            "score": 0.0
        })

    # 3. 返回结果
    return final_chunk_list

@step_log("ranker_answer_llm_deal")
def ranker_answer_llm_deal(answer: str,limit: int, question:str) -> str:
    """
    调用llm对文本进行压缩
    :param answer:
    :param limit:
    :param question:
    :return:
    """
    # 1. 加载模型客户端
    chat_client = llm_provider.chat()
    # 2. 加载提示词  只压缩回答部分
    prompt_text =  load_prompt("rerank_text_refine", question=question,answer=answer,limit=limit)
    # 3. 构建messages
    messages = [
        HumanMessage(
            content=prompt_text
        )
    ]
    # 4.构建调用链
    chain = chat_client | StrOutputParser()
    # 5. 执行
    refine_answer = chain.invoke(messages)
    return refine_answer

@step_log("deal_question_answer_pair_list")
def deal_question_answer_pair_list(rewritten_query, final_chunk_list) -> list[list[str]]:
    """
    生成问题和答案对列表
    超长,调用 ranker_answer_llm_deal 函数进行文本压缩
    :param rewritten_query:
    :param final_chunk_list:
    :return:    [[问题,答案],[]]
    """
    question_answer_pair_list = []
    # 1. 检查问题的长度 rewritten_query
    reranker_model = llm_provider.reranker_model()
    tokenizer =  reranker_model.tokenizer
    # add_special_tokens = False 只计算文本对应的token数量 不计算特殊字符
    # 返回模型字典中对应的数字编号 e.g.[1012, 4567, 2891, 934]
    question_token_ids_list =  tokenizer.encode(rewritten_query, add_special_tokens=False)
    # 列表长度 = token数量
    question_token_len = len(question_token_ids_list)
    # 2. 循环答案final_chunk_list
    for chunk in final_chunk_list:
        current_answer = chunk.get("text","")
        # 3. 检查答案的长度  操作和问题处理方式相同
        current_answer_token_len = len(tokenizer.encode(current_answer, add_special_tokens=False))
        # 4. 超长就进行模型压缩
        if current_answer_token_len + question_token_len + 4 > RERANK_MAX_INPUT_TOKENS: # 4为特殊字符数量 取决于模型种类
            # 计算回答部分字符串长度
            limit = max(
                # 防止问题过长 导致回答部分可用token/字数 过少
                RERANK_MIN_SUMMARY_CHARS,
                # 除以 token 预算换算成中文精炼字数时使用的经验系数 将token数量转为字符数量
                int((RERANK_MAX_INPUT_TOKENS-question_token_len-4) / RERANK_SUMMARY_CHAR_RATIO)
            )
            # 调用模型对回答进行压缩
            current_answer =  ranker_answer_llm_deal(answer=current_answer,limit=limit,question=rewritten_query)
        # 5. 添加到列表中即可
        question_answer_pair_list.append([rewritten_query,current_answer])
    # 6. 返回结果
    return question_answer_pair_list

@step_log("reranker_score_pair_list")
def reranker_score_pair_list(question_answer_pair_list):
    """
    调用reranker模型进行打分
    :param question_answer_pair_list:[[问题, 答案1], [问题, 答案2], [问题, 答案3]...]
    :return:    numpy.ndarray [分数1,分数2,...]
    """
    reranker_model = llm_provider.reranker_model()
    # 归一化 将分值拉倒 0-1之间 方便进行后续算法统计
    score_list =  reranker_model.compute_score(question_answer_pair_list,normalize=True)
    logger.info(f"完成对数据:{question_answer_pair_list}的打分,分数为:{score_list}")
    return score_list

@step_log("sort_final_chunk_list")
def sort_final_chunk_list(final_chunk_list, scores_list):
    """
    分数列表融合进入final_chunk_list 并进行排序
    :param final_chunk_list:
    :param scores_list:
    :return:
    """
    # 数据融合
    for chunk , score  in zip(final_chunk_list,scores_list):
        chunk['score'] = score

    logger.info(f"没排序前的顺序:{final_chunk_list}")
    logger.info("*"*60)
    # 按分数排序
    final_chunk_list.sort(key=lambda x:x['score'],reverse=True)
    logger.info(f"排序后的顺序:{final_chunk_list}")
    return final_chunk_list

@step_log("dynamic_cut_chunk_list")
def dynamic_cut_chunk_list(final_chunk_list) -> list[dict]:
    """
    动态截取数据
    :param final_chunk_list:
    :return:
    """
    max_number = RERANK_MAX_TOPK  # 截取数量的上限
    min_number = RERANK_MIN_TOPK  # 截取数量的下限
    gap_abs = RERANK_GAP_ABS      # 绝对跌幅阈值
    gap_ratio = RERANK_GAP_RATIO  # 相对前者跌幅阈值

    # max_number在配置的RERANK_MAX_TOPK 和 实际可供选择数量中取最大
    # 兼容召回数量过多和过少的情况
    max_number = min(max_number,len(final_chunk_list))

    top_k = max_number            #  top_k 初始化为最大截取数量

    if max_number > min_number:    # 有可能min > max
        # min_number-1: 数组下标和实际位序差一
        # max_number-1: 每轮循环取两个 不进行最后一轮
        for index in range(min_number-1,max_number-1):
            score_1 = final_chunk_list[index].get('score',0.0)
            score_2 = final_chunk_list[index+1].get('score',0.0)
            abs_score = score_1 - score_2
            ratio_score = abs_score / (score_1 + 1e-7)
            # 断崖判断
            if abs_score > gap_abs or ratio_score > gap_ratio:  # 绝对跌幅 or 相对前者跌幅
                top_k = index + 1
                break
    logger.info(f"已经完成断崖数据截取,进入数量:{len(final_chunk_list)},截取数据:{top_k}")
    # 截取数据
    return final_chunk_list[:top_k]

@step_log("rerank_documents")
def rerank_documents(state: QueryGraphState) -> list[dict]:
    """
    重排节点主入口
    流程：校验输入 → 合并本地+网页 → 模型打分排序 → 动态截断
    输出最终高质量候选文档列表
    """
    # 1. 获取数据并且校验
    rrf_chunks,web_search_docs,rewritten_query = get_rewritten_query_and_validate(state)

    # 2. 多路数据融合格式统一
    final_chunk_list = deal_chunk_list(rrf_chunks,web_search_docs)

    # 3. 生成问题和答案列表
    question_answer_pair_list = deal_question_answer_pair_list(rewritten_query,final_chunk_list)

    # 4. reranker模型进行打分
    scores_list = reranker_score_pair_list(question_answer_pair_list)

    # 5. 原始数据进行赋分和排序
    final_chunk_list = sort_final_chunk_list(final_chunk_list,scores_list)

    # 6. 动态截取数据 topk
    final_chunk_list = dynamic_cut_chunk_list(final_chunk_list)

    state['reranked_docs'] = final_chunk_list
    return state