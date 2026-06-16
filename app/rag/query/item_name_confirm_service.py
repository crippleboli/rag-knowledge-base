from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.process.query.agent.state import QueryGraphState
from app.infra.persistence.history_repository import history_repository
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_provider
from app.shared.runtime.load_prompt import load_prompt
from app.infra.vectorstore.milvus_gateway import milvus_gateway


@step_log("get_data_and_validates")
def get_data_and_validates(state: QueryGraphState) -> tuple[str, str]:
    """
    进行参数校验: 原始问题 original_query 和 session_id
    :param state:
    :return: 校验后结果
    """

    original_query = state.get("original_query")
    session_id = state.get("session_id")

    if not original_query or not session_id:
        logger.error(f"业务核心参数original_query或者session_id为空,业务无法继续进行!")
        raise ValueError(f"业务核心参数original_query或者session_id为空,业务无法继续进行!")

    return original_query, session_id


@step_log("get_history_messages")
def get_history_messages(session_id: str, limit: int = 10) -> list[dict]:
    """
    获取历史聊天记录 (倒序 limit=10)
    只获取有效的聊天记录 item_names有数据为判断依据
    :param session_id: 筛选条件
    :param limit: 筛选数量
    :return: 有效数据集合
    """
    history_message_list = history_repository.list_recent(session_id=session_id, limit=limit)
    # 获取列表字典 其中字典的key 由mongo_history_utils中函数save_chat_message定义
    """
    [
    {
        "_id": ObjectId("6653a1b2c4d5e6f7a1234567"), # MongoDB自动生成的二进制唯一标识
        "session_id": "000015_hybrid",               # 会话唯一标识
        "role": "user",                             # 角色: user 或 assistant
        "text": "这个万用表怎么换电池？",                # 原始对话内容
        "rewritten_query": "混合万用表 更换电池的操作步骤",# 重写后的查询语句（可能为空字符串 ""）
        "item_names": ["混合万用表"],                 # 关联的商品名称列表（可能为 None 或空列表 []）
        "image_urls": None,                         # 关联的图片URL列表（可能为 None）
        "ts": 1716769400.0                          # 消息写入时的时间戳（秒级 float）
    },
    {
        ...
    }
    ]
    """
    logger.info(f"查询历史记录数量:{len(history_message_list)}")
    # 有效校验:  item_names 键存在 且  值列表大于等于 1 个
    final_message_list = [item for item in history_message_list if item.get("item_names") and len(item.get('item_names')) > 0]
    logger.info(f"校验后历史记录数量:{len(final_message_list)}")
    return final_message_list


@step_log("build_history_context_text")
def build_history_context_text(history_message_list) -> str:
    """
     历史记录完成校验之后 构建当前会话对应的上下文
     约定格式: 序号,类型: 提问 / 回答 ,内容: text/rewritten_query , 关联主体: 1,2,3 \n
    :param history_message_list:
    :return:
    """
    history_text = ""
    # item 单次聊天记录包含字段:  _id role text rewritten_query ts item_names image_urls
    for index, item in enumerate(history_message_list, start=1):
        history_text += (f"序号:{index},类型:{'提问' if item['role'] == 'user' else '回答'},"
                         f"内容:{item['rewritten_query'] if item['role'] == 'user' else item['text']},"
                         f"关联主体:{','.join(item['item_names'])}\n"
                         )
    logger.info(f"最终拼接历史记录上下文:{history_text}")
    return history_text


@step_log("call_llm_deal_data")
def call_llm_deal_data(history_text, original_query) -> dict:
    """
    调用模型进行问题重写和item_name识别
    注意: 返回json格式  使用JsonOutputParser进行处理
    :param history_text: 历史记录
    :param original_query: 原始问题
    :return: dict
    """
    # 1. 加载模型 json model
    json_llm_client = llm_provider.chat(json_mode=True)
    # 2. 构建提示词
    prompt_text = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=original_query)
    messages = [
        HumanMessage(
            content=prompt_text
        )
    ]

    # 3. 构建调用链
    chain = json_llm_client | JsonOutputParser()
    # 4. 执行获取结果
    # {  item_names : [] , rewritten_query: 重写问题}
    result_dict = chain.invoke(messages)
    # 5. 结果校验
    # 返回的json格式在提示词中定义:
    """
    请直接返回JSON格式结果，格式如下：
        {{
            "item_names": ["商品A", "商品B"],
            "rewritten_query": "关于商品A和商品B，..."
        }}
    """
    if "item_names" not in result_dict:
        result_dict['item_names'] = []
    if "rewritten_query" not in result_dict:
        result_dict['rewritten_query'] = original_query
    # 6. 返回结果
    return result_dict


@step_log("query_item_name_milvus")
def query_item_name_milvus(item_names: list[str]) -> dict[str, list[dict]]:
    """
    从向量数据库进行item_name查询和结果处理 混合查询
    :param item_names: 模型识别,但是没有通过milvus确认的数据
    :return: 返回milvus中关联的高分数据 还未根据分数进行处理
    """
    # 1. 定义前置存储容器
    milvus_result_dict = {}
    # 2. 循环处理每个item_name(模型返回)
    for item_name in item_names:
        # 3. 使用 BGE-M3 对每个item_name向量化,获取对应的稠密和稀疏向量
        embedding_result = llm_provider.embed_documents([item_name])
        dense_vector = embedding_result['dense'][0]
        sparse_vector = embedding_result['sparse'][0]
        # 4. 组装对应AnnSearchRequest对象列表: [dense_req, sparse_req]
        ann_request_list = milvus_gateway.create_requests(dense_vector, sparse_vector, limit=10)
        # 5. 进行混合数据检索,获取结果
        milvus_search_result = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.item_collection_name,
            reqs=ann_request_list,          # AnnSearchRequest对象列表
            ranker_weights=(0.4, 0.6),      # 实际权重
            norm_score=True,                # 归一化
            limit=5                         # 截取数量
        )
        """
        [
            [ -> real 
                {
                   id: 1,                       # 主键
                   distance: 0.7,               # 相似度
                   entity:{                     # 返回字段
                      item_name: name1
                   }
                },
    
                {
                   id: 2,
                   distance: 0.6,
                   entity:{
                      item_name: name2
                   }
                }
            ],
            [
            ...
            ]
        ]
        """
        # 6. 单条结果解析
        real_result = milvus_search_result[0]                   # 第一个列表
        """
        [ -> real_result 
            {
               id: 1,                       # 主键
               distance: 0.7,               # 相似度
               entity:{                     # 返回字段
                  item_name: name1
               }
            },

            {
               id: 2,
               distance: 0.6,
               entity:{
                  item_name: name2
               }
            }
        ]
        """
        if not real_result or len(real_result) == 0:            # 没有查到到数据
            logger.warning(f"模型提供的: {item_name} 没有检索到对应数据库数据,跳过本次")
            continue

        # 变形
        current_item_name_list = [{"item_name": item_dict.get('entity', {}).get('item_name'),
                                   "score": item_dict.get('distance', 0)} for item_dict in real_result]
        """
            [
                {
                    item_name: name1,
                    score: 0.7,
                },
                {
                    item_name: name2,
                    score: 0.6,
                },
            ]
        """
        milvus_result_dict[item_name] = current_item_name_list
        """
        {
            "烫金机": [
                {"item_name": "自动烫金机A型", "score": 0.85},
                {"item_name": "手动精密烫金机", "score": 0.72}
            ],
            "雕刻机": [
                {"item_name": "激光微型雕刻机", "score": 0.91},
                {"item_name": "CNC数控雕刻机", "score": 0.68}
            ]
        }
        """
    return milvus_result_dict


@step_log("select_item_names")
def select_item_names(milvus_result_dict):
    """
    确定可选和确认列表
    :param milvus_result_dict:
    :return:
    """
    # 1.定义两个列表
    confirmed_item_name_list = []
    options_item_name_list = []

    # 2. 循环处理每个item_name对应的列表
    for item_name, milvus_list in milvus_result_dict.items():
        milvus_list.sort(key=lambda x: x['score'], reverse=True)     # 默认milvus查询结果已经降序 所以只是一个保险兜底
        # 筛选列表 确认[>0.7] 可选[0.6-0.7]
        high_item_names = [item['item_name'] for item in milvus_list if item['score'] > 0.7]
        md_item_names = [item['item_name'] for item in milvus_list if 0.6 < item['score'] <= 0.7]

        # 添加确认列表
        if len(high_item_names) > 0:
            confirmed_item_name_list.append(high_item_names[0]) # 只取了高分部分的最高的一个
            continue
        # 添加可选列表
        if len(md_item_names) > 0:
            options_item_name_list.extend(md_item_names[:2])   # 可能是多个
            continue

    return {
        "confirmed_item_name_list": confirmed_item_name_list,
        "options_item_name_list": options_item_name_list
    }


@step_log("change_state_status")
def change_state_status(state, item_name_dict, rewritten_query):
    """
    修改state状态
    :param state:
    :param item_name_dict:
    :param rewritten_query:
    :return:
    """
    confirmed_item_name_list = item_name_dict.get('confirmed_item_name_list', [])
    options_item_name_list = item_name_dict.get('options_item_name_list', [])

    if confirmed_item_name_list and len(confirmed_item_name_list) > 0:
        # 有确认的
        state['item_names'] = confirmed_item_name_list
        state['rewritten_query'] = rewritten_query
        return

    if options_item_name_list and len(options_item_name_list) > 0:
        state['rewritten_query'] = rewritten_query
        answer = f"你是要询问:{','.join(options_item_name_list)} 这些内容吗? 请确认!"
        state['answer'] = answer
        return

    state['rewritten_query'] = rewritten_query
    answer = f"没有在你的提问中识别主体,请确认后再提问!"
    state['answer'] = answer


@step_log("save_history_message")
def save_history_message(state):
    """
    保存聊天记录
    :param state:
    :return:
    """
    history_repository.save_message(
        session_id=state['session_id'],
        role="user",
        text=state['original_query'],
        rewritten_query=state['rewritten_query'],
        item_names=state.get("item_names", [])
    )


@step_log("confirm_item_name")
def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    意图确认服务
    """

    # 1. 获取参数和校验(state) => original_query / session_id
    original_query, session_id = get_data_and_validates(state)

    # 2. 获取当前会话之前的10条包含item_name的历史有效信息
    history_message_list: list[dict] = get_history_messages(session_id, limit=10)

    # 3. 构建上下文,注意角色问题 user用户 -> rewritten_query重写后的问题  assistant大模型 -> text回答
    history_text = build_history_context_text(history_message_list)

    # 4. 使用模型进行item_names和问题重写
    # 参数 history_text和original_query  响应: 字典 {item_names:[],rewritten_query:''}
    result_dict = call_llm_deal_data(history_text, original_query)

    item_name_dict = {}
    # 5. 进行校验,如果没有item_names无需调用向量查询
    if len(result_dict['item_names']) > 0:
        # 6.进行item_names内部识别到模型名称的向量化查询
        # 响应: {item_name(这个是模型查询到的):[ 存储从milvus中匹配 {item_name: 名字 , score: 分数} .. 应该是5个]}
        milvus_result_dict: dict[str, list[dict]] = query_item_name_milvus(result_dict['item_names'])
        # 7. 获取确认和可选地列表  dict{确认:[0.7 + ] 可选:[ 0.6 - 0.7 ]}
        item_name_dict = select_item_names(milvus_result_dict)

    # 6.修改state状态
    change_state_status(state, item_name_dict, result_dict['rewritten_query'])

    # 7. 保存本次问题聊天记录
    save_history_message(state)

    return state