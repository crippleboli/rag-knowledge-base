from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger, step_log

@step_log("get_data_and_validate")
def get_data_and_validate(state):
    """
    获取并校验两路数据
    :param state:
    :return:
    """
    embedding_chunks = state.get("embedding_chunks",[])
    hyde_embedding_chunks = state.get("hyde_embedding_chunks",[])
    if len(embedding_chunks) == 0:
        logger.error(f"embedding_chunks查询数据为空列表,无法继续业务!")
        raise ValueError(f"embedding_chunks查询数据为空列表,无法继续业务!")
    if len(hyde_embedding_chunks) == 0:
        logger.error(f"hyde_embedding_chunks查询数据为空列表,无法继续业务!")
        raise ValueError(f"hyde_embedding_chunks查询数据为空列表,无法继续业务!")
    return embedding_chunks,hyde_embedding_chunks

@step_log("use_rrf_chunks_list")
def use_rrf_chunks_list(chunks_list:list[tuple[float,dict]], limit:int=5, k:int=60):
    """
    带有权重思维的rrf算法
    :param chunks_list: list[tuple[float,dict]]
    :param limit:   截取rrf分数最高的 limit 个
    :param k:   平滑参数 减少排名对结果的过度影响
    :return:
    """
    # 1. 定义两个容器  chunk_id : 累计分(int)   chunk_id : chunk(字典)
    score_dict:dict[str,float] = {}
    chunk_dict:dict[str,dict] = {}
    # 2. 循环每路数据和对应的权重
    for  weight, current_chunks in chunks_list:
        # 3. 循环当前路计算当前路得分
        for rank,chunk in enumerate(current_chunks,start=1):
            # 分数字典中  根据chunk自身id取出分数 不存在说明是第一次并初始化为默认值0  加上 分数计算公式: 1 / k + rank  rank = 排名
            score_dict[chunk['chunk_id']]  = score_dict.get(chunk['chunk_id'],0)+ weight *(1/(k+rank))
            # chunk_dict[chunk['chunk_id']]  = chunk  保留最后一次
            chunk_dict.setdefault(chunk['chunk_id'],chunk)  # 只保留第一次
    # 4. 处理chunk列表,并进行排序
    # chunk_id 分  chunk_id chunk score -> milvus
    chunk_list = []
    for chunk_id , score in score_dict.items():
        chunk = chunk_dict.get(chunk_id)
        chunk['score'] = score      # rrf分数 覆盖 milvus检索分数
        chunk_list.append(chunk)

    chunk_list.sort(key=lambda x : x['score'],reverse=True)
    # 5. 截取limit数量chunk列表
    rrf_chunks = chunk_list[:limit]
    # 6. 返回结果
    return rrf_chunks



@step_log("fuse_by_rrf")
def fuse_by_rrf(state: QueryGraphState) :
    """
    RRF 融合服务：
    1. 合并来自不同检索源的文档列表
    2. 应用 RRF 算法消除分数差异
    3. 给出综合排名最高的文档
    4. 回写 rrf_chunks
    """
    # 1. 获取数据和校验(向量数据库查询)
    embedding_chunks , hyde_embedding_chunks = get_data_and_validate(state)
    # 2. 封装带有权重的结构
    chunks_list = [
        # 1.0 1.0  0.5 0.5
        (1.0 ,embedding_chunks ),
        (1.0 ,hyde_embedding_chunks)
    ]
    # 3. 使用rrf算法计算和解决内容
    rrf_chunks = use_rrf_chunks_list(chunks_list,limit=5,k=60)
    # 4. 返回综合积分高的chunk列表
    state['rrf_chunks'] = rrf_chunks
    return state