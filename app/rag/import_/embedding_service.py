from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_provider


@step_log('require_chunks')
def require_chunks(state) -> list[dict]:
    """
    校验 state中的chunks
    :param state:
    :return:
    """
    chunks = state.get('chunks')
    if not chunks or len(chunks) == 0:  # 非空校验
        logger.error(f'chunks数量为空,无法继续业务')
        raise ValueError(f'chunks数量为空,无法继续业务')
    return  chunks

@step_log('embed_chunks')
def embed_chunks(chunks:list[dict],*,step: int = EMBEDDING_BATCH_SIZE) -> list[dict]:
    """
    批量将chunks转为向量
    :param chunks:
    :param step:    每step批量转为向量
    :return:
    """
    final_chunks = []
    # 1. 取各部分起始chunk
    for index in range(0,len(chunks),step):
        # 2. 切分本批次chunk列表
        step_chunks = chunks[index:index+step]
        # 3，组装列表
        step_vector_list = []
        for current_chunk in step_chunks:
            step_vector_list.append(
                f"主体名:{current_chunk['item_name']},内容:{current_chunk['content']}"
            )
        # 4. 调用批量embed函数
        result = llm_provider.embed_documents(step_vector_list)
        # 结果格式:
        """
          result = {
              "dense":[ [],[],[],[],[] ],
              "sparse":[ {},{},{},{},{}]
          }
        """
        # 5. 结果装回final_chunks
        for idx,chunk in enumerate(step_chunks):    # 老师内外层都用了index变量名
            chunk_new = chunk.copy()
            chunk_new['dense_vector'] = result['dense'][idx]  # 注意顺序
            chunk_new['sparse_vector'] = result['sparse'][idx]

            final_chunks.append(chunk_new)

    logger.info(f'完成chunks向量化:原始数据{chunks[0]},embed后:{final_chunks[0]}')
    return final_chunks


# 主业务流程
@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """
    chunks = require_chunks(state)
    # 带有向量
    final_chunks = embed_chunks(chunks)
    # 修改state
    state['chunks'] = final_chunks
    return state
