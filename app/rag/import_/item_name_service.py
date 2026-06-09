from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import ITEM_NAME_CONTEXT_CHUNK_K, ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_provider
from app.shared.runtime.load_prompt import load_prompt
from app.infra.vectorstore.milvus_gateway import milvus_gateway


@step_log('validate_chunks_and_title')
def validate_chunks_and_title(state) -> tuple[list[dict],str]:
    # 1. 获取数据
    chunks = state.get('chunks')
    file_title = state.get('file_title')
    # 2. 非空校验
    if not chunks:
        logger.error(f'chunks内容为空,无法继续业务！')
        raise ValueError(f'chunks内容为空,无法继续业务！')
    if not file_title:
        file_title =  chunks[0]['file_title'] or 'default_file_title'   # 从第一个片段读取一下

    # 3. 返回
    return chunks, file_title

@step_log('build_document_context')
def build_document_context(chunks) -> str:
    """
    上下文拼接
    :param chunks:
    :return:
    """
    # 1. 截取top k
    top_chunk = chunks[:ITEM_NAME_CONTEXT_CHUNK_K]
    # 2. 拼接上下文
    context = ''
    for idx, chunk in enumerate(top_chunk,start=1):
        # 文档标题 & 章节标题
        context += f'切片:{idx} 标题:{chunk['title']} 父标题:{chunk['parent_title']}  内容:{chunk['content']} \n'
    # 3. 最大上下文限制
    final_context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]
    return final_context

@step_log('recognize_item_name')
def recognize_item_name(context:str,file_title:str) -> str:
    # 1. llm
    chat_model = llm_provider.chat()

    # 2. 加载配置好的提示词
    system_prompt_str = load_prompt('product_recognition_system')
    human_prompt_str = load_prompt(
        'item_name_recognition',
        file_title=file_title,
        context=context,
    )
    # 3. 封装提示词格式
    messages = [
         SystemMessage(content=system_prompt_str),
         HumanMessage(content=human_prompt_str),
    ]

    # 4. chain
    chains = chat_model | StrOutputParser()

    # 5. 调用
    item_name = chains.invoke(messages)
    logger.info(f'调用模型进行item_name识别完毕,item_name:{item_name}')

    # 6. 非空校验
    if not item_name:
        item_name = file_title
    return item_name


@step_log('apply_item_name')
def apply_item_name(chunks:list[dict],item_name:str):
    """
    为每个chunk添加对应的item_name
    :param chunks:
    :param item_name:
    :return:
    """
    for chunk in chunks:
        chunk['item_name'] = item_name

    logger.info(f'完成chunks的item_name数据补充:{chunks[0]["item_name"]}')


@step_log('embed_item_name')
def embed_item_name(item_name:str):
    """
        根据item_name 生成稠密/稀疏向量
    :param item_name:
    :return:
    """
    result = llm_provider.embed_documents([item_name])   # 包装成列表的单个 item_name
    return result['dense'][0],result['sparse'][0]


@step_log("prepare_item_name_collection")
def prepare_item_name_collection():
    """
    创建 milvus 数据库中 用于存储item_name的 collection (类似数据库中的表)
    :return:
    """
    # 1. 获取客户端对象
    milvus_client = milvus_gateway.client
    # 2. 判断集合是否存在
    if milvus_client.has_collection(collection_name=milvus_gateway.item_collection_name):
        logger.info(f"{milvus_gateway.item_collection_name}对应的集合存在,无需创建!")
        return

    # 3. 创建集合对应schema field (表和字段)
    schema = milvus_client.create_schema(
        auto_id=True,                   # 主键自增
        enable_dynamic_field=True,      # 动态添加字段
    )

    # 官网文档 注意版本用的2.6: https://milvus.io/docs/zh/v2.6.x/sparse_vector.md
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)         # 主键
    schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR) # 专门数据类型


    # 4. 创建集合的索引
    #  Milvus 官方推荐使用 AUTOINDE
    """    
    index_params.add_index(
        field_name="vector",
        metric_type="L2",
        index_type="AUTOINDEX",
    )
    """

    index_params = milvus_client.prepare_index_params()

    # 稠密向量添加索引
    """    
    HNSW: 多层图索引，搜索精度与速度的最优平衡（需占用较多内存）
    IVF_FLAT: 聚类分桶索引，通过缩小检索范围提升速度（内存消耗适中）
    FLAT: 全量暴力检索，精度100%但搜索速度最慢
    """
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",          # 分层可导航小世界
        metric_type="COSINE" ,      # 相似度算法: 余弦相似度 [-1, 1]
        params = {
            "M": 64,                # 图中的最大邻居数量
            "efConstruction": 100   # 建库时的搜索深度
        }
    )


    # 稀疏向量添加索引
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",           # 相似度算法: 内积
        params={"inverted_index_algo": "DAAT_MAXSCORE"}     # 剪枝优化: 倒排索引查询
    )

    # 5. 创建集合
    milvus_client.create_collection(
        collection_name=milvus_gateway.item_collection_name,    # 集合名
        schema=schema,                                          # 表
        index_params=index_params                               # 索引
    )
    logger.info(f"{milvus_gateway.item_collection_name}完成初始化")

@step_log("upsert_item_name")
def upsert_item_name(item_name: str, file_title: str, dense_vector: list[float], sparse_vector: dict[int, float]):
    """
      先删除 / 再插入
      幂等性（Idempotence）是一个数学与计算机科学概念，指某操作执行一次或多次所产生的影响（或副作用）均相同
    :param item_name:
    :param file_title:
    :param dense_vector:
    :param sparse_vector:
    :return:
    """
    milvus_client = milvus_gateway.client
    # 1. 根据file_title删除
    milvus_client.delete(
        collection_name=milvus_gateway.item_collection_name,
        filter=f"file_title == '{file_title}'"      #  SQL 中的 WHERE 子句   注意单引号
    )

    # 2. 插入新的数据即可
    result =  milvus_client.insert(
        collection_name=milvus_gateway.item_collection_name,
        data=[{
            "item_name":item_name,
            "file_title":file_title,
            "dense_vector":dense_vector,
            "sparse_vector":sparse_vector
        }]
    )

    logger.info(f"{item_name}对应的数据已经插入到{milvus_gateway.item_collection_name}对应的集合中, 返回结果:{result}")



# 主业务流程
@step_log('recognize_and_index_item_name')
def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. 同步写入主体名称索引
    """
    # 1. 进行参数校验
    chunks , file_title =  validate_chunks_and_title(state)
    # 2. 进行上下文的拼接 chunks
    # chunk content title parent_title
    context =  build_document_context(chunks)
    # 3. 进行item_name的识别了 llm
    item_name = recognize_item_name(context,file_title)
    # 4. 修改所有chunks的item_name属性
    apply_item_name(chunks,item_name)
    # 5. 对item_name进行向量化,生成稠密和稀疏向量
    dense_vector,sparse_vector = embed_item_name(item_name)
    # 6. 准备item_name对应的集合信息
    prepare_item_name_collection()
    # 7. 更新或者存储item_name到对应的集合
    upsert_item_name(item_name, file_title, dense_vector, sparse_vector)
    # 8. 更新state数据
    # item_name
    state['chunks'] = chunks
    state['item_name'] = item_name
    return state