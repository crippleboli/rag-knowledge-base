import datetime
from pymilvus import DataType
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import step_log,logger


@step_log("require_chunks")
def require_chunks(state: dict) -> list[dict]:
    """
    获取非空的chunks
    :param state: LangGraph 流程状态字典
    :return: 已通过校验的切块列表
    """
    chunks = state.get('chunks',[])
    if not chunks:          # 非空校验
        logger.error(f'chunks为空, 无法继续')
        raise ValueError(f'chunks为空, 无法继续')

    return chunks

@step_log("prepare_chunks_collection")
def prepare_chunks_collection() -> None:
    """
    准备 Milvus 切片集合
    :return: 无
    """

    # 1. milvus 客户端
    milvus_client = milvus_gateway.client

    # 2. 集合名称
    collection_name = milvus_gateway.chunk_collection_name

    # 3. 已存在 则无需创建
    if milvus_client.has_collection(collection_name = collection_name):
        logger.info(f"{collection_name}对应的集合以及存在 无需创建直接使用")
        return

    # 4. 不存在继续后续创建 schema
    # 创建 schema，启用自动 ID 和动态字段
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

    # 添加主键字段：chunk_id，INT64 类型，自增
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # 添加文件标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)

    # 添加主体名称字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)

    # 添加切片标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)

    # 添加父标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)

    # 添加切片序号字段：INT8 类型
    schema.add_field(field_name="part", datatype=DataType.INT8)

    # 添加内容字段：VARCHAR 类型，最大长度 65535（支持长文本）
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

    # 添加稠密向量字段：FLOAT_VECTOR 类型，维度 1024
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

    # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR 类型
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 5. 创建 index
    index_params = milvus_client.prepare_index_params()     # 索引参数
    # 稠密向量创建索引
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        metric_type="COSINE",
        params={
            "M": 64,        # 近邻数量
            "efConstruction": 100   # 检索数量
        }
    )

    # 稀疏向量创建索引
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},    # 剪枝策略
    )

    # 6. 创建
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"{collection_name}完成对应的集合创建!")



@step_log("remove_old_chunks")
def remove_old_chunks(file_title: str) -> None:
    """
    根据文件名称删除已存在的切片记录 同一主体重复导入时覆盖旧数据
    :param file_title: 文件名,文件名唯一,主体可能修改
    :return:
    """
    # 获取 Milvus 客户端并执行删除操作
    milvus_gateway.client.delete(
        collection_name=milvus_gateway.chunk_collection_name,
        filter=f"file_title=='{file_title}'",
    )


@step_log("insert_chunks")
def insert_chunks(chunks: list[dict]) -> None:
    """
    批量插入切片数据到 Milvus 集合
    :param chunks: 带向量字段的切片列表
    :return:
    """
    # 执行批量插入操作
    result = milvus_gateway.client.insert(
        collection_name=milvus_gateway.chunk_collection_name,
        data=chunks,
    )
    # 记录插入结果
    logger.info(f"插入数据成功! 总条数:{result.get('insert_count', 0)}")
    logger.info(f"插入数据主键回显:{result.get('ids', [])}")








@step_log("index_chunks")
def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """
    入库服务：
    1. 准备集合 schema 和索引
    2. 根据 item_name 删除旧数据
    3. 批量插入新的 chunks
    4. 回写 chunk_id 等入库结果
    """

    # 1. 获取并校验 chunks
    chunks = require_chunks(state)
    # 2. 准备collection集合
    prepare_chunks_collection()
    # 3. 插入数据
    remove_old_chunks(state['file_title'])
    insert_chunks(chunks)

    logger.info(f"{datetime.datetime.now().strftime('%Y%m%d')}完成{state['task_id']}导入文件数据入库操作")
    return state