"""
配置聚合模块，负责将旧配置对象统一收口到新的基础设施出口。
"""


from app.shared.config.embedding_config import embedding_config, EmbeddingConfig
from app.shared.config.lm_config import lm_config, LLMConfig
from app.shared.config.bailian_mcp_config import mcp_config, McpConfig
from app.shared.config.milvus_config import milvus_config, MilvusConfig
from app.shared.config.mineru_config import mineru_config, MinerUConfig
from app.shared.config.minio_config import minio_config, MinIOConfig
from app.shared.config.reranker_config import reranker_config, RerankerConfig
from app.shared.config.settings_config import settings, AppSettings

from dataclasses import dataclass

@dataclass
class InfrastructureConfig:
    app: AppSettings = settings
    llm: LLMConfig = lm_config
    embedding: EmbeddingConfig = embedding_config
    reranker: RerankerConfig = reranker_config
    mcp: McpConfig = mcp_config
    milvus: MilvusConfig = milvus_config
    mineru: MinerUConfig = mineru_config
    minio: MinIOConfig = minio_config


infra_config = InfrastructureConfig()
