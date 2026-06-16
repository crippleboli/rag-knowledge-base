import json
from agents.mcp import MCPServerStreamableHttp
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger, step_log
from app.infra.config.providers import infra_config
import asyncio

@step_log("get_rewritten_query_and_validate")
def get_rewritten_query_and_validate(state) -> str:

    # 1.获取数据
    rewritten_query = state.get("rewritten_query")
    # 2.校验
    if not rewritten_query:
       logger.error(f"rewritten_query没有内容,业务无法继续进行!")
       raise ValueError("rewritten_query没有内容,业务无法继续进行!")
    return rewritten_query

@step_log("web_search_docs")
async def web_search_docs(rewritten_query:str):
    # -> CallToolResult
    """
    网络调用
    :param rewritten_query:
    :return:
    """
    # 1. 初始化mcp_server
    mcp_server = MCPServerStreamableHttp(
        name="web_search_mcp",      # 名称 一般没有强制要求
        client_session_timeout_seconds=300, # 客户端会话的最大存活时间
        params={
            "url" : infra_config.mcp.mcp_base_url,  # mcp服务器地址
            "headers" : {"Authorization": f"Bearer {infra_config.mcp.api_key}"},    # api key
            "timeout" : 300         # HTTP 请求（单次连接握手、单次发送数据）的最长等待时间
        },
        cache_tools_list = True,    # 是否缓存工具
        max_retry_attempts = 3,     # 重试次数
    )
    try:
        # 2. 创建链接
        await mcp_server.connect()
        # 3. 调用网络工具
        tool_list = await mcp_server.list_tools()
        logger.info(f"本次链接服务对应的工具列表:{tool_list}")
        mcp_result =  await mcp_server.call_tool(tool_name="bailian_web_search",arguments={"query":rewritten_query,"count":5})
        return mcp_result
    except Exception as e:
        logger.exception(f"调用工具出现问题,本次参数:{rewritten_query},错误原因:{str(e)}")
    finally:
        # 4. 断开链接
        await mcp_server.cleanup()


@step_log("search_by_web")
def search_by_web(state: QueryGraphState) -> QueryGraphState:
    """
    网络搜索服务：
    1. 通过 MCP 协议异步调用百炼联网搜索接口
    2. 将用户的查询转化为实时的、结构化的网络搜索结果
    3. 包含标题、链接和摘要
    4. 回写 web_search_docs
    """
    # 1. 获取和校验参数
    rewritten_query = get_rewritten_query_and_validate(state)
    # 2. 调用业务的网络搜索工具
    mcp_result = asyncio.run(web_search_docs(rewritten_query))
    logger.info(f"查询到的结果: {mcp_result}")
    # 3. 获取结果  参考官网或者笔记中的实际返回结构:
    """
        {
          "isError": false,
          "content": [
            {
              "type": "text",
              "text": "{\"pages\":[{\"title\":\"深圳自助餐美食推荐,万德诺富特酒店性价比直接拉满!\",\"url\":\"https://cj.sina.com.cn/articles/view/1215361752/4870f2d800103gelw\",\"snippet\":\"深圳自助餐美食推荐...给生活加点糖\",\"hostname\":\"新浪网\",\"hostlogo\":\"https://mbs1.bdstatic.com/searchbox/mappconsole/image/20220307/88eb511c-5c51-448a-a9b5-df6b24cda8c7.png\"},{\"title\":\"深圳周末五星自助餐推荐,南山希尔顿逸林酒店自助餐\",\"url\":\"https://cj.sina.com.cn/articles/view/1215361752/4870f2d800103gqci\",\"snippet\":\"深圳周末五星自助餐推荐...老餮必食 4+\",\"hostname\":\"新浪网\",\"hostlogo\":\"https://mbs1.bdstatic.com/searchbox/mappconsole/image/20220307/88eb511c-5c51-448a-a9b5-df6b24cda8c7.png\"}],\"request_id\":\"f35219a8-de6f-43c9-bf66-c580229577d4\",\"tools\":[],\"status\":0}"
            }
          ]
        }
    """
    search_text =  mcp_result.content[0].text
    # 使用json.loads() 处理 字符串转为 字典 获取 pages部分:
    """
    {
        "pages": [
            {"title": "深圳自助餐美食推荐...", "url": "...", "snippet": "..."},
            {"title": "深圳周末五星自助餐...", "url": "...", "snippet": "..."}
        ],
        "request_id": "f35219a8-de6f-43c9-bf66-c580229577d4",
        "status": 0
    }
    """
    web_search_docs_list =  json.loads(search_text).get("pages",[])
    logger.info(f"{rewritten_query}问题对应联网查询的结果:{web_search_docs_list}")
    return web_search_docs_list