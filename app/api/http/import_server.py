import shutil
import sys
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from app.api.schema.import_schema import TaskStatusSchema, UploadSchema
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.process.import_.agent.main_graph import kb_import_app
from app.process.import_.agent.state import get_default_state, ImportGraphState, create_default_state
from app.infra.config.providers import settings
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status, add_running_task, add_done_task,
)


app = FastAPI(      # 服务端应用实例
    title=settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行与状态查询。",
    version="0.2.0",
)

# 跨域问题 CORS
app.add_middleware(
    CORSMiddleware,
    # 主机:端口
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 返回import.html文件
@app.get("/html")           # get请求的 路由路径: http://主机IP:端口/html
def html():
    html_path_obj = PROJECT_ROOT / "app" / "resources" / "html" / "import.html"
    return FileResponse(    # 返回文件响应
        path=html_path_obj,                 # 文件路径
        media_type=guess_type(html_path_obj.name)[0]    # 文件类型
    )

# 2. 任务状态查询接口: 返回task_id对应的任务状态
@app.get("/status/{task_id}")
def task_status(task_id:str):
    logger.info(f"获取任务状态接口被调用,task_id:{task_id}")
    return TaskStatusSchema(        # 返回 TaskStatusSchema 对象
        code=200,
        task_id=task_id,
        status= get_task_status(task_id),
        done_list= get_done_task_list(task_id),
        running_list= get_running_task_list(task_id)
    )

def invoke_graph(task_id:str,local_file_path:Path,local_dir:Path):

    state = create_default_state(task_id=task_id,local_file_path=str(local_file_path),local_dir=str(local_dir))

    try:
        logger.info(f"{task_id}对应的文件解析任务开始执行! 参数state:{state}")
        update_task_status(task_id,TASK_STATUS_PROCESSING)
        final_state = kb_import_app.invoke(state)           # 调用图对象 .invoke
        logger.info(f"{task_id}对应的文件解析任务完成! 最终结果为:{final_state}")
        update_task_status(task_id,TASK_STATUS_COMPLETED)
    except Exception as e:
        update_task_status(task_id,TASK_STATUS_FAILED)
        logger.exception(f"===== 全流程测试运行失败 =====")


# 3. 上传文件的异步接口   post /upload  files : 文件列表  后天执行图过程
@app.post("/upload")
def upload_and_invoke_graph(backgroundtasks:BackgroundTasks,files:list[UploadFile]):

    """
        1. 接收上传的文件 (文件存储到项目下)
        2. 异步执行导入图对象 (state local_file_path , local_dir , task_id )  10 20s
        3. 直接返回结果
    :param backgroundtasks:
    :param files:
    :return:
    """
    # 1. 准备存储路径和文件夹: D:\项目名\output\20260610\任务_123

    task_id = str(uuid.uuid4())     # 不使用 random模块伪随机
    add_running_task(task_id, "upload_file")
    local_dir_path_obj = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id  # Path 对象
    local_dir_path_obj.mkdir(parents=True,exist_ok=True)

    # 2 上传的文件存储到准备的路径下
    current_file = files[0]     # 上传文件列表中的第一个
    local_file_path_obj = local_dir_path_obj / current_file.filename

    with local_file_path_obj.open("wb") as file_buffer: # 二进制写入
        shutil.copyfileobj(current_file.file, file_buffer)  # 网络数据流 -> 流式缓冲区 -> 硬盘文件

    add_done_task(task_id,"upload_file")


    # 2. 异步调用图解析 local_file_path_obj local_dir_path_obj task_id
    # # 除首参数外，其余实参的键值对必须与目标异步函数的【函数签名（形参列表）】严格契合，由底层在运行时自动反射解包
    backgroundtasks.add_task(
        invoke_graph,                       # 异步执行的函数名,
        task_id=task_id,                    # 任务id
        local_file_path=local_file_path_obj,# 上传文件在本地的路径
        local_dir=local_dir_path_obj        # 文件根目录
    )

    # 3. 返回结果
    return UploadSchema(    # 返回 上传文件的响应数据类型
        code=200,
        message="文件上传成功!",
        task_ids=[task_id]
    )

# 测试
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.import_app_port)