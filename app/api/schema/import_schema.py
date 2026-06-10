from pydantic import BaseModel


# 上传文件的响应数据类型
# 当用户在前端点击“上传文档”按钮，前端把文件推给后端后，后端处理完会立刻返回这个结构
class UploadSchema(BaseModel):
    code:int = 200
    message:str
    task_ids:list[str]  # 对应的任务id  因为可以上传多个 所以使用列表


# 查询任务状态的数据类型
class TaskStatusSchema(BaseModel):
    code:int = 200
    task_id:str         # 当前查询的任务ID
    status:str          # "processing": 处理中
                        # "completed": 已完成
                        # "failed": 失败
    done_list:list[str] # 已完成节点列表
    running_list:list[str]  # 运行中的节点