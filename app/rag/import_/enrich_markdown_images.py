from pathlib import Path
import re
from typing import List, Dict
import mimetypes
import base64
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_provider
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.rate_limit_utils import apply_api_rate_limit
from app.infra.object_storage.minio_gateway import minio_gateway

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}     # scan_images函数中 支持处理的图片文件后缀


@step_log('load_markdown_and_image_dir')
def load_markdown_and_image_dir(state) -> tuple[str,Path,Path]:
    """
    从状态中读取到 md_content md_path images_path_obj
    :param state: LangGraph全局状态
    :return:   md_content   md_path_obj     images_path_obj
    """
    # 1. 获取参数 md_content md_path
    md_path = state.get("md_path")
    md_content = state.get("md_content")

    # 2. md_path非空校验
    if not md_path:
        logger.error("md_path为空,无法获取图片地址等,业务无法继续")
        raise ValueError("md_path为空,无法获取图片地址等,业务无法继续")

    # 3. md_content进行非空校验 / 空给与默认值
    md_path_obj:Path = Path(md_path)
    if not md_content:
        logger.info(f"md_content没有内容,尝试根据md_path进行二次读取")   # 解决之前节点 只保存md路径 没读取保存内容的情况
        md_content = md_path_obj.read_text(encoding="utf-8")
        if not md_content:
            logger.error(f"从{md_path}读取md_content内容失败,业务无法继续进行")
            raise ValueError(f"从{md_path}读取md_content内容失败,业务无法继续进行")
        # state 没有md_content,但是重新读取到了md_content时 更改state方便后续使用
        state['md_content'] = md_content
    # 4. 获取images对应Path
    images_path_obj = md_path_obj.parent / "images"

    return md_content,md_path_obj,images_path_obj


@step_log('scan_images')
def scan_images(md_content:str,image_path_obj:Path,context_length:int=100) -> list[tuple[str,str,tuple[str,str]]]:
    """
        扫描 images 目录与 Markdown 内容的关联关系，提取被引用图片及其上下文
    :param md_content: Markdown 文本内容
    :param image_path_obj: images 目录路径（Path对象）
    :param context_length: 截取图片前后上下文的字符长度
    :return:
        list[tuple[str, str, tuple[str, str]]]:
            [
                (图片文件名,
                 图片完整路径,
                 (图片前文context, 图片后文context))
            ]
    """
    images_context = []
    #   从image_path_obj中获取每一个文件
    for image_file_obj in image_path_obj.iterdir():     # Path.iterdir() 当前目录下一层的所有文件 / 文件夹，且每一个都是 Path 对象
        image_name = image_file_obj.name
        if not image_file_obj.suffix in SUPPORTED_IMAGE_EXTENSIONS:         # 判断是不是图片
            # 不是图片
            logger.warning(f"文件:{image_name}不是图片类型,无需处理并跳过本次循环")
            continue

        #2. 定义图片正则规则
        #  ![任意内容](任意路径 + demo.png + 任意内容)
        reg = re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+".*?\)")   # 构建 Pattern 对象
        match =  reg.search(md_content)                         # 返回第一个满足正则表达式的匹配对象 re.Match / None

        #3.match校验: 图片存在，但 Markdown 没引用
        if not match:
            logger.warning(f"图片:{image_name}没有被md内容引用!无需处理并跳过本次循环")
            continue

        #4.match 为每张图片构建局部语义窗口
        start,end = match.span()            # 图片语法![任意内容](任意路径 + demo.png + 任意内容) 的 开始位置 结束位置
        pre_context = md_content[max(start-context_length,0):start]     # 防止切片切到负数
        post_context = md_content[end:min(end+context_length,len(md_content))]  # 防止切片切出文本

        images_context.append(          # 列表
            (
                image_name,
                str(image_file_obj),    # Path对象的路径字符串
                (
                    pre_context,        # 前文
                    post_context        # 后文
                )
            )
        )

    logger.info(f"完成了图片的上下文提取: {images_context}")
    return images_context


@step_log("summarize_images")
def summarize_images(image_context_list: list[tuple[str, str, tuple[str, str]]], stem: str) -> Dict[str, str]:
    """
    进行图片意图识别
    :param image_context_list: 图片名 地址 上下文
    :param stem: 图片所在的文件夹
    :return: {图片和对应的含义}
    """
    # 1. 获取视觉模型对象
    vision_model = llm_provider.vision_chat()
    # 2. 存储含义的字典
    images_summary_dict:Dict[str,str] = {}
    # 3. 循环取出图片及其上下文输入ai总结
    for image_name,image_path,(pre_context,post_context)  in image_context_list:
        apply_api_rate_limit()        # 添加访问限制

        # 4. 加载封装提示词: 提示词名称 文件名 上下文
        # 文字提示词
        image_summary_prompt = load_prompt("image_summary" , root_folder=stem,image_content=(pre_context,post_context))
        # 图片提示词
        """
        图片文件 -> base64字符串
            base64.b64encode(文件.read_bytes()) -> 原始的字节转成base64处理的字节   .decode("utf-8") 转成base64字符串
        base64字符串 -> 原始的字节数据
            base64.b64decode(base64字符串) -> bytes
        """
        image_path_obj = Path(image_path)
        image_base_str = base64.b64encode(image_path_obj.read_bytes()).decode(encoding="utf-8")
        # https://help.aliyun.com/zh/model-studio/vision#bc4fd98b485d
        human_message = HumanMessage(
            content =  [
                {
                    # 图片的内容
                    "type": "image_url",
                    # 图片具体内容
                    # http地址
                    # base64     data:图片类型;base64,base64字符串
                    # import mimetypes  . guess_type (文件名 带后缀名)
                    "image_url": {"url": f"data:{mimetypes.guess_type(image_name)[0]};base64,{image_base_str}"},
                },
                # 图片对应的辅助描述
                {"type": "text", "text": f"{image_summary_prompt}"},
            ]
        )
        # 5. 调用视觉模型
        # 普通写法
        """        
            response = vision_model.invoke(human_message)
            response.content
        """
        # chains
        vision_chains = vision_model | StrOutputParser()
        image_summary = vision_chains.invoke([human_message])   # message列表 []
        # 6. 存储到对应字典
        images_summary_dict[image_name] = image_summary

    logger.info(f"完成图片内容识别,识别结果为: {images_summary_dict}")
    return images_summary_dict



@step_log("upload_images_and_replace")
def upload_images_and_replace(image_context_list: list[tuple[str, str, tuple[str, str]]],
            image_summaries_dict: Dict[str, str], md_content: str, stem: str) -> str:
    """
        进行minio的文件上传和md_content内容替换
    :param image_context_list:  [(图片名,图片地址,(上,下))]
    :param image_summaries_dict: {图片名:描述}
    :param md_content: md内容 ![](./)
    :param stem  eg: 烫金机
    :return: 新的md_content md内容 ![描述](http...)
    """

    # 存储图片的路径
    """
       object_name
          image_dir -> 所有图片的公共前缀
              stem ->  对应每个文件的文件夹 方便进行文件的删除和查看
                 image_name.jpg -> 具体的图片
    """
    # 1. 删除原文件在minio中存储的图片信息

    list_object = minio_gateway.client().list_objects(   # 查询要删除的对象列表
        bucket_name=minio_gateway.bucket_name,
        prefix=f"{minio_gateway.image_dir[1:]}/{stem}",  # 删除指定文件夹对应的图片 [1:] 用于去除多余的开头/
        recursive=True
    )

    # 不理解查看官方文档: 把 list_objects 查询出来的文件对象，转换成 remove_objects 需要的删除对象列表
    delete_object_list = [ DeleteObject(lo.object_name) for lo in list_object]

    # 根据删除对象列表 进行删除
    errors = minio_gateway.client().remove_objects(
        bucket_name=minio_gateway.bucket_name,
        delete_object_list=delete_object_list
    )

    for error in errors:
        # 修改老师内容: logger.warning(f"删除文件出现异常! {error}")
        logger.warning(
            f"删除失败: {error.object_name}, "
            f"错误码:{error.code}, "
            f"错误信息:{error.message}"
        )

    logger.info("文件已删除")

    # 后续步骤涉及的两个字典:
    """
    image_minio_url_dict:Dict[str,str]   {图片名 : MinIO访问地址}      自定义后循环保存获得
    image_summaries_dict: Dict[str, str] {图片名 : 图片描述总结}        summarize_images函数循环调用视觉模型总结获取
    """

    # 2. 循环传递图片到minio服务器
    image_minio_url_dict:Dict[str,str] = {}

    for image_name,image_path_str, _ in image_context_list:    # [ (图片文件名, 图片完整路径, (图片前文context, 图片后文context)]
        try:
            object_name = f"{minio_gateway.image_dir}/{stem}/{image_name}"  # 固定的前缀 / 文件名 / 图片名
            minio_gateway.client().fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=object_name,
                file_path=image_path_str,
                content_type=mimetypes.guess_type(image_name)[0]
            )
            # 3. 存储每张图片对应的minio的网络地址
            image_minio_url_dict[image_name] = minio_gateway.build_image_url(stem,image_name)
        except Exception as e:
            logger.warning(f"{image_name}的图片上传失败!执行跳过并继续上传")


    # 4. 循环遍历并替换:
    """
        用 image_summaries_dict 获取图片描述
        用 image_minio_url_dict 获取 MinIO URL
        替换 Markdown 中原始图片语法 ![](image_name) -> ![图片描述](图片URL)
    """
    for  image_name, image_ur in image_minio_url_dict.items():
        image_summary = image_summaries_dict[image_name]    # 获取单张图片总结

        reg = re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+r".*?\)")  # 正则表达式对象
        # re.sub(匹配模式,替换内容)
        r"""
            eg.sub(f"![{image_summary}]({image_ur})",md_content) 会解析\1、\2 等分组引用
            使用匿名函数 sub只会调用一次 不进行返回值解析 防止干扰
        """
        md_content = reg.sub(lambda _ : f"![{image_summary}]({image_ur})",md_content)

    # 5. 返回新md_content
    return md_content





@step_log("back_up_new_md_content")
def back_up_new_md_content(md_content_new, md_path_obj) -> str:
    """
       备份新的md_content内容
    :param md_content_new:  内容
    :param md_path_obj:  原地址 _new.md
    :return: 新的字符串地址
    """
    # 新的地址 Path对象 不创建文件
    """
        with_name() ：仅在内存中创建一个新的 Path 对象，文件系统不动
        write_text(): 根据 Path 对象写入内容到磁盘，文件不存在则新建，存在则覆盖
        rename()    ：直接修改磁盘上文件的名字或位置，不生成新文件
    """
    new_md_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    # 创建文件并写入 / 覆盖数据
    new_md_path_obj.write_text(md_content_new,encoding="utf-8")
    return str(new_md_path_obj)



@step_log("enrich_markdown_images")
def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """
    # 1. state 中获取操作参数: md内容 md地址 images地址
    md_content,md_path_obj,image_path_obj = load_markdown_and_image_dir(state)

    # 2. 校验image_path_obj是否存在内容
    if not any(image_path_obj.iterdir()):   # 没有图片也一定有images 此时没有内容
        logger.warning(f"当前{md_content}没有图片,无需图片处理,正常进入下一个节点")
        return state    # 直接结束进行下一节点

    # 3. 识别md_content图片的上下文
    """
        list[tuple[str, str, tuple[str, str]]]:
        [
            (图片文件名,
             图片完整路径,
             (图片前文context, 图片后文context))
        ]
    """
    images_context : List[tuple[str,str,tuple[str,str]]] = scan_images(md_content,image_path_obj)

    # 4. 使用视觉模型对图片进行意图识别
    # {图片的.png : 描述 }
    images_summary_dict =  summarize_images(images_context, md_path_obj.stem)

    # 5. 上传图片并且替换md_content
    md_content_new = upload_images_and_replace(images_context,images_summary_dict, md_content, md_path_obj.stem)

    # 6. 备份新的md_content_new -> md_path_obj  烫金机.md  烫金机_new.md
    new_md_path_str = back_up_new_md_content(md_content_new,md_path_obj)
    # 7. 更新state md_content md_path
    state['md_content'] = md_content_new
    state['md_path'] = new_md_path_str
    # 8. 返回结果
    return state