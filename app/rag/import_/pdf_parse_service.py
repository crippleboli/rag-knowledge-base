import shutil
import time
import requests
from app.process.import_.agent.state import ImportGraphState
from pathlib import Path
from app.rag.import_.config import MINERU_MODEL_VERSION, MINERU_DOWNLOAD_TIMEOUT_SECONDS, MINERU_POLL_TIMEOUT_SECONDS, \
    MINERU_POLL_INTERVAL_SECONDS
from app.shared.runtime.logger import logger, PROJECT_ROOT, step_log
from app.infra.config.providers import infra_config




# 包装为Path对象
@step_log('validate_pdf_paths')
def validate_pdf_paths(state: ImportGraphState) -> tuple[Path,Path]:
    """
        检查输入/输出路径，统一转为 Path 对象。

        功能：
          1. 拦截：pdf_path 为空或文件不存在，直接报错拦截。
          2. 兜底：local_path 为空则走默认目录，不存在则自动创建。

        :param state: 全局状态
        :return: (PDF文件Path对象, 输出目录Path对象)
        """
    # 1. 读取 pdf_path  local_dir
    pdf_path = state.get('pdf_path')
    local_path = state.get('local_dir')

    # 2. 检查pdf_path源文件路径是否为空  源文件路径为空则无法继续
    if not pdf_path:
        logger.error(f"进行pdf转md过程中,检测到pdf_path为空,无法继续进行")
        raise ValueError('进行pdf转md过程中,检测到pdf_path为空,无法继续进行')

    # 3. local_dir为空时  未指定时直接写入默认输出目录
    if not local_path:
        logger.error(f'进行pdf转md过程中发现local_dir为空,进行赋值为默认值:项目/output处理')
        local_dir = PROJECT_ROOT / 'output'     # 默认值

    # 4. 转为 Path 对象 方便后续调用方法处理
    pdf_path_obj = Path(pdf_path)
    local_path_obj = Path(local_path)

    # 5. 校验 PDF 文件是否存在
    if not pdf_path_obj.exists():
        logger.error(f'进行pdf转md过程中,pdf_path值为{pdf_path},实际不存在该文件,业务无法继续')
        raise FileNotFoundError(f'进行pdf转md过程中,pdf_path值为{pdf_path},实际不存在该文件,业务无法继续')

    # 6. 输出目录不存在时自动创建
    if not local_path_obj.exists():
        logger.warning(f'进行pdf转md过程中,local_dir值为{local_path_obj},实际不存在该文件,进行自行创建处理')
        local_path_obj.mkdir(parents=True, exist_ok=True)       # 递归创建父目录 & 已创建时忽略

    # 7. 返回两个obj
    return pdf_path_obj, local_path_obj









# 获取解析后的下载url
@step_log('upload_pdf_and_poll')
def upload_pdf_and_poll(pdf_path_obj: Path) -> str:
    """
        上传本地PDF至MinerU云端，并轮询等待解析完成，获取结果下载链接

        流程:
          1. 校验MinerU配置项(Base URL & API Key)
          2. 申请上传凭证(batch_id)与临时通道(file_urls)
          3. 以二进制流(PUT)方式上传PDF文件
          4. 阻塞式轮询状态，成功则返回压缩包下载地址

        :param pdf_path_obj: 本地PDF文件的Path对象
        :return: 远端解析成果包(.zip)的完整下载URL字符串
        :raises ValueError: 核心配置参数缺失
        :raises RuntimeError: 远端服务接口报错或解析失败
        :raises TimeoutError: 轮询获取结果超时
        """

    # 1. 校验 MinerU 配置
    if not infra_config.mineru.base_url or not infra_config.mineru.api_key:
        logger.error(f'minerU请求核心参数为空(base_url 或 api_key）,业务无法继续进行')
        raise ValueError(f"minerU请求核心参数为空（base_url或api_key），业务无法继续进行")



    # 2. 调用 申请上传地址 与 batch_id
    """官方文档要求 
    Python 请求示例（适用于pdf、doc、ppt、excel、图片文件）：

        import requests
        token = "官网申请的api token"
        url = "https://mineru.net/api/v4/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name":"demo.pdf", "data_id": "abcd"}
            ],
            "model_version":"vlm"
        }
        file_path = ["demo.pdf"]
        try:
            response = requests.post(url,headers=header,json=data)
            if response.status_code == 200:
                result = response.json()
                print('response success. result:{}'.format(result))
                if result["code"] == 0:
                    batch_id = result["data"]["batch_id"]
                    urls = result["data"]["file_urls"]
                    print('batch_id:{},urls:{}'.format(batch_id, urls))
                    for i in range(0, len(urls)):
                        with open(file_path[i], 'rb') as f:
                            res_upload = requests.put(urls[i], data=f)
                            if res_upload.status_code == 200:
                                print(f"{urls[i]} upload success")
                            else:
                                print(f"{urls[i]} upload failed")
                else:
                    print('apply upload url failed,reason:{}'.format(result["msg"]))
            else:
                print('response not success. status:{} ,result:{}'.format(response.status_code, response))
        except Exception as err:
            print(err)
    
    """

    token = infra_config.mineru.api_key
    url = f'{infra_config.mineru.base_url}/file-urls/batch'
    header = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}',
    }
    data = {
        'files':[
            {'name':f"{pdf_path_obj.name}"}
        ],
        'model_version': MINERU_MODEL_VERSION,
    }


    try:
        # post请求向服务器提交数据  返回请求状态对象
        response = requests.post(url, headers=header, json=data, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)# 请求头 数据 超时时间


        # ====================================================================================
        # MinerU 官方标准响应体 (response.json()) 完整 JSON 结构示例：
        # ====================================================================================
        """
        {
            "code": 0,                                         # [int]    接口业务状态码。0: 成功；非0: 失败
            "msg": "ok",                                       # [string] 接口提示信息。成功时通常固定为 "ok"
            "trace_id": "c876cd60b202f2396de1f9e39a1b0172",     # [string] 全局请求唯一ID，用于前后端链路排查与纠错
            "data": {                                          # [dict]   核心业务数据包裹层
                "batch_id": "2bb2f0ec-a336-4a0a-b61a-472ef91", # [string] 此次PDF解析任务的唯一批次ID (UUID字符串)
                "file_urls": [                                 # [list]   云端存储预签名临时上传通道链接列表
                    "https://mineru.oss-cn-shanghai.aliyuncs.com/api-upload/your_pdf_name.pdf?Expires=1717316000&OSSAccessKeyId=LTAI5t7...&Signature=vXz..."
                ]                                              #  ▲ 注意：即便单文件上传，这里也是列表，代码中需用 [0] 显式提取
            }
        }
        """
        # ====================================================================================


        # 检查 HTTP 状态码 是否为正常200
        if response.status_code != 200:
            logger.error(f'服务器发生异常!无法进行业务!响应状态码为:{response.status_code}')
            raise RuntimeError(f'服务器发生异常!无法进行业务!响应状态码为:{response.status_code}')

        # 判断 接口状态码 是否为正常0
        response_dict = response.json()     # 解析为字典
        code = response_dict.get('code')    # 如果没有 code 键 code 变量就被赋值为 None
        if code != 0:
            logger.error(f'MinerU服务器接口发生异常无法进行业务,响应状态码为:{response.status_code}')
            raise RuntimeError(f'MinerU服务器接口发生异常无法进行业务,响应状态码为:{response.status_code}')

        # 解析响应
        batch_id = response_dict.get('data',{}).get('batch_id')     # 批量提取任务 ID 用于后续轮询解析结果
        upload_file_url = response_dict.get('data',{}).get('file_urls')[0]  # 预签名的文件上传临时链接列表
        logger.info(f"调用 `/file-urls/batch` 申请上传地址与 `batch_id`, batch_id:{batch_id},上传地址:{upload_file_url}")
    except Exception as e:
        logger.exception(f"向minerU申请上传文件地址发生异常, url参数: {url},key参数:{token}")
        raise e




    # 3. 使用 HTTP 会话对象 上传 PDF 文件
    try:
        with requests.Session() as session: # HTTP 会话对象
            session.trust_env = False       # 禁用系统代理 防止MinerU 检测到异常断开
            put_response = session.put(upload_file_url,data = pdf_path_obj.read_bytes())    # 将本地 PDF 文件转为二进制字节流，通过 PUT 方法直传云端存储
            if put_response.status_code != 200:
                logger.error(f"向地址:{upload_file_url}上传文件发生异常,状态码:{put_response.status_code},业务无法继续")
                raise RuntimeError(f"向地址:{upload_file_url}上传文件发生异常,状态码:{put_response.status_code},业务无法继续")
    except Exception as e:
        logger.exception(f"向minerU文件服务器上传文件发生异常{str(e)}! 业务无继续!!")
        raise e



    # 4. 根据 batch_id 轮询任务状态
    # 官方示例文档:
    """
    通过 batch_id 批量查询提取任务的进度。
    
    import requests
    token = "官网申请的api token"
    batch_id = "上一步批量提交返回的 batch_id"
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    res = requests.get(url, headers=header)
    print(res.status_code)
    print(res.json())
    print(res.json()["data"])
    """

    get_zip_url = f'{infra_config.mineru.base_url}/extract-results/batch/{batch_id}'
    timeout = MINERU_DOWNLOAD_TIMEOUT_SECONDS       # 超时时限
    interval_time = MINERU_POLL_INTERVAL_SECONDS    # 轮询间隔
    start_time = time.time()                        # 开始时间

    while True:
        # 1. 超时判断
        if time.time() - start_time > timeout:
            logger.error(f"轮询获取:{batch_id}结果超时,用时:{time.time() - start_time}")
            raise TimeoutError(f"轮询获取:{batch_id}结果超时,用时:{time.time() - start_time}")
        # 2. 发起网络请求
        try:
            get_response = requests.get(get_zip_url, headers=header)
        except Exception as e:
            logger.warning(f'获取下载zipurl地址请求失败,等待后继续尝试')
            time.sleep(interval_time)
            continue

        # 3. 检查 HTTP 状态码 是否为正常200
        if get_response.status_code != 200:
            # 服务器问题可以等待后重试
            if 500 <= get_response.status_code < 600:   # MinerU服务器问题
                logger.warning(f"获取下载的zipurl地址,与minerU服务器链接发生异常! 状态码:{get_response.status_code},等待后再次尝试!!")
                time.sleep(interval_time+2)
                continue
            logger.error(f"获取下载的zipurl地址,与minerU服务器链接发生异常! 状态码:{get_response.status_code},业务无法继续!")
            raise RuntimeError(f"获取下载的zipurl地址,与minerU服务器链接发生异常! 状态码:{get_response.status_code},业务无法继续!")

        # 4. 判断 接口状态码 是否为正常0
        get_response_dict = get_response.json() # 解析为字典
        if get_response_dict.get('code') != 0:
            logger.error(
                f"获取下载的zipurl地址,minerU服务器接口发生异常! 业务码:{get_response_dict.get('code')} ,错误信息:"
                f"{get_response_dict.get('msg')},业务无法继续")
            raise RuntimeError(
                f"获取下载的zipurl地址,minerU服务器接口发生异常! 业务码:{get_response_dict.get('code')} ,错误信息:"
                f"{get_response_dict.get('msg')},业务无法继续")


        # 5. 全部正常时 尝试获取结果
        # 响应示例:
        """
        {
          "code": 0,
          "data": {
            "batch_id": "2bb2f0ec-a336-4a0a-b61a-241afaf9cc87",
            "extract_result": [
              {
                "file_name": "example.pdf",
                "state": "done",            # 利用判断状态
                "err_msg": "",
                "full_zip_url": "https://cdn-mineru.openxlab.org.cn/pdf/018e53ad-d4f1-475d-b380-36bf24db9914.zip"
              },
              {
                "file_name": "demo.pdf",
                "state": "running",
                "err_msg": "",
                "extract_progress": {
                  "extracted_pages": 1,
                  "total_pages": 2,
                  "start_time": "2025-01-20 11:43:20"
                }
              }
            ]
          },
          "msg": "ok",
          "trace_id": "c876cd60b202f2396de1f9e39a1b0172"
        }
        """

        result_dict = get_response_dict.get('data',{}).get('extract_result',[])[0]
        result_state = result_dict.get('state','failed')

        if result_state == 'done':
            full_zip_url = result_dict.get('full_zip_url')
            if not full_zip_url:    # 下载地址为空
                logger.error(f"获取下载的zipurl地址,minerU对应服务器发生异常!full_zip_url为空,解析失败业务无法继续进行")
                raise RuntimeError(f"获取下载的zipurl地址,minerU对应服务器发生异常! full_zip_url为空,解析失败业务无法继续进行")
            return full_zip_url

        if result_state == 'failed':    # 下载地址为空
            logger.error(f"获取下载的zipurl地址,minerU对应服务器发生异常! result_state == 'failed',解析失败了业务无法继续进行")
            raise RuntimeError(f"获取下载的zipurl地址,minerU对应服务器发生异常! result_state == 'failed', 解析失败业务无法继续进行")

        # 正在解析
        logger.warning(f"{pdf_path_obj.name}minerU正在解析中......")
        time.sleep(interval_time)








@step_log('download_and_extract_markdown')
def download_and_extract_markdown(zip_url:str,local_dir_path_obj:Path,stem:str) -> Path:
    """

    :param zip_url: 获取的MinerU解析结果下载地址
    :param local_dir_path_obj:
    :param stem: 原pdf 的纯文件名
    :return:
    """

    # 1. MinerU 根据解析后的下载地址下载
    response = requests.get(zip_url,timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)

    if response.status_code != 200: # HTTP 链接状态码
        logger.error(f"下载地址:{zip_url}下载失败,响应状态码为:{response.status_code},业务无法继续进行")
        raise RuntimeError(f"下载地址:{zip_url}下载失败,响应状态码为:{response.status_code},业务无法继续进行")

    # 2，zip 保存到输出目录
    # 目标存储位置
    zip_path_obj:Path = local_dir_path_obj / f'{stem}_result.zip'   # pathlib运算符重载实现 字符串和Path对象直接拼接获取新的Path对象
    zip_path_obj.write_bytes(response.content)                      # 二进制下载


    # 3. 清理旧目录并重新解压
    zip_extract_dir_obj = local_dir_path_obj / stem
    # 判断是否是真实有效的文件夹
    if zip_extract_dir_obj.is_dir():
        # 清空解压文件夹
        shutil.rmtree(zip_extract_dir_obj)

    # 没有
    zip_extract_dir_obj.mkdir(parents=True, exist_ok=True)
    # 解压
    shutil.unpack_archive(zip_path_obj,zip_extract_dir_obj)

    # 4. 解压目录中递归查找 .md 文件
    md_file_obj_list = list(zip_extract_dir_obj.rglob('*.md'))
    if not md_file_obj_list or len(md_file_obj_list) == 0: # 没有md文件
        logger.error(f"下载地址：{zip_url}下载成功，解压后发现没有任何md文件，业务无法继续进行")
        raise RuntimeError(f"下载地址:{zip_url}下载成功，解压后发现没有任何md文件，业务无法继续进行")

    # 5. 优先选择与原 pdf 同名的 md文件
    for md_file_obj in md_file_obj_list:
        # 解压文件名 == 原始文件夹名
        if md_file_obj.stem == stem:
            logger.info(f"解压的文件名就是原文件名，无需二次处理：{md_file_obj.stem}")
            return md_file_obj

    # 6. 若没有同名文件 则退化选择 full.md 或 第一个 md文件
    target_md_obj = None
    # 取full文件名
    for md_file_obj_new in md_file_obj_list:
        if md_file_obj_new.name.lower() == 'full.md':
            target_md_obj = md_file_obj_new
            break

    # 异常兜底不规则命名
    if not target_md_obj:
        target_md_obj = md_file_obj_list[0]


    # 7. 同一重命名为 stem.md 并返回路径
    # Path(full) . rename(目标命名) 重命名,并且会修改磁盘文件名称 ||  with_name () 获取修改名称,但是他不改变磁盘
    # target_md_obj.with_name(f"{stem}.md")   xx/full.md -> Path => xx/文件名.md 不会修改磁盘
    # rename(新的地址)  target_md_obj -> 改成目标path 修改磁盘
    logger.info(f"进行解压md文件重命名,原名称:{target_md_obj} , 目标名:{stem}.md")
    return target_md_obj.rename(target_md_obj.with_name(f"{stem}.md"))







@step_log('parse_pdf_to_markdown')
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """

    :param state:
    :return:
    """
    # 1. pdf dir 路径校验和完善
    pdf_path_obj,local_dir_obj = validate_pdf_paths(state)
    # 2. pdf 上传 url地址获取
    zip_url = upload_pdf_and_poll(pdf_path_obj)
    # 3. 下载解压并返回md_path的Path 对象
    md_path_obj = download_and_extract_markdown(zip_url,local_dir_obj,pdf_path_obj.stem)
    # 4. 修改 state状态 md_path: str|md_content
    state['md_content'] = md_path_obj.read_text(encoding='utf-8')
    state['md_path'] = str(md_path_obj)
    return state



# 1，pathlib 模块中，stem 是 Path 对象的一个属性，专门用来获取纯文件名