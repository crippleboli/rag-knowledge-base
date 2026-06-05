from minio import Minio
from app.shared.clients.minio_utils import get_minio_client
from app.infra.config.providers import infra_config



class MinioGateway:
    """
    封装minio的gateway minio对外提供 属性 和 方法的网关
    对外的属性: bucket_name  image_dir
    对外的函数: client()  build_image_url()
    """
    # 使用dataclass:
    """
    MinioGateway是服务类 而不是数据类 使用dataclass意味着实例化时变为对象上的固定属性 丧失了动态实时性
    from dataclasses import dataclass
    bucket_name: str = infra_config.minio.bucket_name
    image_dir: str = infra_config.minio.minio_img_dir
    """

    # 使用 @ property 装饰器: 将方法包装成属性 可以实现动态更新 以及 安全只读
    @property
    def bucket_name(self):
        return infra_config.minio.bucket_name

    @property
    def image_dir(self):
        return infra_config.minio.minio_img_dir


    def client(self):
        """
        :return: 获取统一入口 单例复用 自动初始化 bucket 的 MinIO 客户端管理器对象
        """
        return get_minio_client()

    def build_image_url(self, stem:str, object_name:str):
        """
        协议 :// 端点:9000  / 桶 / minio_img_dir /  文件名 / 对象名
        http://39.105.7.90:9000 / ergouzi / minio_img_dir / hak180使用说明书 / demo.png
        :param stem:
        :param object_name:
        :return:    获取直接访问图片的 URL 地址
        """

        protocol = "https" if infra_config.minio.minio_secure else "http"

        return (
            f"{protocol}://{infra_config.minio.endpoint}/{infra_config.minio.bucket_name}"
            f"{infra_config.minio.minio_img_dir}/{stem}/{object_name}"
        )

minio_gateway = MinioGateway()