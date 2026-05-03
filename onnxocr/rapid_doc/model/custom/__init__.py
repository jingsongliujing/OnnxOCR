from abc import ABC, abstractmethod
import numpy as np

class CustomBaseModel(ABC):
    """自定义模型基类

    定义所有模型的通用接口
    """
    @abstractmethod
    def batch_predict(self, image_list: list[np.ndarray], **kwargs) -> list[str]:
        """批量识别

        Args:
            image_list: 图片列表
            **kwargs: 其他参数

        Returns:
            结果列表
        """
        pass
