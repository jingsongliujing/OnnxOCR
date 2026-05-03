from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from ..utils import ModelType

InputType = Union[str, np.ndarray, bytes, Path]


class PPPreProcess:
    def __init__(self, img_size: Tuple[int, int], model_type: ModelType):
        self.size = img_size
        if model_type in [ModelType.PP_DOCLAYOUT_L, ModelType.PP_DOCLAYOUT_PLUS_L, ModelType.PP_DOCLAYOUTV2, ModelType.PP_DOCLAYOUTV3]:
            self.mean = np.array([0, 0, 0])
            self.std = np.array([1.0, 1.0, 1.0])
        else:
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
        self.scale = 1 / 255.0

    def __call__(self, img: Optional[np.ndarray] = None) -> np.ndarray:
        if img is None:
            raise ValueError("img is None.")

        img = self.resize(img) #Resize
        img = self.normalize(img) #Normalize
        img = self.permute(img) #ToCHWImage
        img = np.expand_dims(img, axis=0) #ToBatch
        return img.astype(np.float32)

    def resize(self, img: np.ndarray) -> np.ndarray:
        resize_h, resize_w = self.size
        # img = cv2.resize(img, (int(resize_w), int(resize_h)))
        img = cv2.resize(img, (int(resize_w), int(resize_h)), interpolation=2)  # interp: 2
        return img

    def normalize(self, img: np.ndarray) -> np.ndarray:
        return (img.astype("float32") * self.scale - self.mean) / self.std

    def permute(self, img: np.ndarray) -> np.ndarray:
        return img.transpose((2, 0, 1))
