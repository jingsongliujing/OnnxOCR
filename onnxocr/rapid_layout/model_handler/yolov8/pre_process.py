# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import Tuple

import cv2
import numpy as np


class YOLOv8PreProcess:
    def __init__(self, img_size: Tuple[int, int]):
        self.img_size = img_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        input_img = cv2.resize(image, self.img_size)
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        return input_img[np.newaxis, :, :, :].astype(np.float32)
