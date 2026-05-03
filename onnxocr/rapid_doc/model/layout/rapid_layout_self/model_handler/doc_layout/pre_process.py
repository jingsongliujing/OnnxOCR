# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from .utils import LetterBox

InputType = Union[str, np.ndarray, bytes, Path]


class DocLayoutPreProcess:
    def __init__(self, img_size: Tuple[int, int]):
        self.img_size = img_size
        self.letterbox = LetterBox(new_shape=img_size, auto=False, stride=32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        input_img = self.letterbox(image=image)
        input_img = input_img[None, ...]
        input_img = input_img[..., ::-1].transpose(0, 3, 1, 2)
        input_img = np.ascontiguousarray(input_img)
        input_img = input_img / 255
        input_tensor = input_img.astype(np.float32)
        return input_tensor
