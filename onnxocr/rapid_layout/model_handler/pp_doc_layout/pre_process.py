# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

InputType = Union[str, np.ndarray, bytes, Path]


class PPDocLayoutPreProcess:
    def __init__(self, img_size: Tuple[int, int]):
        self.size = [800, 800]

        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]
        self.scale = 1 / 255.0
        self.alpha = [self.scale / self.std[i] for i in range(len(self.std))]
        self.beta = [-self.mean[i] / self.std[i] for i in range(len(self.std))]

    def __call__(self, img: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if img is None:
            raise ValueError("img is None.")

        data = self.resize(img)
        data = self.normalize(data)
        data = self.permute(data)
        ori_data = copy.deepcopy(data)
        batch_inputs = self.to_batch(data)
        return ori_data, batch_inputs

    def resize(self, img: np.ndarray):
        resize_h, resize_w = self.size
        img_ori_h, img_ori_w = img.shape[:2]

        img = cv2.resize(
            img, (int(resize_w), int(resize_h)), interpolation=cv2.INTER_CUBIC
        )
        img_h, img_w = img.shape[:2]
        data = {
            "img": img,
            "img_size": [img_w, img_h],
            "scale_factors": [img_w / img_ori_w, img_h / img_ori_h],
            "ori_img_size": [img_ori_w, img_ori_h],
        }
        return data

    def normalize(self, data: Dict[str, Any]) -> np.ndarray:
        img = data["img"]
        split_im = list(cv2.split(img))
        for c in range(img.shape[2]):
            split_im[c] = split_im[c].astype(np.float32)
            split_im[c] *= self.alpha[c]
            split_im[c] += self.beta[c]

        res = cv2.merge(split_im)
        data["img"] = res
        return data

    def permute(self, data: Dict[str, Any]) -> np.ndarray:
        img = data["img"]
        data["img"] = img.transpose((2, 0, 1))
        return data

    def to_batch(self, data, dtype: np.dtype = np.float32) -> list[np.ndarray]:
        result = []
        for key in ["img_size", "img", "scale_factors"]:
            if key == "img_size":
                val = [data[key][::-1]]
            elif key == "scale_factors":
                val = [data.get(key, [1.0, 1.0])[::-1]]
            else:
                val = [data[key]]
            result.append(np.array(val, dtype=dtype))
        return result
