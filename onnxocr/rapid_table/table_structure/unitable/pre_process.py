# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .consts import IMG_SIZE


class TablePreprocess:
    def __init__(self, device: str):
        self.img_size = IMG_SIZE
        self.transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.86597056, 0.88463002, 0.87491087],
                    std=[0.20686628, 0.18201602, 0.18485524],
                ),
            ]
        )

        self.device = device

    def __call__(self, imgs: List[np.ndarray]):
        processed_imgs, ori_shapes = [], []
        for img in imgs:
            if img is None:
                continue

            ori_h, ori_w = img.shape[:2]
            ori_shapes.append((ori_h, ori_w))

            processed_img = self.preprocess_img(img)
            processed_imgs.append(processed_img)
        return torch.concatenate(processed_imgs), ori_shapes

    def preprocess_img(self, ori_image: np.ndarray) -> torch.Tensor:
        image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image
