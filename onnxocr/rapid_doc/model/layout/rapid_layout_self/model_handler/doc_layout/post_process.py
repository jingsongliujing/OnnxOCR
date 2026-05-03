# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import List, Tuple

from .utils import scale_boxes


class DocLayoutPostProcess:
    def __init__(self, labels: List[str], conf_thres=0.2, iou_thres=0.5):
        self.labels = labels
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.input_width, self.input_height = None, None
        self.img_width, self.img_height = None, None

    def __call__(
        self,
        preds,
        ori_img_shape: Tuple[int, int],
        img_shape: Tuple[int, int] = (1024, 1024),
    ):
        preds = preds[0]
        mask = preds[..., 4] > self.conf_threshold
        preds = [p[mask[idx]] for idx, p in enumerate(preds)][0]
        preds[:, :4] = scale_boxes(list(img_shape), preds[:, :4], list(ori_img_shape))

        boxes = preds[:, :4]
        confidences = preds[:, 4]
        class_ids = preds[:, 5].astype(int)
        labels = [self.labels[i] for i in class_ids]
        return boxes, confidences, labels
