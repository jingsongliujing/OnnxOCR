# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import time

import numpy as np

from ...inference_engine.base import InferSession
from ...utils.typings import RapidLayoutOutput
from ..base import BaseModelHandler
from .post_process import DocLayoutPostProcess
from .pre_process import DocLayoutPreProcess


class DocLayoutModelHandler(BaseModelHandler):
    def __init__(self, labels, conf_thres, iou_thres, session: InferSession):
        self.img_size = (1024, 1024)
        self.preprocess = DocLayoutPreProcess(img_size=self.img_size)
        self.postprocess = DocLayoutPostProcess(labels, conf_thres, iou_thres)

        self.session = session

    def __call__(self, ori_img: np.ndarray) -> RapidLayoutOutput:
        s1 = time.perf_counter()

        ori_img_shape = ori_img.shape[:2]

        img = self.preprocess(ori_img)
        preds = self.session(img)
        boxes, scores, class_names = self.postprocess(
            preds, ori_img_shape, self.img_size
        )

        elapse = time.perf_counter() - s1
        return RapidLayoutOutput(
            img=ori_img,
            boxes=boxes,
            class_names=class_names,
            scores=scores,
            elapse=elapse,
        )

    def preprocess(self, image):
        return self.preprocess(image)

    def postprocess(self, preds, ori_img_shape, img_shape):
        return self.postprocess(preds, ori_img_shape, img_shape)
