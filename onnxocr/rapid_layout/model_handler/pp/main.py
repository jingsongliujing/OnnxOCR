# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import time

import numpy as np

from ...inference_engine.base import InferSession
from ...utils.typings import RapidLayoutOutput
from ..base import BaseModelHandler
from .post_process import PPPostProcess
from .pre_process import PPPreProcess


class PPModelHandler(BaseModelHandler):
    def __init__(self, labels, conf_thres, iou_thres, session: InferSession):
        self.img_size = (800, 608)
        self.pp_preprocess = PPPreProcess(img_size=self.img_size)
        self.pp_postprocess = PPPostProcess(labels, conf_thres, iou_thres)

        self.session = session

    def __call__(self, ori_img: np.ndarray) -> RapidLayoutOutput:
        s1 = time.perf_counter()

        ori_img_shape = ori_img.shape[:2]
        img = self.preprocess(ori_img)
        preds = self.session(img)
        boxes, scores, class_names = self.postprocess(ori_img_shape, img, preds)

        elapse = time.perf_counter() - s1
        return RapidLayoutOutput(
            img=ori_img,
            boxes=boxes,
            class_names=class_names,
            scores=scores,
            elapse=elapse,
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return self.pp_preprocess(image)

    def postprocess(self, ori_img_shape, img, preds):
        return self.pp_postprocess(ori_img_shape, img, preds)
