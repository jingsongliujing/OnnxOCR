# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import time
from typing import List

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

    # def __call__(self, ori_img_list: List[np.ndarray]) -> List[RapidLayoutOutput]:
    #     s1 = time.perf_counter()
    #     result_list = []
    #     img_inputs = []
    #     ori_img_shape_list = []
    #     for ori_img in ori_img_list:
    #         ori_img_shape = ori_img.shape[:2]
    #         img = self.preprocess(ori_img)
    #         ori_img_shape_list.append(ori_img_shape)
    #         img_inputs.append(img)
    #     img_inputs = np.concatenate(img_inputs, axis=0)  # 拼接 batch
    #
    #     batch_preds = self.session(img_inputs, None)
    #     for i, preds in enumerate(batch_preds):
    #         boxes, scores, class_names = self.postprocess(
    #             preds, ori_img_shape_list[i], self.img_size
    #         )
    #         elapse = time.perf_counter() - s1
    #         result = RapidLayoutOutput(img=ori_img_list[i], boxes=boxes, class_names=class_names,
    #                     scores=scores, elapse=elapse,)
    #         result_list.append(result)
    #
    #     return result_list

    def __call__(self, ori_img_list: List[np.ndarray]) -> List[RapidLayoutOutput]:
        s1 = time.perf_counter()
        result_list = []
        for ori_img in ori_img_list:
            ori_img_shape = ori_img.shape[:2]
            img = self.preprocess(ori_img)
            preds = self.session(img)
            boxes, scores, class_names = self.postprocess(
                preds, ori_img_shape, self.img_size
            )

            elapse = time.perf_counter() - s1
            result = RapidLayoutOutput(img=ori_img,
                boxes=boxes,
                class_names=class_names,
                scores=scores,
                elapse=elapse,
            )
            result_list.append(result)
        return result_list

    def preprocess(self, image):
        return self.preprocess(image)

    def postprocess(self, preds, ori_img_shape, img_shape):
        return self.postprocess(preds, ori_img_shape, img_shape)
