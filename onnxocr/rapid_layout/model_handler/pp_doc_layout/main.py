# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import time
from typing import Any, List

import numpy as np

from ...inference_engine.base import InferSession
from ...utils.typings import RapidLayoutOutput
from ..base import BaseModelHandler
from .post_process import PPDocLayoutPostProcess
from .pre_process import PPDocLayoutPreProcess


class PPDocLayoutModelHandler(BaseModelHandler):
    def __init__(
        self,
        labels: List[str],
        conf_thres: float,
        iou_thres: float,
        session: InferSession,
    ):
        self.img_size = (800, 800)
        self.pp_preprocess = PPDocLayoutPreProcess(img_size=self.img_size)
        self.pp_postprocess = PPDocLayoutPostProcess(labels=labels)

        self.session = session

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def __call__(self, ori_img: np.ndarray) -> RapidLayoutOutput:
        s1 = time.perf_counter()

        ori_data, ort_inputs = self.preprocess(ori_img)
        ort_outputs = self.session(ort_inputs)
        preds_list = self.format_output(ort_outputs)
        boxes, scores, class_names = self.postprocess(
            batch_outputs=preds_list,
            datas=[ori_data],
            threshold=self.conf_thres,
            layout_nms=True,
            layout_shape_mode="auto",
            filter_overlap_boxes=True,
            skip_order_labels=None,
        )

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

    def postprocess(self, **kwargs: Any):
        return self.pp_postprocess(**kwargs)

    @staticmethod
    def format_output(pred):
        box_idx_start = 0
        np_boxes_num = pred[1][0]
        box_idx_end = box_idx_start + np_boxes_num
        np_boxes = pred[0][box_idx_start:box_idx_end]
        return [{"boxes": np.array(np_boxes)}]
