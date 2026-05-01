# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import numpy as np

from .consts import IMG_SIZE


def rescale_bboxes(ori_h, ori_w, bboxes):
    scale_h = ori_h / IMG_SIZE
    scale_w = ori_w / IMG_SIZE
    bboxes[:, 0::2] *= scale_w
    bboxes[:, 1::2] *= scale_h
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, ori_w - 1)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, ori_h - 1)
    return bboxes
