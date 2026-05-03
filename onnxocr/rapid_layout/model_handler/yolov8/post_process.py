# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import List, Tuple

import numpy as np

from ..utils import multiclass_nms, rescale_boxes, xywh2xyxy


class YOLOv8PostProcess:
    def __init__(self, labels: List[str], conf_thres=0.7, iou_thres=0.5):
        self.labels = labels
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.input_width, self.input_height = None, None
        self.img_width, self.img_height = None, None

    def __call__(
        self,
        output: List[np.ndarray],
        ori_img_shape: Tuple[int, int],
        img_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        self.img_height, self.img_width = ori_img_shape
        self.input_height, self.input_width = img_shape

        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        labels = [self.labels[i] for i in class_ids[indices]]
        return boxes[indices], scores[indices], labels

    def extract_boxes(self, predictions: np.ndarray) -> np.ndarray:
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = rescale_boxes(
            boxes, self.input_width, self.input_height, self.img_width, self.img_height
        )

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes
