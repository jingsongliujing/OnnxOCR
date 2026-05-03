from copy import deepcopy
from typing import Any, List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def caculate_bbox_area(bbox):
    """Calculate bounding box area"""
    x1, y1, x2, y2 = map(float, bbox)
    area = abs((x2 - x1) * (y2 - y1))
    return area

def calculate_overlap_ratio(
    bbox1: Union[list, tuple], bbox2: Union[list, tuple], mode="union"
) -> float:
    """
    Calculate the overlap ratio between two bounding boxes.

    Args:
        bbox1 (list or tuple): The first bounding box, format [x_min, y_min, x_max, y_max]
        bbox2 (list or tuple): The second bounding box, format [x_min, y_min, x_max, y_max]
        mode (str): The mode of calculation, either 'union', 'small', or 'large'.

    Returns:
        float: The overlap ratio value between the two bounding boxes
    """
    x_min_inter = max(bbox1[0], bbox2[0])
    y_min_inter = max(bbox1[1], bbox2[1])
    x_max_inter = min(bbox1[2], bbox2[2])
    y_max_inter = min(bbox1[3], bbox2[3])

    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)

    inter_area = float(inter_width) * float(inter_height)

    bbox1_area = caculate_bbox_area(bbox1)
    bbox2_area = caculate_bbox_area(bbox2)

    if mode == "union":
        ref_area = bbox1_area + bbox2_area - inter_area
    elif mode == "small":
        ref_area = min(bbox1_area, bbox2_area)
    elif mode == "large":
        ref_area = max(bbox1_area, bbox2_area)
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    if ref_area == 0:
        return 0.0

    return inter_area / ref_area

def _get_minbox_if_overlap_by_ratio(
    bbox1: Union[List[int], Tuple[int, int, int, int]],
    bbox2: Union[List[int], Tuple[int, int, int, int]],
    ratio: float,
    smaller: bool = True,
) -> Optional[int]:
    """
    Determine if the overlap area between two bounding boxes exceeds a given ratio
    and return which one to drop (1 for bbox1, 2 for bbox2).

    Args:
        bbox1: Coordinates of the first bounding box [x_min, y_min, x_max, y_max].
        bbox2: Coordinates of the second bounding box [x_min, y_min, x_max, y_max].
        ratio: The overlap ratio threshold.
        smaller: If True, drop the smaller bounding box; otherwise, drop the larger one.

    Returns:
        Optional[int]: 1 if bbox1 should be dropped, 2 if bbox2 should be dropped, None otherwise.
    """
    area1 = caculate_bbox_area(bbox1)
    area2 = caculate_bbox_area(bbox2)
    overlap_ratio = calculate_overlap_ratio(bbox1, bbox2, mode="small")

    if overlap_ratio > ratio:
        if (area1 <= area2 and smaller) or (area1 >= area2 and not smaller):
            return 1
        else:
            return 2
    return None


def remove_overlap_blocks(
    bboxes: List[List[int]], threshold: float = 0.65, smaller: bool = True
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Remove overlapping bounding boxes based on a specified overlap ratio threshold.

    Args:
        bboxes: List of bounding boxes, each in format [x_min, y_min, x_max, y_max].
        threshold: Ratio threshold to determine significant overlap.
        smaller: If True, the smaller block in overlap is removed.

    Returns:
        Tuple[List[List[int]], List[List[int]]]:
            A tuple containing the updated list of bounding boxes and a list of dropped boxes.
    """
    dropped_indexes = set()
    bboxes = deepcopy(bboxes)
    dropped_boxes = []

    for i, bbox1 in enumerate(bboxes):
        for j in range(i + 1, len(bboxes)):
            bbox2 = bboxes[j]
            if i in dropped_indexes or j in dropped_indexes:
                continue

            drop_flag = _get_minbox_if_overlap_by_ratio(
                bbox1, bbox2, threshold, smaller=smaller
            )

            if drop_flag is not None:
                drop_index = i if drop_flag == 1 else j
                dropped_indexes.add(drop_index)

    for index in sorted(dropped_indexes, reverse=True):
        dropped_boxes.append(bboxes[index])
        del bboxes[index]

    return bboxes, dropped_boxes


from typing import Optional, Tuple

import cv2
import numpy as np

class VisReadOrder:
    @classmethod
    def draw_order(
        cls,
        image: np.ndarray,
        boxes: Optional[np.ndarray],
        order_indexes: Optional[np.ndarray],
        mask_alpha=0.3,
    ) -> Optional[np.ndarray]:
        if boxes is None or order_indexes is None:
            return None

        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        det_img = cls.draw_masks(det_img, boxes, mask_alpha)

        for box, order_index in zip(boxes, order_indexes):
            color = cls.get_color()

            cls.draw_box(det_img, box, color)
            caption = f"{order_index}"
            cls.draw_text(det_img, caption, box, color, font_size, text_thickness)

        return det_img

    @staticmethod
    def draw_box(
        image: np.ndarray,
        box: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(int)
        return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def draw_text(
        image: np.ndarray,
        text: str,
        box: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),
        font_size: float = 0.001,
        text_thickness: int = 2,
    ) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(int)
        (tw, th), _ = cv2.getTextSize(
            text=text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_size,
            thickness=text_thickness,
        )
        th = int(th * 1.2)

        cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

        return cv2.putText(
            image,
            text,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    @classmethod
    def draw_masks(
        cls,
        image: np.ndarray,
        boxes: np.ndarray,
        mask_alpha: float = 0.3,
    ) -> np.ndarray:
        mask_img = image.copy()
        for box in boxes:
            color = cls.get_color()
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

    @staticmethod
    def get_color():
        colors = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        return colors
