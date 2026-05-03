from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from .logger import Logger
from .utils import save_img

logger = Logger(logger_name=__name__).get_log()

PP_DOCLAYOUT_PLUS_L_Threshold = {
    0: 0.3,   # paragraph_title
    1: 0.5,   # image
    2: 0.4,   # text
    3: 0.5,   # number
    4: 0.5,   # abstract
    5: 0.5,   # content
    6: 0.5,   # figure_table_chart_title
    7: 0.3,   # formula
    8: 0.5,   # table
    9: 0.5,   # reference
    10: 0.5,  # doc_title
    11: 0.5,  # footnote
    12: 0.5,  # header
    13: 0.5,  # algorithm
    14: 0.5,  # footer
    15: 0.45, # seal
    16: 0.5,  # chart
    17: 0.5,  # formula_number
    18: 0.5,  # aside_text
    19: 0.5,  # reference_content
}

PP_DOCLAYOUT_PLUS_L_layout_merge_bboxes_mode = {
    0: "large",  # paragraph_title
    1: "large",  # image
    2: "union",  # text
    3: "union",  # number
    4: "union",  # abstract
    5: "union",  # content
    6: "union",  # figure_table_chart_title
    7: "large",  # formula
    8: "union",  # table
    9: "union",  # reference
    10: "union",  # doc_title
    11: "union",  # footnote
    12: "union",  # header
    13: "union",  # algorithm
    14: "union",  # footer
    15: "union",  # seal
    16: "large",  # chart
    17: "union",  # formula_number
    18: "union",  # aside_text
    19: "union", # reference_content
}

PP_DOCLAYOUT_L_Threshold = {
    0: 0.3,    # paragraph_title
    1: 0.5,    # image
    2: 0.4,    # text
    3: 0.5,    # number
    4: 0.5,    # abstract
    5: 0.5,    # content
    6: 0.5,    # figure_title (默认值)
    7: 0.5,    # formula
    8: 0.5,    # table
    9: 0.5,    # table_title (默认值)
    10: 0.5,   # reference
    11: 0.5,   # doc_title
    12: 0.5,   # footnote
    13: 0.5,   # header
    14: 0.5,   # algorithm
    15: 0.5,   # footer
    16: 0.45,  # seal
    17: 0.5,   # chart_title (默认值)
    18: 0.5,   # chart
    19: 0.5,   # formula_number
    20: 0.5,   # header_image (默认值)
    21: 0.5,   # footer_image (默认值)
    22: 0.5,    # aside_text
}

PP_DOCLAYOUTV2_Threshold = {
    0: 0.5,  # abstract
    1: 0.5,  # algorithm
    2: 0.5,  # aside_text
    3: 0.5,  # chart
    4: 0.5,  # content
    5: 0.5,  # formula
    6: 0.4,  # doc_title
    7: 0.5,  # figure_title
    8: 0.5,  # footer
    9: 0.5,  # footer
    10: 0.5,  # footnote
    11: 0.5,  # formula_number
    12: 0.5,  # header
    13: 0.5,  # header
    14: 0.5,  # image
    15: 0.5,  # inline_formula
    16: 0.5,  # number
    17: 0.4,  # paragraph_title
    18: 0.5,  # reference
    19: 0.5,  # reference_content
    20: 0.45,  # seal
    21: 0.5,  # table
    22: 0.4,  # text
    23: 0.4,  # vertical_text
    24: 0.5,  # vision_footnote
}

PP_DOCLAYOUTV2_layout_merge_bboxes_mode= {
    0: "union",  # abstract
    1: "union",  # algorithm
    2: "union",  # aside_text
    3: "large",  # chart
    4: "union",  # content
    5: "large",  # display_formula
    6: "large",  # doc_title
    7: "union",  # figure_title
    8: "union",  # footer
    9: "union",  # footer
    10: "union",  # footnote
    11: "union",  # formula_number
    12: "union",  # header
    13: "union",  # header
    14: "union", # image
    15: "large",  # inline_formula
    16: "union",  # number
    17: "large",  # paragraph_title
    18: "union",  # reference
    19: "union",  # reference_content
    20: "union",  # seal
    21: "union",  # table
    22: "union",  # text
    23: "union",  # text
    24: "union",  # vision_footnote
}

class ModelType(Enum):
    PP_DOCLAYOUT_PLUS_L = "pp_doclayout_plus_l"
    PP_DOCLAYOUTV2 = "pp_doclayoutv2"
    PP_DOCLAYOUTV3 = "pp_doclayoutv3"
    PP_DOCLAYOUT_L = "pp_doclayout_l"
    PP_DOCLAYOUT_M = "pp_doclayout_m"
    PP_DOCLAYOUT_S = "pp_doclayout_s"
    DOCLAYOUT_DOCSTRUCTBENCH = "doclayout_docstructbench"
    RT_DETR_L_WIRED_TABLE_CELL_DET = "rt_detr_l_wired_table_cell_det"
    RT_DETR_L_WIRELESS_TABLE_CELL_DET = "rt_detr_l_wireless_table_cell_det"


class EngineType(Enum):
    ONNXRUNTIME = "onnxruntime"
    OPENVINO = "openvino"


@dataclass
class RapidLayoutInput:
    model_type: ModelType = ModelType.PP_DOCLAYOUTV2
    model_dir_or_path: Union[str, Path, None] = None

    engine_type: EngineType = EngineType.ONNXRUNTIME
    engine_cfg: dict = field(default_factory=dict)

    conf_thresh: Union[float, dict] = None
    iou_thresh: float = 0.5
    layout_shape_mode: Optional[str] = "auto" # "rect" / "auto"


@dataclass
class RapidLayoutOutput:
    img: Optional[np.ndarray] = None
    boxes: Optional[List[List[float]]] = None
    polygon_points: Optional[List[List[float]]] = None
    class_names: Optional[List[str]] = None
    scores: Optional[List[float]] = None
    orders: Optional[List[int]] = None
    elapse: Optional[float] = None

    def vis(self, save_path: Union[str, Path, None] = None) -> Optional[np.ndarray]:
        if self.img is None or self.boxes is None:
            logger.warning("No image or boxes to visualize.")
            return None

        from .vis_res import VisLayout

        vis_img = VisLayout.draw_detections(
            self.img,
            np.array(self.boxes),
            self.polygon_points,
            np.array(self.scores),
            np.array(self.class_names),
            self.orders,
        )
        if save_path is not None and vis_img is not None:
            save_img(save_path, vis_img)
            logger.info(f"Visualization saved as {save_path}")

        return vis_img

    def crop(self) -> List[np.ndarray]:
        if self.img is None or self.boxes is None:
            logger.warning("No image or boxes to visualize.")
            return None
        img_list = []
        polygon_pointses = self.polygon_points if self.polygon_points is not None else [None] * len(self.boxes)
        for xyxy, polygon_points in zip(self.boxes, polygon_pointses):
            input_res = {
                "bbox": xyxy,
                "polygon_points": polygon_points
            }
            return_image, _ = crop_img(input_res, self.img)
            img_list.append(return_image)
        return img_list


def crop_img(input_res, input_img: np.ndarray, crop_paste_x=0, crop_paste_y=0, layout_shape_mode="auto"):

    crop_xmin, crop_ymin = int(input_res['bbox'][0]), int(input_res['bbox'][1])
    crop_xmax, crop_ymax = int(input_res['bbox'][2]), int(input_res['bbox'][3])

    # Calculate new dimensions
    crop_new_width = crop_xmax - crop_xmin + crop_paste_x * 2
    crop_new_height = crop_ymax - crop_ymin + crop_paste_y * 2

    # Create a white background array
    return_image = np.ones((crop_new_height, crop_new_width, 3), dtype=np.uint8) * 255

    # Crop the original image using numpy slicing
    cropped_img = input_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    polygon = input_res.get("polygon_points")
    if layout_shape_mode != "rect" and polygon:
        polygon = np.array(polygon, dtype=np.int32)
        if polygon.ndim == 1:
            polygon = polygon.reshape((-1, 2))
        polygon = polygon.reshape((-1, 1, 2))
        polygon = polygon - np.array([crop_xmin, crop_ymin])
        mask = np.zeros(cropped_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        mask = mask.astype(bool)
        cropped_img = cropped_img.copy()
        cropped_img[~mask] = 255

    # Paste the cropped image onto the white background
    return_image[crop_paste_y:crop_paste_y + (crop_ymax - crop_ymin),
    crop_paste_x:crop_paste_x + (crop_xmax - crop_xmin)] = cropped_img

    return_list = [crop_paste_x, crop_paste_y, crop_xmin, crop_ymin, crop_xmax, crop_ymax, crop_new_width,
                   crop_new_height]
    return return_image, return_list
