# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .logger import Logger
from .utils import save_img, save_txt


class VisTable:
    def __init__(self):
        self.logger = Logger(logger_name=__name__).get_log()

    def __call__(
        self,
        img: np.ndarray,
        pred_html: str,
        cell_bboxes: np.ndarray,
        logic_points: np.ndarray,
        save_html_path: Union[str, Path, None] = None,
        save_drawed_path: Union[str, Path, None] = None,
        save_logic_path: Union[str, Path, None] = None,
    ):
        if pred_html and save_html_path:
            html_with_border = self.insert_border_style(pred_html)
            save_txt(save_html_path, html_with_border)
            self.logger.info(f"Save HTML to {save_html_path}")

        if cell_bboxes is None:
            return None

        drawed_img = self.draw(img, cell_bboxes)
        if save_drawed_path:
            save_img(save_drawed_path, drawed_img)
            self.logger.info(f"Saved table struacter result to {save_drawed_path}")

        if save_logic_path and logic_points.size > 0:
            self.plot_rec_box_with_logic_info(
                img, save_logic_path, logic_points, cell_bboxes
            )
            self.logger.info(f"Saved rec and box result to {save_logic_path}")
        return drawed_img

    def insert_border_style(self, table_html_str: str) -> str:
        style_res = """<meta charset="UTF-8"><style>
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
        }
                    </style>"""

        prefix_table, suffix_table = table_html_str.split("<body>")
        html_with_border = f"{prefix_table}{style_res}<body>{suffix_table}"
        return html_with_border

    def draw(self, img: np.ndarray, cell_bboxes: np.ndarray) -> np.ndarray:
        dims_bboxes = cell_bboxes.shape[1]
        if dims_bboxes == 4:
            return self.draw_rectangle(img, cell_bboxes)

        if dims_bboxes == 8:
            return self.draw_polylines(img, cell_bboxes)

        raise ValueError("Shape of table bounding boxes is not between in 4 or 8.")

    @staticmethod
    def draw_rectangle(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        img_copy = img.copy()
        for box in boxes.astype(int):
            x1, y1, x2, y2 = box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return img_copy

    @staticmethod
    def draw_polylines(img: np.ndarray, points) -> np.ndarray:
        img_copy = img.copy()
        for point in points.astype(int):
            point = point.reshape(4, 2)
            cv2.polylines(img_copy, [point.astype(int)], True, (255, 0, 0), 2)
        return img_copy

    def plot_rec_box_with_logic_info(
        self, img: np.ndarray, output_path, logic_points, cell_bboxes
    ):
        img = cv2.copyMakeBorder(
            img, 0, 0, 0, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

        polygons = [[box[0], box[1], box[4], box[5]] for box in cell_bboxes]
        for idx, polygon in enumerate(polygons):
            x0, y0, x1, y1 = polygon[0], polygon[1], polygon[2], polygon[3]
            x0 = round(x0)
            y0 = round(y0)
            x1 = round(x1)
            y1 = round(y1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)

            # 增大字体大小和线宽
            font_scale = 0.9  # 原先是0.5
            thickness = 1  # 原先是1
            logic_point = logic_points[idx]
            cv2.putText(
                img,
                f"row: {logic_point[0]}-{logic_point[1]}",
                (x0 + 3, y0 + 8),
                cv2.FONT_HERSHEY_PLAIN,
                font_scale,
                (0, 0, 255),
                thickness,
            )
            cv2.putText(
                img,
                f"col: {logic_point[2]}-{logic_point[3]}",
                (x0 + 3, y0 + 18),
                cv2.FONT_HERSHEY_PLAIN,
                font_scale,
                (0, 0, 255),
                thickness,
            )

        save_img(output_path, img)
