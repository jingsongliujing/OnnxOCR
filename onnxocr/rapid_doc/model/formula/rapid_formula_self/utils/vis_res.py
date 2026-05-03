from typing import Optional, Tuple

import cv2
import numpy as np


class VisLayout:
    @classmethod
    def draw_detections(
        cls,
        image: np.ndarray,
        boxes: Optional[np.ndarray],
        scores: Optional[np.ndarray],
        class_names: Optional[np.ndarray],
        mask_alpha=0.3,
    ) -> Optional[np.ndarray]:
        """_summary_

        Args:
            image (np.ndarray): H x W x C
            boxes (np.ndarray): (N, 4)
            scores (np.ndarray): (N, )
            class_ids (np.ndarray): (N, )
            mask_alpha (float, optional): _description_. Defaults to 0.3.

        Returns:
            np.ndarray: _description_
        """
        if boxes is None or scores is None or class_names is None:
            return None

        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        det_img = cls.draw_masks(det_img, boxes, mask_alpha)

        for label, box, score in zip(class_names, boxes, scores):
            color = cls.get_color()

            cls.draw_box(det_img, box, color)
            caption = f"{label} {int(score * 100)}%"
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
