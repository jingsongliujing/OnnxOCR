from typing import Optional, Tuple, List
import cv2
import numpy as np


class VisLayout:
    @classmethod
    def draw_detections(
        cls,
        image: np.ndarray,
        boxes: Optional[np.ndarray],
        polygon_points: Optional[List[np.ndarray]],
        scores: Optional[np.ndarray],
        class_names: Optional[np.ndarray],
        orders: Optional[List[int]],
        mask_alpha=0.3,
    ) -> Optional[np.ndarray]:

        if (boxes is None and polygon_points is None) or scores is None or class_names is None:
            return None

        det_img = image.copy()
        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        # 1. 绘制半透明 Mask (优先用多边形)
        det_img = cls.draw_masks(det_img, boxes, polygon_points, mask_alpha)

        # 准备迭代数据：如果 polygon_points 为空，则用 boxes 占位进行 zip
        iterations = polygon_points if polygon_points else boxes

        for idx, (label, shape, score) in enumerate(zip(class_names, iterations, scores)):
            color = cls.get_color()

            # 2. 绘制边框/轮廓
            if polygon_points:
                cls.draw_polygons(det_img, shape, color)
                # 计算用于放文字的参考框 (x1, y1, x2, y2)
                x, y, w, h = cv2.boundingRect(shape.astype(int))
                ref_box = np.array([x, y, x + w, y + h])
            else:
                cls.draw_box(det_img, shape, color)
                ref_box = shape

            # 3. 基础 caption
            caption = f"{label} {int(score * 100)}%"

            # 如果有 orders，追加阅读顺序
            if orders is not None:
                order_map = {box_idx: o for o, box_idx in enumerate(orders)}
                if idx in order_map:
                    caption = f"[{order_map[idx]}] " + caption

            # 4. 绘制文字
            cls.draw_text(det_img, caption, ref_box, color, font_size, text_thickness)

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
    def draw_polygons(
            image: np.ndarray,
            points: np.ndarray,
            color: Tuple[int, int, int] = (0, 0, 255),
            thickness: int = 2,
    ) -> np.ndarray:
        # points shape: (N, 2)
        pts = points.astype(np.int32).reshape((-1, 1, 2))
        return cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

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

        # 文本背景框
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
            boxes: Optional[np.ndarray],
            polygon_points: Optional[List[np.ndarray]],
            mask_alpha: float = 0.3,
    ) -> np.ndarray:
        mask_img = image.copy()

        if polygon_points:
            for pts in polygon_points:
                color = cls.get_color()
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask_img, [pts], color)
        elif boxes is not None:
            for box in boxes:
                color = cls.get_color()
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

    @staticmethod
    def get_color():
        return (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )