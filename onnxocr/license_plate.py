import base64
import copy
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .inference_engine import create_session

PLATE_CHARS = (
    "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
    "学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
)
MEAN_VALUE = 0.588
STD_VALUE = 0.193


class LicensePlateRecognizer:
    """ONNX license plate detector and recognizer."""

    def __init__(
        self,
        detect_model_path: Optional[str] = None,
        rec_model_path: Optional[str] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "models", "license_plate")
        self.detect_model_path = detect_model_path or os.path.join(
            model_dir, "car_plate_detect.onnx"
        )
        self.rec_model_path = rec_model_path or os.path.join(model_dir, "plate_rec.onnx")
        self.providers = list(providers) if providers is not None else None

        self.session_detect = create_session(
            self.detect_model_path, providers=self.providers
        )
        self.session_rec = create_session(
            self.rec_model_path, providers=self.providers
        )

    def recognize(
        self,
        img: np.ndarray,
        min_score: float = 0.4,
        iou_thresh: float = 0.5,
    ) -> List[Dict]:
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("img must be a decoded OpenCV image.")

        img_size = (640, 640)
        img0 = copy.deepcopy(img)
        blob, ratio, pad_left, pad_top = self._detect_preprocess(img, img_size)
        outputs = self.session_detect.run(
            [self.session_detect.get_outputs()[0].name],
            {self.session_detect.get_inputs()[0].name: blob},
        )[0]
        boxes = self._postprocess(outputs, ratio, pad_left, pad_top, min_score, iou_thresh)
        return self._recognize_plates(boxes, img0, min_score)

    def recognize_base64(self, image_base64: str, min_score: float = 0.4) -> List[Dict]:
        image_bytes = base64.b64decode(image_base64)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from base64.")
        return self.recognize(img, min_score=min_score)

    @staticmethod
    def _decode_plate(preds: np.ndarray) -> str:
        previous = 0
        decoded = []
        for pred in preds:
            pred = int(pred)
            if pred != 0 and pred != previous and pred < len(PLATE_CHARS):
                decoded.append(PLATE_CHARS[pred])
            previous = pred
        return "".join(decoded)

    def _recognize_text(self, img: np.ndarray) -> str:
        img = cv2.resize(img, (168, 48))
        img = img.astype(np.float32)
        img = (img / 255 - MEAN_VALUE) / STD_VALUE
        img = img.transpose(2, 0, 1)
        img = img.reshape(1, *img.shape)

        output = self.session_rec.run(
            [self.session_rec.get_outputs()[0].name],
            {self.session_rec.get_inputs()[0].name: img},
        )[0]
        index = np.argmax(output[0], axis=1)
        return self._decode_plate(index)

    @staticmethod
    def _letterbox(
        img: np.ndarray, size: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, float, int, int]:
        height, width, _ = img.shape
        ratio = min(size[0] / height, size[1] / width)
        new_h, new_w = int(height * ratio), int(width * ratio)
        top = int((size[0] - new_h) / 2)
        left = int((size[1] - new_w) / 2)
        bottom = size[0] - new_h - top
        right = size[1] - new_w - left
        resized = cv2.resize(img, (new_w, new_h))
        boxed = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return boxed, ratio, left, top

    def _detect_preprocess(
        self, img: np.ndarray, img_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, float, int, int]:
        img, ratio, left, top = self._letterbox(img, img_size)
        img = img[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)
        img = img / 255
        img = img.reshape(1, *img.shape)
        return img, ratio, left, top

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        xyxy = copy.deepcopy(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return xyxy

    @staticmethod
    def _nms(boxes: np.ndarray, iou_thresh: float) -> List[int]:
        order = np.argsort(boxes[:, 4])[::-1]
        keep = []
        while order.size > 0:
            current = order[0]
            keep.append(current)
            x1 = np.maximum(boxes[current, 0], boxes[order[1:], 0])
            y1 = np.maximum(boxes[current, 1], boxes[order[1:], 1])
            x2 = np.minimum(boxes[current, 2], boxes[order[1:], 2])
            y2 = np.minimum(boxes[current, 3], boxes[order[1:], 3])

            width = np.maximum(0, x2 - x1)
            height = np.maximum(0, y2 - y1)
            inter_area = width * height
            current_area = (boxes[current, 2] - boxes[current, 0]) * (
                boxes[current, 3] - boxes[current, 1]
            )
            other_area = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
                boxes[order[1:], 3] - boxes[order[1:], 1]
            )
            union_area = current_area + other_area - inter_area
            iou = inter_area / np.maximum(union_area, 1e-6)
            remaining = np.where(iou <= iou_thresh)[0]
            order = order[remaining + 1]
        return keep

    @staticmethod
    def _restore_box(boxes: np.ndarray, ratio: float, left: int, top: int) -> np.ndarray:
        boxes[:, [0, 2, 5, 7, 9, 11]] -= left
        boxes[:, [1, 3, 6, 8, 10, 12]] -= top
        boxes[:, [0, 2, 5, 7, 9, 11]] /= ratio
        boxes[:, [1, 3, 6, 8, 10, 12]] /= ratio
        return boxes

    def _postprocess(
        self,
        dets: np.ndarray,
        ratio: float,
        left: int,
        top: int,
        conf_thresh: float,
        iou_thresh: float,
    ) -> np.ndarray:
        choice = dets[:, :, 4] > conf_thresh
        dets = dets[choice]
        if dets.size == 0:
            return np.empty((0, 14), dtype=np.float32)

        dets[:, 13:15] *= dets[:, 4:5]
        boxes = self._xywh_to_xyxy(dets[:, :4])
        scores = np.max(dets[:, 13:15], axis=-1, keepdims=True)
        labels = np.argmax(dets[:, 13:15], axis=-1).reshape(-1, 1)
        output = np.concatenate((boxes, scores, dets[:, 5:13], labels), axis=1)
        output = output[self._nms(output, iou_thresh)]
        return self._restore_box(output, ratio, left, top)

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        sums = pts.sum(axis=1)
        rect[0] = pts[np.argmin(sums)]
        rect[2] = pts[np.argmax(sums)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self._order_points(pts.astype("float32"))
        tl, tr, br, bl = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b), 1)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b), 1)
        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height))

    @staticmethod
    def _split_merge(img: np.ndarray) -> np.ndarray:
        height, _, _ = img.shape
        upper = img[0 : int(5 / 12 * height), :]
        lower = img[int(1 / 3 * height) :, :]
        upper = cv2.resize(upper, (lower.shape[1], lower.shape[0]))
        return np.hstack((upper, lower))

    def _recognize_plates(
        self, outputs: np.ndarray, img0: np.ndarray, min_score: float
    ) -> List[Dict]:
        results = []
        image_h, image_w = img0.shape[:2]
        for output in outputs:
            score = float(output[4])
            if score < min_score:
                continue

            landmarks = output[5:13].reshape(4, 2)
            roi_img = self._four_point_transform(img0, landmarks)
            label = int(output[-1])
            plate_type = "double_layer" if label == 1 else "single_layer"
            if label == 1:
                roi_img = self._split_merge(roi_img)

            x1, y1, x2, y2 = output[:4]
            axis = [
                int(max(0, min(image_w - 1, x1))),
                int(max(0, min(image_h - 1, y1))),
                int(max(0, min(image_w - 1, x2))),
                int(max(0, min(image_h - 1, y2))),
            ]
            results.append(
                {
                    "cls": "plate",
                    "axis": axis,
                    "score": round(score, 4),
                    "plate": self._recognize_text(roi_img),
                    "type": plate_type,
                    "landmarks": landmarks.astype(float).round(2).tolist(),
                }
            )
        return results
