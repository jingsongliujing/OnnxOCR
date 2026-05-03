import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .inference_engine import create_session


DEFAULT_ORIENTATION_MODEL = (
    Path(__file__).resolve().parent / "models" / "orientation" / "rapid_orientation.onnx"
)


class RapidOrientationClassifier:
    """RapidOrientation wrapper compatible with PaddleOCR's text classifier output."""

    def __init__(self, args):
        self.model_path = getattr(args, "orientation_model_dir", None) or str(
            DEFAULT_ORIENTATION_MODEL
        )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RapidOrientation model not found: {self.model_path}")

        self.batch_num = int(getattr(args, "orientation_batch_num", 6))
        self.cls_thresh = float(getattr(args, "orientation_thresh", 0.6))
        self.session = create_session(
            self.model_path,
            use_gpu=getattr(args, "use_gpu", False),
            gpu_id=getattr(args, "gpu_id", 0),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.labels = self._load_labels()

    def __call__(self, img_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[List]]:
        img_list = [self._ensure_bgr(img).copy() for img in img_list]
        img_num = len(img_list)
        cls_res = [["0", 0.0]] * img_num

        for beg in range(0, img_num, self.batch_num):
            end = min(img_num, beg + self.batch_num)
            batch = np.stack([self._preprocess(img_list[idx]) for idx in range(beg, end)])
            outputs = self.session.run(self.output_names, {self.input_name: batch})[0]
            probs = self._normalize_outputs(outputs)

            for offset, prob in enumerate(probs):
                idx = beg + offset
                label_idx = int(np.argmax(prob))
                label = self.labels[label_idx] if label_idx < len(self.labels) else str(label_idx)
                score = float(prob[label_idx])
                cls_res[idx] = [label, score]
                if score >= self.cls_thresh:
                    img_list[idx] = self._rotate_to_upright(img_list[idx], label)

        return img_list, cls_res

    def predict_image(self, img: np.ndarray) -> Tuple[str, float]:
        batch = self._preprocess(self._ensure_bgr(img))[None, ...]
        outputs = self.session.run(self.output_names, {self.input_name: batch})[0]
        prob = self._normalize_outputs(outputs)[0]
        label_idx = int(np.argmax(prob))
        label = self.labels[label_idx] if label_idx < len(self.labels) else str(label_idx)
        return label, float(prob[label_idx])

    def _load_labels(self) -> List[str]:
        meta = self.session.get_modelmeta().custom_metadata_map
        labels = meta.get("character", "")
        if labels:
            return labels.splitlines()
        return ["0", "90", "180", "270"]

    @classmethod
    def _preprocess(cls, img: np.ndarray) -> np.ndarray:
        img = cls._resize_short(img, resize_short=256)
        img = cls._center_crop(img, size=224)
        img = img.astype("float32") / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        img = (img - mean) / std
        return img.transpose((2, 0, 1)).astype(np.float32)

    @staticmethod
    def _resize_short(img: np.ndarray, resize_short: int = 256) -> np.ndarray:
        img_h, img_w = img.shape[:2]
        if img_h <= 0 or img_w <= 0:
            raise ValueError("Invalid image for orientation classification.")
        scale = float(resize_short) / min(img_w, img_h)
        width = max(1, int(round(img_w * scale)))
        height = max(1, int(round(img_h * scale)))
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def _center_crop(img: np.ndarray, size: int = 224) -> np.ndarray:
        img_h, img_w = img.shape[:2]
        if img_h < size or img_w < size:
            scale = float(size) / min(img_w, img_h)
            img = cv2.resize(
                img,
                (int(math.ceil(img_w * scale)), int(math.ceil(img_h * scale))),
                interpolation=cv2.INTER_LANCZOS4,
            )
            img_h, img_w = img.shape[:2]
        left = max(0, (img_w - size) // 2)
        top = max(0, (img_h - size) // 2)
        return img[top : top + size, left : left + size, :]

    @staticmethod
    def _normalize_outputs(outputs: np.ndarray) -> np.ndarray:
        outputs = outputs.astype(np.float32)
        row_sums = np.sum(outputs, axis=1, keepdims=True)
        if np.all(outputs >= 0) and np.allclose(row_sums, 1.0, atol=1e-4):
            return outputs
        outputs = outputs - np.max(outputs, axis=1, keepdims=True)
        exp = np.exp(outputs)
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def _ensure_bgr(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    @staticmethod
    def _rotate_to_upright(img: np.ndarray, label: str) -> np.ndarray:
        label = str(label)
        if label == "90":
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if label == "180":
            return cv2.rotate(img, cv2.ROTATE_180)
        if label == "270":
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img
