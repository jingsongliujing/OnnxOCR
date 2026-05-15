from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2

from onnxocr.onnx_paddleocr import ONNXPaddleOcr

from .base import OcrLine, SkillInput
from .extractors import normalize_onnxocr_result


class OnnxOCREngine:
    """Lazy ONNXPaddleOcr adapter used by vertical skills."""

    def __init__(self, ocr_kwargs: Optional[Dict[str, Any]] = None):
        self.ocr_kwargs = ocr_kwargs or {"use_angle_cls": False, "use_gpu": False}
        self._general_model = None
        self._table_model = None
        self._plate_model = None

    def recognize(self, skill_input: SkillInput) -> List[OcrLine]:
        if skill_input.ocr_lines is not None:
            return skill_input.ocr_lines
        img = self._load_image(skill_input)
        if self._general_model is None:
            self._general_model = ONNXPaddleOcr(**self.ocr_kwargs)
        return normalize_onnxocr_result(self._general_model.ocr(img, cls=False))

    def recognize_table(self, skill_input: SkillInput) -> Dict[str, Any]:
        img = self._load_image(skill_input)
        if self._table_model is None:
            kwargs = dict(self.ocr_kwargs)
            kwargs.update({"use_table_recognition": True})
            self._table_model = ONNXPaddleOcr(**kwargs)
        return self._table_model.ocr(img)

    def recognize_plate(self, skill_input: SkillInput) -> List[Dict[str, Any]]:
        img = self._load_image(skill_input)
        if self._plate_model is None:
            kwargs = dict(self.ocr_kwargs)
            kwargs.update({"use_plate_recognition": True})
            self._plate_model = ONNXPaddleOcr(**kwargs)
        return self._plate_model.ocr(img)

    @staticmethod
    def _load_image(skill_input: SkillInput):
        if skill_input.image is not None:
            return skill_input.image
        if not skill_input.image_path:
            raise ValueError("SkillInput requires image or image_path.")
        img = cv2.imread(skill_input.image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {skill_input.image_path}")
        return img
