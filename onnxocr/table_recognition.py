import os
from typing import Dict, List, Optional

import numpy as np

from .rapid_table import ModelType, RapidTable, RapidTableInput


TABLE_MODEL_PATHS = {
    "slanet_plus": "slanet-plus.onnx",
    "ppstructure_zh": "ch_ppstructure_mobile_v2_SLANet.onnx",
    "ppstructure_en": "en_ppstructure_mobile_v2_SLANet.onnx",
}


class TableRecognizer:
    """RapidTable wrapper that uses local ONNX table models."""

    def __init__(
        self,
        model_type: str = "slanet_plus",
        model_path: Optional[str] = None,
        engine_cfg: Optional[Dict] = None,
    ) -> None:
        if model_type not in TABLE_MODEL_PATHS:
            supported = ", ".join(TABLE_MODEL_PATHS)
            raise ValueError(f"Unsupported table model_type: {model_type}. Supported: {supported}")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = model_path or os.path.join(
            base_dir, "models", "table", TABLE_MODEL_PATHS[model_type]
        )

        input_args = RapidTableInput(
            model_type=ModelType(model_type),
            model_dir_or_path=model_path,
            engine_cfg=engine_cfg or {},
            use_ocr=True,
        )
        self.model_type = model_type
        self.model_path = model_path
        self.table_engine = RapidTable(input_args)

    def recognize(self, img: np.ndarray, ocr_result: List) -> Dict:
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("img must be a decoded OpenCV image.")

        ocr_results = [self._format_onnxocr_result(ocr_result)]
        result = self.table_engine(img, ocr_results=ocr_results)
        return {
            "html": result.pred_htmls[0] if result.pred_htmls else "",
            "cell_bboxes": self._array_to_list(result.cell_bboxes[0]) if result.cell_bboxes else [],
            "logic_points": self._array_to_list(result.logic_points[0]) if result.logic_points else [],
            "processing_time": float(result.elapse),
            "model_type": self.model_type,
        }

    @staticmethod
    def _format_onnxocr_result(ocr_result: List):
        boxes, texts, scores = [], [], []
        for line in ocr_result:
            boxes.append(line[0])
            texts.append(line[1][0])
            scores.append(float(line[1][1]))
        return np.array(boxes), tuple(texts), tuple(scores)

    @staticmethod
    def _array_to_list(value):
        if value is None:
            return []
        if hasattr(value, "tolist"):
            return value.tolist()
        return value
