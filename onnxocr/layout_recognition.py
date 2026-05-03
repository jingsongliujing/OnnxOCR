import os
from typing import Dict, Optional

import numpy as np

from .rapid_layout import EngineType, ModelType, RapidLayout, RapidLayoutInput


LAYOUT_MODEL_PATHS = {
    "pp_layout_cdla": "layout_cdla.onnx",
    "pp_layout_publaynet": "layout_publaynet.onnx",
}


class LayoutRecognizer:
    """RapidLayout wrapper that uses local Chinese/English ONNX layout models."""

    def __init__(
        self,
        model_type: str = "pp_layout_cdla",
        model_path: Optional[str] = None,
        engine_cfg: Optional[Dict] = None,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.5,
    ) -> None:
        if model_type not in LAYOUT_MODEL_PATHS:
            supported = ", ".join(LAYOUT_MODEL_PATHS)
            raise ValueError(f"Unsupported layout model_type: {model_type}. Supported: {supported}")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = model_path or os.path.join(
            base_dir, "models", "layout", LAYOUT_MODEL_PATHS[model_type]
        )

        cfg = RapidLayoutInput(
            model_type=ModelType(model_type),
            model_dir_or_path=model_path,
            engine_type=EngineType.ONNXRUNTIME,
            engine_cfg=engine_cfg or {},
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )
        self.model_type = model_type
        self.model_path = model_path
        self.layout_engine = RapidLayout(cfg=cfg)

    def recognize(self, img: np.ndarray) -> Dict:
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("img must be a decoded OpenCV image.")

        result = self.layout_engine(img)
        return {
            "boxes": result.boxes or [],
            "class_names": result.class_names or [],
            "scores": result.scores or [],
            "processing_time": float(result.elapse or 0),
            "model_type": self.model_type,
        }
