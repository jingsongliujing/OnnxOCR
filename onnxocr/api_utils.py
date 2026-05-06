"""Shared utilities for OnnxOCR API services."""
import base64
import cv2
import numpy as np
from onnxocr.onnx_paddleocr import ONNXPaddleOcr
from onnxocr.layout_markdown import LayoutMarkdownConverter


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode a base64-encoded string into an OpenCV BGR image.

    Raises:
        ValueError: If the data cannot be decoded into a valid image.
    """
    image_bytes = base64.b64decode(image_base64)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from base64.")
    return img


def format_ocr_results(result) -> list:
    """Format raw OCR output into a list of dicts with text, confidence, bounding_box."""
    ocr_results = []
    for line in result[0]:
        if isinstance(line[0], (list, np.ndarray)):
            bounding_box = np.array(line[0]).reshape(4, 2).tolist()
        else:
            bounding_box = []
        ocr_results.append({
            "text": line[1][0],
            "confidence": float(line[1][1]),
            "bounding_box": bounding_box,
        })
    return ocr_results


class ModelRegistry:
    """Lazy-initialized, cached model registry for API services."""

    def __init__(self, use_gpu: bool = False):
        self._use_gpu = use_gpu
        self._ocr_model = None
        self._plate_model = None
        self._table_model = None
        self._layout_models = {}
        self._layout_markdown_converters = {}

    def get_ocr_model(self) -> ONNXPaddleOcr:
        if self._ocr_model is None:
            self._ocr_model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=self._use_gpu)
        return self._ocr_model

    def get_plate_model(self) -> ONNXPaddleOcr:
        if self._plate_model is None:
            self._plate_model = ONNXPaddleOcr(use_plate_recognition=True, use_gpu=self._use_gpu)
        return self._plate_model

    def get_table_model(self) -> ONNXPaddleOcr:
        if self._table_model is None:
            self._table_model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=self._use_gpu, use_table_recognition=True)
        return self._table_model

    def get_layout_model(self, model_type="pp_layout_cdla", conf_thresh=0.5, iou_thresh=0.5) -> ONNXPaddleOcr:
        key = (model_type, float(conf_thresh), float(iou_thresh))
        if key not in self._layout_models:
            self._layout_models[key] = ONNXPaddleOcr(
                use_layout_analysis=True,
                use_gpu=self._use_gpu,
                layout_model_type=model_type,
                layout_conf_thresh=float(conf_thresh),
                layout_iou_thresh=float(iou_thresh),
            )
        return self._layout_models[key]

    def get_layout_markdown_converter(self, model_type="pp_doclayoutv2", conf_thresh=0.4, iou_thresh=0.5) -> LayoutMarkdownConverter:
        key = (model_type, float(conf_thresh), float(iou_thresh))
        if key not in self._layout_markdown_converters:
            self._layout_markdown_converters[key] = LayoutMarkdownConverter(
                layout_model_type=model_type,
                layout_conf_thresh=float(conf_thresh),
                layout_iou_thresh=float(iou_thresh),
                ocr_kwargs={"use_angle_cls": True, "use_gpu": self._use_gpu},
            )
        return self._layout_markdown_converters[key]
