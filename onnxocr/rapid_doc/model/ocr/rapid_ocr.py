from typing import Any, List, Optional

import cv2
import numpy as np

from onnxocr.onnx_paddleocr import ONNXPaddleOcr


class RapidOcrModel:
    """RapidDoc OCR adapter backed by this project's ONNX PaddleOCR.

    RapidDoc expects a small OCR object with `ocr`, `__call__`, and
    `det_batch_predict`.  Keeping that contract here lets the document pipeline
    reuse OnnxOCR models without depending on the external rapidocr package.
    """

    def __init__(
        self,
        det_db_box_thresh: float = 0.3,
        lang: Optional[str] = None,
        ocr_config: Optional[dict] = None,
        use_dilation: bool = True,
        det_db_unclip_ratio: float = 1.8,
        enable_merge_det_boxes: bool = True,
        is_seal: bool = False,
    ) -> None:
        del lang, use_dilation, enable_merge_det_boxes, is_seal

        ocr_config = ocr_config or {}
        kwargs = {
            "use_angle_cls": bool(ocr_config.get("Global.use_cls", False)),
            "use_gpu": bool(ocr_config.get("use_gpu", False)),
            "det_db_box_thresh": det_db_box_thresh,
            "det_db_unclip_ratio": det_db_unclip_ratio,
        }

        config_map = {
            "Det.model_path": "det_model_dir",
            "Rec.model_path": "rec_model_dir",
            "Cls.model_path": "cls_model_dir",
            "Rec.rec_keys_path": "rec_char_dict_path",
        }
        for source_key, target_key in config_map.items():
            if ocr_config.get(source_key):
                kwargs[target_key] = ocr_config[source_key]

        self.ocr_engine = ONNXPaddleOcr(**kwargs)
        self.drop_score = self.ocr_engine.drop_score

    def __call__(self, img: np.ndarray, mfd_res: Optional[Any] = None):
        del mfd_res
        result = self.ocr(img, det=True, rec=True)[0]
        if not result:
            return [], []
        boxes = [np.asarray(item[0], dtype=np.float32) for item in result]
        rec_res = [tuple(item[1]) for item in result]
        return boxes, rec_res

    def ocr(
        self,
        img,
        det: bool = True,
        rec: bool = True,
        cls: bool = False,
        mfd_res: Optional[Any] = None,
        tqdm_enable: bool = False,
        tqdm_desc: str = "OCR-rec Predict",
        return_word_box: bool = False,
        ori_img: Optional[np.ndarray] = None,
        dt_boxes: Optional[List] = None,
    ):
        del mfd_res, tqdm_enable, tqdm_desc, return_word_box, ori_img, dt_boxes

        if det:
            return self.ocr_engine.ocr(img, det=det, rec=rec, cls=cls)

        crops = img if isinstance(img, list) else [img]
        rec_res = self.ocr_engine.text_recognizer(crops)
        return [rec_res]

    def det_batch_predict(self, img_list: List[np.ndarray], max_batch_size: int = 8):
        del max_batch_size
        results = []
        for img in img_list:
            det_res = self.ocr(img, det=True, rec=False)[0]
            boxes = None if det_res is None else np.asarray(det_res, dtype=np.float32)
            results.append((boxes, 0.0))
        return results


if __name__ == "__main__":
    demo = cv2.imread("tests/test_files/ch.jpg")
    if demo is not None:
        print(RapidOcrModel().ocr(demo))
