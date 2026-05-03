from argparse import Namespace

import cv2

from onnxocr.orientation import RapidOrientationClassifier


class RapidOrientationModel:
    """RapidDoc orientation adapter using OnnxOCR's local ONNX classifier."""

    def __init__(self):
        self.orientation_engine = RapidOrientationClassifier(
            Namespace(use_gpu=False, gpu_id=0, orientation_thresh=0.6)
        )

    def predict(self, input_img, det_res=None):
        rotate_label = "0"
        bgr_image = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        img_height, img_width = bgr_image.shape[:2]
        img_aspect_ratio = img_height / img_width if img_width > 0 else 1.0
        img_is_portrait = img_aspect_ratio > 1.2

        if not img_is_portrait:
            return rotate_label

        if det_res:
            vertical_count = 0
            for box_ocr_res in det_res:
                p1, _, p3, _ = box_ocr_res
                width = p3[0] - p1[0]
                height = p3[1] - p1[1]
                aspect_ratio = width / height if height > 0 else 1.0
                if aspect_ratio < 0.8:
                    vertical_count += 1
            if vertical_count < len(det_res) * 0.28 or vertical_count < 3:
                return rotate_label

        rotate_label, _ = self.orientation_engine.predict_image(input_img)
        return rotate_label
