import time
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2PlateImg
from tests.common import TEST_IMAGE_DIR, ensure_result_dir


def run_plate_ocr():
    model = ONNXPaddleOcr(
        use_angle_cls=True,
        use_gpu=False,
        use_plate_recognition=True,
        plate_min_score=0.4,
    )
    img = cv2.imread(str(TEST_IMAGE_DIR / "license_plate_single_blue.jpg"))
    if img is None:
        raise RuntimeError("Failed to read license plate OCR test image.")

    start = time.time()
    result = model.ocr(img)
    print("plate OCR total time: {:.3f}".format(time.time() - start))
    print("plate OCR result:", result)

    sav2PlateImg(img, result, name=str(ensure_result_dir() / "test_plate_vis.jpg"))
    return result


if __name__ == "__main__":
    run_plate_ocr()
