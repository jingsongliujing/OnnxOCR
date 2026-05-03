import time
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Img
from tests.common import TEST_IMAGE_DIR, ensure_result_dir


def run_general_ocr():
    model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)
    img = cv2.imread(str(TEST_IMAGE_DIR / "715873facf064583b44ef28295126fa7.jpg"))
    if img is None:
        raise RuntimeError("Failed to read general OCR test image.")

    start = time.time()
    result = model.ocr(img)
    print("general OCR total time: {:.3f}".format(time.time() - start))
    print("general OCR result:", result)

    sav2Img(img, result, name=str(ensure_result_dir() / "test_ocr_general.jpg"))
    return result


if __name__ == "__main__":
    run_general_ocr()
