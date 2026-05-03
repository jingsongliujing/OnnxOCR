import time
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2TableImg
from tests.common import TEST_IMAGE_DIR, ensure_result_dir


def run_table_ocr():
    model = ONNXPaddleOcr(
        use_angle_cls=True,
        use_gpu=False,
        use_table_recognition=True,
        table_model_type="slanet_plus",
    )
    img = cv2.imread(str(TEST_IMAGE_DIR / "table.jpg"))
    if img is None:
        raise RuntimeError("Failed to read table OCR test image.")

    start = time.time()
    result = model.ocr(img)
    print("table OCR total time: {:.3f}".format(time.time() - start))
    print("table OCR html:", result["html"][:500])

    sav2TableImg(img, result, name=str(ensure_result_dir() / "test_table_vis.jpg"))
    return result


if __name__ == "__main__":
    run_table_ocr()
