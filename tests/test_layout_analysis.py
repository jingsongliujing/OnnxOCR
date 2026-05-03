import time
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2LayoutImg
from tests.common import TEST_IMAGE_DIR, ensure_result_dir


def run_layout_analysis():
    model = ONNXPaddleOcr(
        use_gpu=False,
        use_layout_analysis=True,
        layout_model_type="pp_layout_cdla",
    )
    img = cv2.imread(str(TEST_IMAGE_DIR / "layout_cdla.jpg"))
    if img is None:
        raise RuntimeError("Failed to read layout analysis test image.")

    start = time.time()
    result = model.ocr(img)
    print("layout analysis total time: {:.3f}".format(time.time() - start))
    print("layout analysis result count:", len(result["boxes"]))

    sav2LayoutImg(img, result, name=str(ensure_result_dir() / "test_layout_vis.jpg"))
    return result


if __name__ == "__main__":
    run_layout_analysis()
