import os
import time

import cv2

from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Img


def run_general_ocr():
    model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)
    img = cv2.imread("./onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg")
    if img is None:
        raise RuntimeError("Failed to read general OCR test image.")

    start = time.time()
    result = model.ocr(img)
    print("general OCR total time: {:.3f}".format(time.time() - start))
    print("general OCR result:", result)
    for box in result[0]:
        print(box)

    os.makedirs("./result_img", exist_ok=True)
    sav2Img(img, result, name="./result_img/test_ocr_general.jpg")


def run_plate_ocr():
    plate_model = ONNXPaddleOcr(
        use_angle_cls=True,
        use_gpu=False,
        use_plate_recognition=True,
        plate_min_score=0.4,
    )
    img = cv2.imread("./onnxocr/test_images/license_plate_single_blue.jpg")
    if img is None:
        raise RuntimeError("Failed to read license plate OCR test image.")

    start = time.time()
    result = plate_model.ocr(img)
    print("plate OCR total time: {:.3f}".format(time.time() - start))
    print("plate OCR result:", result)


if __name__ == "__main__":
    run_general_ocr()
    run_plate_ocr()
