import os
import time

import cv2

from onnxocr.onnx_paddleocr import (
    ONNXPaddleOcr,
    sav2Img,
    sav2LayoutImg,
    sav2PlateImg,
    sav2TableImg,
)


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
    os.makedirs("./result_img", exist_ok=True)
    sav2PlateImg(img, result, name="./result_img/test_plate_vis.jpg")


def run_table_ocr():
    table_model = ONNXPaddleOcr(
        use_angle_cls=True,
        use_gpu=False,
        use_table_recognition=True,
        table_model_type="slanet_plus",
    )
    img = cv2.imread("./onnxocr/test_images/table.jpg")
    if img is None:
        raise RuntimeError("Failed to read table OCR test image.")

    start = time.time()
    result = table_model.ocr(img)
    print("table OCR total time: {:.3f}".format(time.time() - start))
    print("table OCR html:", result["html"][:500])
    os.makedirs("./result_img", exist_ok=True)
    sav2TableImg(img, result, name="./result_img/test_table_vis.jpg")


def run_layout_analysis():
    layout_model = ONNXPaddleOcr(
        use_gpu=False,
        use_layout_analysis=True,
        layout_model_type="pp_layout_cdla",
    )
    img = cv2.imread("./onnxocr/test_images/layout_cdla.jpg")
    if img is None:
        raise RuntimeError("Failed to read layout analysis test image.")

    start = time.time()
    result = layout_model.ocr(img)
    print("layout analysis total time: {:.3f}".format(time.time() - start))
    print("layout analysis result count:", len(result["boxes"]))
    os.makedirs("./result_img", exist_ok=True)
    sav2LayoutImg(img, result, name="./result_img/test_layout_vis.jpg")


if __name__ == "__main__":
    run_general_ocr()
    run_plate_ocr()
    run_table_ocr()
    run_layout_analysis()
