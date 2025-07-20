import cv2
import time

from onnxocr.onnx_paddleocr import ONNXPaddleOcr,sav2Img
import sys,os
import time
#import torch
#固定到onnx路径·
# sys.path.append('./paddle_to_onnx/onnx')
MODEL_OPTIONS = ["PP-OCRv5","PP-OCRv5_Server", "PP-OCRv4", "ch_ppocr_server_v2.0"]
model_name = MODEL_OPTIONS[1]  # 默认模型
model_map = {
    "PP-OCRv5": "ppocrv5",
    "PP-OCRv5_Server": "ppocrv5_sobile",
    "PP-OCRv4": "ppocrv4",
    "ch_ppocr_server_v2.0": "ch_ppocr_server_v2.0"
}
base_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "onnxocr", "models"))
model_dir = model_map.get(model_name, "ppocrv5")
model_path = os.path.join(base_model_dir, model_dir)
det_model_dir = os.path.join(model_path, "det", "det.onnx")
cls_model_dir = os.path.join(model_path, "cls", "cls.onnx")
rec_char_dict_path = os.path.join(base_model_dir, "ppocrv5", "ppocrv5_dict.txt")
rec_model_dir = os.path.join(model_path, "rec", "rec.onnx") if os.path.exists(os.path.join(model_path, "rec", "rec.onnx")) else None

ocr_kwargs = dict(
    use_angle_cls=True,
    use_gpu=True,  # 关键：传递GPU参数
    det_model_dir=det_model_dir,
    cls_model_dir=cls_model_dir,
    rec_char_dict_path=rec_char_dict_path
)
if rec_model_dir and os.path.exists(rec_model_dir):
    ocr_kwargs["rec_model_dir"] = rec_model_dir
model = ONNXPaddleOcr(**ocr_kwargs)


img = cv2.imread('./onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg')
s = time.time()
result = model.ocr(img)
e = time.time()
print("total time: {:.3f}".format(e - s))
print("result:", result)
for box in result[0]:
    print(box)

sav2Img(img, result,name=str(time.time())+'.jpg')