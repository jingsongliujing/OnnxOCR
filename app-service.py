import cv2
import time
import base64
import numpy as np
from flask import Flask, request, jsonify,render_template
from onnxocr.onnx_paddleocr import ONNXPaddleOcr

# 初始化 Flask 应用
app = Flask(__name__)

# 初始化 OCR 模型
model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def ocr_service():
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid request, 'image' field is required."}), 400

        # 解码 base64 图像
        image_base64 = data["image"]
        try:
            image_bytes = base64.b64decode(image_base64)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Failed to decode image from base64."}), 400
        except Exception as e:
            return jsonify({"error": f"Image decoding failed: {str(e)}"}), 400

        # 执行 OCR
        start_time = time.time()
        result = model.ocr(img)
        end_time = time.time()
        processing_time = end_time - start_time

        # 格式化结果
        ocr_results = []
        for line in result[0]:
            # 确保 line[0] 是 NumPy 数组或列表
            if isinstance(line[0], (list, np.ndarray)):
                # 将 bounding_box 转换为 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] 格式
                bounding_box = np.array(line[0]).reshape(4, 2).tolist()  # 转换为 4x2 列表
            else:
                bounding_box = []

            ocr_results.append({
                "text": line[1][0],  # 识别文本
                "confidence": float(line[1][1]),  # 置信度
                "bounding_box": bounding_box  # 文本框坐标
            })

        # 返回结果
        return jsonify({
            "processing_time": processing_time,
            "results": ocr_results
        })

    except Exception as e:
        # 捕获所有异常并返回错误信息
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # 启动 Flask 服务
    app.run(host="0.0.0.0", port=5005, debug=False)
