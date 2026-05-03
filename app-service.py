import cv2
import time
import base64
import os
import numpy as np
from flask import Flask, request, jsonify,render_template
from werkzeug.utils import secure_filename
from onnxocr.layout_markdown import LayoutMarkdownConverter
from onnxocr.onnx_paddleocr import ONNXPaddleOcr
from onnxocr.visualization import (
    draw_layout_analysis,
    draw_plate_recognition,
    draw_table_recognition,
    image_to_base64,
)

# 初始化 Flask 应用
app = Flask(__name__)

# 初始化 OCR 模型
model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)
plate_model = None
table_model = None
layout_models = {}
layout_markdown_converters = {}


def get_plate_model():
    global plate_model
    if plate_model is None:
        plate_model = ONNXPaddleOcr(use_plate_recognition=True, use_gpu=False)
    return plate_model


def get_table_model():
    global table_model
    if table_model is None:
        table_model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False, use_table_recognition=True)
    return table_model


def get_layout_model(model_type="pp_layout_cdla", conf_thresh=0.5, iou_thresh=0.5):
    key = (model_type, float(conf_thresh), float(iou_thresh))
    if key not in layout_models:
        layout_models[key] = ONNXPaddleOcr(
            use_layout_analysis=True,
            use_gpu=False,
            layout_model_type=model_type,
            layout_conf_thresh=float(conf_thresh),
            layout_iou_thresh=float(iou_thresh),
        )
    return layout_models[key]


def get_layout_markdown_converter(model_type="pp_doclayoutv2", conf_thresh=0.4, iou_thresh=0.5):
    key = (model_type, float(conf_thresh), float(iou_thresh))
    if key not in layout_markdown_converters:
        layout_markdown_converters[key] = LayoutMarkdownConverter(
            layout_model_type=model_type,
            layout_conf_thresh=float(conf_thresh),
            layout_iou_thresh=float(iou_thresh),
            ocr_kwargs={"use_angle_cls": True, "use_gpu": False},
        )
    return layout_markdown_converters[key]

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


@app.route('/plate', methods=['POST'])
def plate_service():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid request, 'image' field is required."}), 400

        try:
            min_score = float(data.get("min_score", 0.4))
        except (TypeError, ValueError):
            return jsonify({"error": "'min_score' must be a number."}), 400

        start_time = time.time()
        image_bytes = base64.b64decode(data["image"])
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image from base64."}), 400
        results = get_plate_model().ocr(img, plate_min_score=min_score)
        processing_time = time.time() - start_time

        response = {
            "processing_time": processing_time,
            "results": results
        }
        if data.get("visualize", False):
            response["visualization"] = image_to_base64(draw_plate_recognition(img, results))
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/table', methods=['POST'])
def table_service():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid request, 'image' field is required."}), 400

        image_bytes = base64.b64decode(data["image"])
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image from base64."}), 400

        start_time = time.time()
        result = get_table_model().ocr(img)
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        if data.get("visualize", False):
            result["visualization"] = image_to_base64(
                draw_table_recognition(img, result, show_logic=data.get("show_logic", False))
            )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/layout', methods=['POST'])
def layout_service():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid request, 'image' field is required."}), 400

        image_bytes = base64.b64decode(data["image"])
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image from base64."}), 400

        model_type = data.get("model_type", "pp_doclayoutv2")
        conf_thresh = float(data.get("conf_thresh", 0.4))
        iou_thresh = float(data.get("iou_thresh", 0.5))
        start_time = time.time()
        result = get_layout_model(model_type, conf_thresh, iou_thresh).ocr(img)
        result["processing_time"] = time.time() - start_time
        if data.get("visualize", False):
            result["visualization"] = image_to_base64(draw_layout_analysis(img, result))

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/layout_markdown', methods=['POST'])
def layout_markdown_service():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid request, 'image' field is required."}), 400

        image_bytes = base64.b64decode(data["image"])
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image from base64."}), 400

        model_type = data.get("model_type", "pp_layout_cdla")
        conf_thresh = float(data.get("conf_thresh", 0.5))
        iou_thresh = float(data.get("iou_thresh", 0.5))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", timestamp)
        os.makedirs(result_dir, exist_ok=True)
        filename = secure_filename(data.get("filename", "layout_markdown.md")) or "layout_markdown.md"
        if not filename.lower().endswith(".md"):
            filename = f"{filename}.md"
        output_md_path = os.path.join(result_dir, filename)

        result = get_layout_markdown_converter(model_type, conf_thresh, iou_thresh).convert_images(
            [img],
            output_md_path=output_md_path,
            source_name=os.path.splitext(os.path.basename(output_md_path))[0],
        )
        if data.get("visualize", False):
            result["visualization"] = image_to_base64(draw_layout_analysis(img, result["pages"][0]["layout"]))

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # 启动 Flask 服务
    app.run(host="0.0.0.0", port=5005, debug=False)
