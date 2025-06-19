import os
import time
import zipfile
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from onnxocr.ocr_images_pdfs import OCRLogic
import cv2
import base64
import numpy as np
from onnxocr.onnx_paddleocr import ONNXPaddleOcr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, "uploads")
RESULT_ROOT = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)

MODEL_OPTIONS = ["PP-OCRv5", "PP-OCRv4", "ch_ppocr_server_v2.0"]

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

ocr_logic = OCRLogic(lambda msg: print(msg))
# 独立 OCR 模型实例，避免影响 ocr_logic
ocr_model_api = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)

@app.route("/")
def index():
    return render_template("webui.html", model_options=MODEL_OPTIONS)

@app.errorhandler(404)
def not_found(e):
    path = request.path
    if not path.startswith("/static") and not path.startswith("/download"):
        return redirect(url_for("index"))
    return jsonify({"detail": "NotFound"}), 404

@app.route("/set_model", methods=["POST"])
def set_model():
    model_name = request.form.get("model_name")
    try:
        ocr_logic.set_model(model_name)
        return {"success": True, "msg": f"模型已切换为 {model_name}"}
    except Exception as e:
        return {"success": False, "msg": str(e)}

@app.route("/ocr", methods=["POST"])
def ocr_files():
    files = request.files.getlist("files")
    model_name = request.form.get("model_name")
    if not files or not model_name:
        return jsonify({"success": False, "msg": "缺少文件或模型参数"}), 400
    try:
        ocr_logic.set_model(model_name)
    except Exception as e:
        return jsonify({"success": False, "msg": f"模型切换失败: {e}"}), 500
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(RESULT_ROOT, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    file_paths = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(session_dir, filename)
        file.save(file_path)
        file_paths.append(file_path)
    results = []
    def status_callback(msg): pass
    logic = OCRLogic(status_callback)
    logic.set_model(model_name)
    logic.run(file_paths, save_txt=True, merge_txt=False, output_img=False)
    txt_files = []
    for file_path in file_paths:
        out_dir = os.path.join(os.path.dirname(file_path), "Output_OCR")
        if not os.path.exists(out_dir):
            continue
        for fname in os.listdir(out_dir):
            if fname.endswith(".txt") and fname.startswith(os.path.splitext(os.path.basename(file_path))[0]):
                txt_files.append(os.path.join(out_dir, fname))
                with open(os.path.join(out_dir, fname), "r", encoding="utf-8") as f:
                    content = f.read()
                results.append({"filename": fname, "content": content})
    zip_path = os.path.join(session_dir, f"ocr_txt_{timestamp}.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for txt_file in txt_files:
            zipf.write(txt_file, os.path.basename(txt_file))
    return jsonify({
        "success": True,
        "results": results,
        "zip_url": f"/download/{timestamp}"
    })

@app.route("/download/<timestamp>")
def download_zip(timestamp):
    session_dir = os.path.join(RESULT_ROOT, timestamp)
    zip_path = os.path.join(session_dir, f"ocr_txt_{timestamp}.zip")
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True, download_name=f"ocr_txt_{timestamp}.zip")
    return jsonify({"success": False, "msg": "文件不存在"}), 404

@app.route("/ocr_api", methods=["POST"])
def ocr_api():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Invalid request, 'image' field is required."}), 400
    image_base64 = data["image"]
    try:
        image_bytes = base64.b64decode(image_base64)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image from base64."}), 400
    except Exception as e:
        return jsonify({"error": f"Image decoding failed: {str(e)}"}), 400
    start_time = time.time()
    result = ocr_model_api.ocr(img)
    end_time = time.time()
    processing_time = end_time - start_time
    ocr_results = []
    for line in result[0]:
        if isinstance(line[0], (list, np.ndarray)):
            bounding_box = np.array(line[0]).reshape(4, 2).tolist()
        else:
            bounding_box = []
        ocr_results.append({
            "text": line[1][0],
            "confidence": float(line[1][1]),
            "bounding_box": bounding_box
        })
    return jsonify({
        "processing_time": processing_time,
        "results": ocr_results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
