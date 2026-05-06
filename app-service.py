import time
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from onnxocr.api_utils import ModelRegistry, decode_base64_image, format_ocr_results
from onnxocr.visualization import (
    draw_layout_analysis,
    draw_plate_recognition,
    draw_table_recognition,
    image_to_base64,
)

app = Flask(__name__)
models = ModelRegistry(use_gpu=False)

RESULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_ROOT, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def ocr_service():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid request, 'image' field is required."}), 400

        try:
            img = decode_base64_image(data["image"])
        except (ValueError, Exception) as e:
            return jsonify({"error": f"Image decoding failed: {str(e)}"}), 400

        start_time = time.time()
        result = models.get_ocr_model().ocr(img)
        processing_time = time.time() - start_time

        return jsonify({
            "processing_time": processing_time,
            "results": format_ocr_results(result)
        })

    except Exception as e:
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
        img = decode_base64_image(data["image"])
        results = models.get_plate_model().ocr(img, plate_min_score=min_score)
        processing_time = time.time() - start_time

        response = {
            "processing_time": processing_time,
            "results": results
        }
        if data.get("visualize", False):
            response["visualization"] = image_to_base64(draw_plate_recognition(img, results))
        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/table', methods=['POST'])
def table_service():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid request, 'image' field is required."}), 400

        img = decode_base64_image(data["image"])
        start_time = time.time()
        result = models.get_table_model().ocr(img)
        result["processing_time"] = time.time() - start_time
        if data.get("visualize", False):
            result["visualization"] = image_to_base64(
                draw_table_recognition(img, result, show_logic=data.get("show_logic", False))
            )

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/layout', methods=['POST'])
def layout_service():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid request, 'image' field is required."}), 400

        img = decode_base64_image(data["image"])
        model_type = data.get("model_type", "pp_doclayoutv2")
        conf_thresh = float(data.get("conf_thresh", 0.4))
        iou_thresh = float(data.get("iou_thresh", 0.5))
        start_time = time.time()
        result = models.get_layout_model(model_type, conf_thresh, iou_thresh).ocr(img)
        result["processing_time"] = time.time() - start_time
        if data.get("visualize", False):
            result["visualization"] = image_to_base64(draw_layout_analysis(img, result))

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/layout_markdown', methods=['POST'])
def layout_markdown_service():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Invalid request, 'image' field is required."}), 400

        img = decode_base64_image(data["image"])
        model_type = data.get("model_type", "pp_layout_cdla")
        conf_thresh = float(data.get("conf_thresh", 0.5))
        iou_thresh = float(data.get("iou_thresh", 0.5))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(RESULT_ROOT, timestamp)
        os.makedirs(result_dir, exist_ok=True)
        filename = secure_filename(data.get("filename", "layout_markdown.md")) or "layout_markdown.md"
        if not filename.lower().endswith(".md"):
            filename = f"{filename}.md"
        output_md_path = os.path.join(result_dir, filename)

        result = models.get_layout_markdown_converter(model_type, conf_thresh, iou_thresh).convert_images(
            [img],
            output_md_path=output_md_path,
            source_name=os.path.splitext(os.path.basename(output_md_path))[0],
        )
        if data.get("visualize", False):
            result["visualization"] = image_to_base64(draw_layout_analysis(img, result["pages"][0]["layout"]))

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5005, debug=False)
