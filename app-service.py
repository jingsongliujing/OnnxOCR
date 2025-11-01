import cv2
import time
import base64
import numpy as np
from flask import Flask, request, jsonify,render_template, send_file
from onnxocr.onnx_paddleocr import ONNXPaddleOcr
from onnxocr.utils import draw_ocr
from io import BytesIO
import requests
import hashlib
import uuid
import re
import json
import os
from threading import RLock
from PIL import Image, ImageDraw, ImageFont

# 初始化 Flask 应用
app = Flask(__name__)

# 初始化 OCR 模型
model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)

# 简单内存缓存：按图片 URL 缓存最近一次 /url= 的结果（包含翻译）
TRANSLATION_CACHE = {}
CACHE_LOCK = RLock()


def _youdao_truncate(q):
    if q is None:
        return ''
    size = len(q)
    return q if size <= 20 else (q[:10] + str(size) + q[-10:])


def youdao_translate_text(app_id, app_key, text, from_lang='auto', to_lang='zh-CHS', timeout=10):
    if not text:
        return ''
    for _ in range(2):  # 简单重试一次
        try:
            salt = uuid.uuid4().hex
            curtime = str(int(time.time()))
            sign_str = app_id + _youdao_truncate(text) + salt + curtime + app_key
            sign = hashlib.sha256(sign_str.encode('utf-8')).hexdigest()
            payload = {
                'q': text,
                'from': from_lang,
                'to': to_lang,
                'appKey': app_id,
                'salt': salt,
                'sign': sign,
                'signType': 'v3',
                'curtime': curtime
            }
            r = requests.post('https://openapi.youdao.com/api', data=payload, timeout=timeout)
            r.raise_for_status()
            jd = r.json()
            if str(jd.get('errorCode')) != '0':
                continue
            trans = jd.get('translation')
            if isinstance(trans, list) and len(trans) > 0:
                return str(trans[0])
        except Exception:
            continue
    return ''


def deepseek_translate_mapping(api_key, system_prompt, text_mapping, timeout=30):
    try:
        if not isinstance(text_mapping, dict) or not text_mapping:
            return {}
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": json.dumps(text_mapping, ensure_ascii=False)}
            ],
            "temperature": 0.2,
            "max_tokens": 2048
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        r = requests.post("https://api.deepseek.com/chat/completions", headers=headers, data=json.dumps(payload), timeout=timeout)
        r.raise_for_status()
        jd = r.json()
        content = (
            jd.get("choices", [{}])[0]
              .get("message", {})
              .get("content", "")
        )
        if not content:
            return {}
        # 直接尝试解析为 JSON
        try:
            return json.loads(content)
        except Exception:
            pass
        # 从文本中提取第一个 JSON 对象
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}
    except Exception:
        return {}


def draw_ocr_without_conf(image_bgr, boxes, texts):
    # 绘制检测框
    img_out = image_bgr.copy()
    for box in boxes:
        pts = np.array(box).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img_out, [pts], True, (255, 0, 0), 2)

    # 右侧文本面板（PIL 绘制中文，不显示置信度）
    panel_width = 600
    panel_height = img_out.shape[0]
    panel_img = Image.new('RGB', (panel_width, panel_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel_img)

    # 字体路径（使用项目内 simfang.ttf）
    try:
        font_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'onnxocr', 'fonts', 'simfang.ttf'))
        font = ImageFont.truetype(font_path, 20, encoding='utf-8')
    except Exception:
        font = ImageFont.load_default()

    margin_left = 8
    margin_top = 12
    line_gap = 24

    def wrap_text(text, max_width):
        if not text:
            return ['']
        lines = []
        current = ''
        for ch in str(text):
            test = current + ch
            if draw.textlength(test, font=font) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                    current = ch
                else:
                    # 极窄面板兜底
                    lines.append(ch)
                    current = ''
        if current:
            lines.append(current)
        return lines

    y = margin_top
    index = 1
    max_text_width = panel_width - margin_left * 2
    for t in texts:
        prefix = f"{index}: "
        # 首行带序号，后续行缩进
        wrapped = wrap_text(t, max_text_width - draw.textlength(prefix, font=font))
        if not wrapped:
            wrapped = ['']
        # 画首行
        draw.text((margin_left, y), prefix + wrapped[0], fill=(0, 0, 0), font=font)
        y += line_gap
        # 画后续行
        for seg in wrapped[1:]:
            draw.text((margin_left + int(draw.textlength('    ', font=font)), y), seg, fill=(0, 0, 0), font=font)
            y += line_gap
        index += 1
        if y + line_gap >= panel_height:
            break

    # 拼接图像与面板
    panel_bgr = cv2.cvtColor(np.array(panel_img), cv2.COLOR_RGB2BGR)
    combined = np.concatenate([img_out, panel_bgr], axis=1)
    return combined


def draw_translated_overlay(image_bgr, boxes, texts):
    # 在检测框处覆盖背景并绘制译文（必要时扩展区域以完整显示）
    img = image_bgr.copy()

    # PIL 绘制中文
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'onnxocr', 'fonts', 'simfang.ttf'))
    except Exception:
        font_path = None

    def wrap_lines_for_font(text, font, max_w):
        if not text:
            return ['']
        lines = []
        current = ''
        for ch in str(text):
            test = current + ch
            if draw.textlength(test, font=font) <= max_w:
                current = test
            else:
                if current:
                    lines.append(current)
                    current = ch
                else:
                    lines.append(ch)
                    current = ''
        if current:
            lines.append(current)
        return lines

    for box, text in zip(boxes, texts):
        pts = np.array(box).reshape(-1, 1, 2).astype(np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        # 边界裁剪
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, image_bgr.shape[1] - x))
        h = max(1, min(h, image_bgr.shape[0] - y))
        if w < 5 or h < 12:
            continue
        # 自动选择字体大小与换行
        max_font = max(14, int(h * 0.9))
        best_font = None
        best_lines = None
        for size in range(max_font, 5, -1):
            try:
                font = ImageFont.truetype(font_path, size, encoding='utf-8') if font_path else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            lines = wrap_lines_for_font(text, font, w - 4)
            line_gap = int(size * 1.15)
            total_h = line_gap * len(lines)
            # 宽度验证
            max_line_w = 0
            for ln in lines:
                lw = draw.textlength(ln, font=font)
                if lw > max_line_w:
                    max_line_w = lw
            if total_h <= h and max_line_w <= (w - 4):
                best_font = font
                best_lines = lines
                break
        if best_font is None:
            # 即使最小字号仍放不下，则以最小字号绘制，并扩展覆盖区域高度至 total_h
            try:
                best_font = ImageFont.truetype(font_path, 6, encoding='utf-8') if font_path else ImageFont.load_default()
            except Exception:
                best_font = ImageFont.load_default()
            best_lines = wrap_lines_for_font(text, best_font, w - 4)
            line_gap = int(6 * 1.15)
        else:
            line_gap = int(best_font.size * 1.15)

        total_h = line_gap * len(best_lines)
        # 若需要，扩展背景覆盖区高度（不裁剪文本）
        target_h = max(h, total_h + 4)
        y2 = min(pil_img.size[1], y + target_h)
        # 绘制白色矩形背景（覆盖原文及扩展区）
        draw.rectangle((x, y, x + w, y2), fill=(255, 255, 255))

        ty = y + 2
        for ln in best_lines:
            if ty + line_gap > y2:
                break
            draw.text((x + 2, ty), ln, fill=(0, 0, 0), font=best_font)
            ty += line_gap

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out_bgr


def resolve_prompt_path(query_prompt: str | None) -> str | None:
    candidates = []
    if query_prompt:
        candidates.append(os.path.abspath(query_prompt))
    env1 = os.getenv('DEEPSEEK_PROMPT_PATH')
    env2 = os.getenv('PROMPT_PATH')
    for p in (env1, env2):
        if p:
            candidates.append(os.path.abspath(p))
    # 回退：项目根 prompt.txt（上一版位置）
    candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prompt.txt')))
    # 回退：当前目录同级 prompt.txt（如果被移动到 OnnxOCR/ 下）
    candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'prompt.txt')))
    for path in candidates:
        try:
            if path and os.path.exists(path):
                return path
        except Exception:
            continue
    return None

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


@app.route('/ocr_image', methods=['POST'])
def ocr_service_image():
    try:
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
        result = model.ocr(img)
        end_time = time.time()
        _ = end_time - start_time  # 保留处理时间变量以便后续扩展

        boxes = [line[0] for line in result[0]]
        txts = [line[1][0] for line in result[0]]
        scores = [float(line[1][1]) for line in result[0]]

        vis_rgb = draw_ocr(img[:, :, ::-1], boxes, txts, scores)
        vis_bgr = vis_rgb[:, :, ::-1]
        ok, buf = cv2.imencode('.png', vis_bgr)
        if not ok:
            return jsonify({"error": "Failed to encode result image."}), 500
        return send_file(BytesIO(buf.tobytes()), mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/ocr_url', methods=['POST'])
def ocr_service_url_json():
    try:
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "Invalid request, 'url' field is required."}), 400
        url = data["url"].strip()
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            image_bytes = resp.content
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Failed to decode image from URL."}), 400
        except Exception as e:
            return jsonify({"error": f"Fetching image failed: {str(e)}"}), 400

        start_time = time.time()
        result = model.ocr(img)
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
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/ocr_url_image', methods=['POST'])
def ocr_service_url_image():
    try:
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "Invalid request, 'url' field is required."}), 400
        url = data["url"].strip()
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            image_bytes = resp.content
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Failed to decode image from URL."}), 400
        except Exception as e:
            return jsonify({"error": f"Fetching image failed: {str(e)}"}), 400

        start_time = time.time()
        result = model.ocr(img)
        end_time = time.time()
        _ = end_time - start_time

        boxes = [line[0] for line in result[0]]
        txts = [line[1][0] for line in result[0]]
        scores = [float(line[1][1]) for line in result[0]]

        vis_rgb = draw_ocr(img[:, :, ::-1], boxes, txts, scores)
        vis_bgr = vis_rgb[:, :, ::-1]
        ok, buf = cv2.imencode('.png', vis_bgr)
        if not ok:
            return jsonify({"error": "Failed to encode result image."}), 500
        return send_file(BytesIO(buf.tobytes()), mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/url=<path:image_url>', methods=['GET'])
def ocr_service_get_url_json(image_url):
    try:
        try:
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            image_bytes = resp.content
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Failed to decode image from URL."}), 400
        except Exception as e:
            return jsonify({"error": f"Fetching image failed: {str(e)}"}), 400

        start_time = time.time()
        result = model.ocr(img)
        end_time = time.time()
        processing_time = end_time - start_time

        ocr_results = []
        for line in result[0]:
            if isinstance(line[0], (list, np.ndarray)):
                bounding_box = np.array(line[0]).reshape(4, 2).tolist()
            else:
                bounding_box = []
            text_val = line[1][0]
            conf_val = float(line[1][1])
            ocr_results.append({
                "text": text_val,
                "confidence": conf_val,
                "bounding_box": bounding_box
            })

        # DeepSeek 翻译：当 query 中提供 key 时启用（优先 DeepSeek，其次 Youdao 兼容）
        deepseek_key = request.args.get('key')
        if deepseek_key:
            try:
                prompt_arg = request.args.get('prompt')
                prompt_path = resolve_prompt_path(prompt_arg)
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    sys_prompt = f.read()
            except Exception:
                sys_prompt = ""

            unique_texts = []
            seen = set()
            for item in ocr_results:
                t = item.get('text', '')
                if t not in seen:
                    seen.add(t)
                    unique_texts.append(t)
            mapping = {t: "" for t in unique_texts}
            ds_result = deepseek_translate_mapping(deepseek_key, sys_prompt, mapping)
            for item in ocr_results:
                original = item.get('text', '')
                translated = ds_result.get(original) if isinstance(ds_result, dict) else None
                item['text_translated'] = translated if translated else original
            # 缓存结果供 /img= 使用
            try:
                with CACHE_LOCK:
                    TRANSLATION_CACHE[image_url] = {
                        'results': ocr_results,
                        'ts': time.time(),
                        'source': 'deepseek'
                    }
            except Exception:
                pass
        else:
            # 兼容旧的有道参数
            app_id = request.args.get('id')
            app_key = request.args.get('key')
            from_lang = request.args.get('from', 'auto')
            to_lang = request.args.get('to', 'zh-CHS')
            if app_id and app_key:
                for item in ocr_results:
                    original = item.get('text', '')
                    t = youdao_translate_text(app_id, app_key, original, from_lang, to_lang)
                    item['text_translated'] = t if t else original
                # 缓存有道结果（如果也想被 /img= 复用）
                try:
                    with CACHE_LOCK:
                        TRANSLATION_CACHE[image_url] = {
                            'results': ocr_results,
                            'ts': time.time(),
                            'source': 'youdao'
                        }
                except Exception:
                    pass

        return jsonify({
            "processing_time": processing_time,
            "results": ocr_results
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/img=<path:image_url>', methods=['GET'])
def ocr_service_get_url_image(image_url):
    try:
        try:
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            image_bytes = resp.content
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Failed to decode image from URL."}), 400
        except Exception as e:
            return jsonify({"error": f"Fetching image failed: {str(e)}"}), 400

        start_time = time.time()
        result = model.ocr(img)
        end_time = time.time()
        _ = end_time - start_time

        boxes = [line[0] for line in result[0]]
        txts = [line[1][0] for line in result[0]]
        scores = [float(line[1][1]) for line in result[0]]

        # DeepSeek 翻译：当 query 中提供 key 时启用（优先 DeepSeek，其次 Youdao 兼容）
        deepseek_key = request.args.get('key')
        used_deepseek = False
        if deepseek_key and len(txts) > 0:
            try:
                prompt_arg = request.args.get('prompt')
                prompt_path = resolve_prompt_path(prompt_arg)
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    sys_prompt = f.read()
            except Exception:
                sys_prompt = ""
            unique_texts = []
            seen = set()
            for t in txts:
                if t not in seen:
                    seen.add(t)
                    unique_texts.append(t)
            mapping = {t: "" for t in unique_texts}
            ds_result = deepseek_translate_mapping(deepseek_key, sys_prompt, mapping)
            new_txts = []
            for t in txts:
                translated = ds_result.get(t) if isinstance(ds_result, dict) else None
                new_txts.append(translated if translated else t)
            txts = new_txts
            used_deepseek = True
        else:
            app_id = request.args.get('id')
            app_key = request.args.get('key')
            from_lang = request.args.get('from', 'auto')
            to_lang = request.args.get('to', 'zh-CHS')
            if app_id and app_key and len(txts) > 0:
                new_txts = []
                for t in txts:
                    tt = youdao_translate_text(app_id, app_key, t, from_lang, to_lang)
                    new_txts.append(tt if tt else t)
                txts = new_txts

        # DeepSeek 图片：不显示置信度，只显示原文与译文
        if used_deepseek:
            merged_lines = []
            for orig, trans in zip([line[1][0] for line in result[0]], txts):
                merged_lines.append(trans if trans else orig)
            vis_bgr = draw_translated_overlay(img, boxes, merged_lines)
            ok, buf = cv2.imencode('.png', vis_bgr)
            if not ok:
                return jsonify({"error": "Failed to encode result image."}), 500
            return send_file(BytesIO(buf.tobytes()), mimetype='image/png')

        # 如果未带 key，则尝试从缓存复用翻译
        try:
            with CACHE_LOCK:
                cached = TRANSLATION_CACHE.get(image_url)
        except Exception:
            cached = None
        if cached and isinstance(cached.get('results'), list):
            # 使用缓存的 text_translated（若缺失则回退原文），并直接覆盖原图
            merged_lines = []
            map_trans = {}
            for it in cached['results']:
                src_txt = it.get('text')
                tgt_txt = it.get('text_translated') or src_txt
                if src_txt:
                    map_trans[src_txt] = tgt_txt
            for orig in [line[1][0] for line in result[0]]:
                trans = map_trans.get(orig, orig)
                merged_lines.append(trans if trans else orig)
            vis_bgr = draw_translated_overlay(img, boxes, merged_lines)
            ok, buf = cv2.imencode('.png', vis_bgr)
            if not ok:
                return jsonify({"error": "Failed to encode result image."}), 500
            return send_file(BytesIO(buf.tobytes()), mimetype='image/png')

        vis_rgb = draw_ocr(img[:, :, ::-1], boxes, txts, scores)
        vis_bgr = vis_rgb[:, :, ::-1]
        ok, buf = cv2.imencode('.png', vis_bgr)
        if not ok:
            return jsonify({"error": "Failed to encode result image."}), 500
        return send_file(BytesIO(buf.tobytes()), mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # 启动 Flask 服务
    app.run(host="0.0.0.0", port=5005, debug=False)
