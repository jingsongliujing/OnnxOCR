# logic.py
import sys
import os
# 添加父目录到sys.path，便于导入onnxocr包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Img
import cv2
from typing import List, Callable
from pathlib import Path
import time
import numpy as np

# 尝试导入pdf2image用于PDF转图片
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

# 尝试导入pymupdf用于PDF转图片
try:
    import fitz  # pymupdf
    def pdf_to_images(pdf_path, dpi=200):
        """
        使用pymupdf将PDF每一页转为图片（numpy数组）
        """
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            images.append(img)
        return images
except ImportError:
    pdf_to_images = None

class OCRLogic:
    """
    OCR 业务逻辑主类，支持批量图片/PDF识别，多线程加速，模型热切换等
    """
    def __init__(self, status_callback: Callable[[str], None]):
        """
        初始化，传入状态回调函数用于UI进度提示
        """
        self.status_callback = status_callback
        # 默认初始化OCR模型
        self.model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)

    def run(self, files: List[str], save_txt: bool, merge_txt: bool, output_img: bool = False, file_time_callback=None, pdf_progress_callback=None, max_workers: int = 4):
        """
        批量图片/PDF识别主入口，支持多线程加速
        files: 待识别文件路径列表
        save_txt: 是否保存txt
        merge_txt: 是否合并为一个txt
        output_img: 是否输出带框图片
        file_time_callback: 单文件识别耗时回调
        pdf_progress_callback: PDF页进度回调
        max_workers: 最大线程数，默认4
        """
        import concurrent.futures
        start_time = time.time()
        all_text = [None] * len(files)  # 用于顺序合并结果
        def process_one(idx_file):
            idx, file = idx_file
            ext = os.path.splitext(file)[1].lower()
            self.status_callback(f"正在处理: {os.path.basename(file)} ({idx+1}/{len(files)})")
            t0 = time.time()
            text = ""
            if ext == ".pdf":
                # PDF转图片后识别
                if pdf_to_images is None:
                    raise RuntimeError("未安装pymupdf库，无法处理PDF文件。请先安装pymupdf。")
                images = pdf_to_images(file, dpi=300)
                text = self._ocr_images(images, file, save_txt, merge_txt, output_img=output_img, is_pdf=True, pdf_progress_callback=pdf_progress_callback, max_workers=max_workers)
            else:
                # 普通图片识别，兼容中文路径
                try:
                    if file.lower().endswith('.bmp'):
                        img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
                    else:
                        with open(file, 'rb') as fimg:
                            img_array = np.frombuffer(fimg.read(), np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except Exception as e:
                    self.status_callback(f"图片读取失败: {file}，错误: {e}")
                    if file_time_callback:
                        file_time_callback(idx, 0)
                    return (idx, "")
                if img is None:
                    self.status_callback(f"文件无法读取或不是有效图片: {file}")
                    if file_time_callback:
                        file_time_callback(idx, 0)
                    return (idx, "")
                text = self._ocr_image(img, file, save_txt, output_img=output_img)
            t1 = time.time()
            if file_time_callback:
                file_time_callback(idx, t1-t0)
            self.status_callback(f"{os.path.basename(file)} 识别用时: {t1-t0:.2f} 秒")
            if len(files) > 1:
                avg = (t1 - start_time) / (idx + 1)
                self.status_callback(f"已完成 {idx+1}/{len(files)}，平均单张用时: {avg:.2f} 秒")
            return (idx, text)
        # 多线程处理所有文件，结果按索引回填，保证顺序
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one, (idx, file)) for idx, file in enumerate(files)]
            for future in concurrent.futures.as_completed(futures):
                idx, text = future.result()
                all_text[idx] = text
        # 合并写入txt
        if save_txt and merge_txt and len(files) > 1:
            out_dir = self._get_output_dir(files[0])
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_txt = os.path.join(out_dir, f"merged_ocr_{timestamp}.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                for text in all_text:
                    if text:
                        f.write(text)
                        f.write("\n\n")
        elapsed = time.time() - start_time
        if files:
            out_dir = self._get_output_dir(files[0])
            self.status_callback(f"识别完成，总耗时：{elapsed:.2f}秒，文件保存在：{out_dir}")
        else:
            self.status_callback(f"识别完成，总耗时：{elapsed:.2f}秒")

    def _ocr_images(self, images, pdf_path, save_txt, merge_txt, output_img=False, is_pdf=False, pdf_progress_callback=None, max_workers: int = 4):
        """
        PDF转图片后，批量图片识别，支持多线程加速
        images: PDF每页图片（numpy数组）
        pdf_path: 原PDF路径
        save_txt: 是否保存txt
        merge_txt: 是否合并txt（未用）
        output_img: 是否输出带框图片
        pdf_progress_callback: 页进度回调
        max_workers: 最大线程数，默认4
        """
        import concurrent.futures
        out_dir = self._get_output_dir(pdf_path)
        pdf_text = [None] * len(images)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        total = len(images)
        def process_page(i_img):
            i, img = i_img
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            result = self.model.ocr(img_cv)
            if output_img:
                out_img_path = os.path.join(out_dir, f"{Path(pdf_path).stem}_page{i+1}_ocr.jpg")
                sav2Img(img_cv, result, name=out_img_path)
            page_text = self._result_to_text(result)
            return (i, page_text)
        # 多线程识别每一页，结果按页码顺序合并
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_page, (i, img)) for i, img in enumerate(images)]
            for future in concurrent.futures.as_completed(futures):
                i, page_text = future.result()
                pdf_text[i] = page_text
                if pdf_progress_callback:
                    pdf_progress_callback(i + 1, total)
        if save_txt:
            txt_path = os.path.join(out_dir, f"{Path(pdf_path).stem}_ocr_{timestamp}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(pdf_text))
        return "\n\n".join(pdf_text)

    def _ocr_image(self, img, img_path, save_txt, output_img=False):
        """
        单张图片OCR识别，支持保存txt和输出带框图片
        """
        out_dir = self._get_output_dir(img_path)
        result = self.model.ocr(img)
        if output_img:
            out_img_path = os.path.join(out_dir, f"{Path(img_path).stem}_ocr.jpg")
            sav2Img(img, result, name=out_img_path)
        text = self._result_to_text(result)
        if save_txt:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            txt_path = os.path.join(out_dir, f"{Path(img_path).stem}_ocr_{timestamp}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def _result_to_text(self, result):
        """
        将OCR识别结果结构化为纯文本，兼容只检测无识别内容的情况
        """
        # 健壮性检查，防止result为空或结构异常
        if not result or not isinstance(result, list) or not result[0] or not isinstance(result[0], list):
            return "[未检测到内容]"
        lines = []
        for box in result[0]:
            # 兼容只检测无识别内容的情况
            if isinstance(box, list) and len(box) == 2 and isinstance(box[1], (list, tuple)) and len(box[1]) >= 1:
                lines.append(str(box[1][0]))
            elif isinstance(box, list) and (isinstance(box[0], (list, tuple)) or isinstance(box[0], float)):
                # 只有检测框，无识别内容
                lines.append("[未识别] " + str(box))
            else:
                lines.append(str(box))
        return "\n".join(lines)

    def _get_output_dir(self, file_path):
        """
        获取输出目录，自动创建
        """
        base_dir = os.path.dirname(file_path)
        out_dir = os.path.join(base_dir, "Output_OCR")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def set_model(self, model_name, use_gpu=False):
        """
        切换OCR模型，支持多模型热切换，所有模型统一用ppocrv5字典
        use_gpu: 是否启用GPU
        """
        import os
        import tkinter.messagebox as messagebox
        base_model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "onnxocr", "models"))
        model_map = {
            "PP-OCRv5": "ppocrv5",
            "PP-OCRv4": "ppocrv4",
            "ch_ppocr_server_v2.0": "ch_ppocr_server_v2.0"
        }
        model_dir = model_map.get(model_name, "ppocrv5")
        model_path = os.path.join(base_model_dir, model_dir)
        det_model_dir = os.path.join(model_path, "det", "det.onnx")
        cls_model_dir = os.path.join(model_path, "cls", "cls.onnx")
        rec_char_dict_path = os.path.join(base_model_dir, "ppocrv5", "ppocrv5_dict.txt")
        rec_model_dir = os.path.join(model_path, "rec", "rec.onnx") if os.path.exists(os.path.join(model_path, "rec", "rec.onnx")) else None
        ocr_kwargs = dict(
            use_angle_cls=True,
            use_gpu=use_gpu,  # 关键：传递GPU参数
            det_model_dir=det_model_dir,
            cls_model_dir=cls_model_dir,
            rec_char_dict_path=rec_char_dict_path
        )
        if rec_model_dir and os.path.exists(rec_model_dir):
            ocr_kwargs["rec_model_dir"] = rec_model_dir
        try:
            self.model = ONNXPaddleOcr(**ocr_kwargs)
            if use_gpu:
                try:
                    import onnxruntime as ort
                    providers = self.model.session.get_providers() if hasattr(self.model, 'session') else []
                    if not any('CUDA' in p for p in providers):
                        msg = ("未检测到可用GPU，已自动切换为CPU推理。请检查CUDA/cuDNN环境配置。")
                        if hasattr(self, 'ui_ref') and hasattr(self.ui_ref, 'update_gpu_status'):
                            self.ui_ref.update_gpu_status(msg)
                        if hasattr(self, 'status_callback'):
                            self.status_callback("[警告] 未检测到可用GPU，已切换为CPU推理。请检查CUDA/cuDNN环境配置。")
                except Exception:
                    msg = ("检测GPU状态时发生异常，可能未正确安装CUDA/cuDNN或onnxruntime-gpu。已自动切换为CPU推理。")
                    if hasattr(self, 'ui_ref') and hasattr(self.ui_ref, 'update_gpu_status'):
                        self.ui_ref.update_gpu_status(msg)
                    if hasattr(self, 'status_callback'):
                        self.status_callback("[警告] GPU检测异常，已切换为CPU推理。请检查CUDA/cuDNN环境配置。")
        except Exception as e:
            if use_gpu:
                msg = f"GPU初始化失败，已自动切换为CPU。请检查CUDA/cuDNN环境配置。错误信息: {e}"
                if hasattr(self, 'ui_ref') and hasattr(self.ui_ref, 'update_gpu_status'):
                    self.ui_ref.update_gpu_status(msg)
                if hasattr(self, 'status_callback'):
                    self.status_callback("[警告] GPU初始化失败，已切换为CPU推理。请检查CUDA/cuDNN环境配置。")
                ocr_kwargs["use_gpu"] = False
                self.model = ONNXPaddleOcr(**ocr_kwargs)
            else:
                raise
