# logic.py
import sys
import os
# Add parent directory to sys.path for onnxocr package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Img
import cv2
from typing import List, Callable
from pathlib import Path
import time
import numpy as np

from .logger import get_logger

log = get_logger("ocr_images_pdfs")

# Try to import pdf2image for PDF-to-image conversion
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

# Try to import pymupdf for PDF-to-image conversion
try:
    import fitz  # pymupdf
    def pdf_to_images(pdf_path, dpi=200):
        """Convert each PDF page to a numpy image array using pymupdf."""
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
    OCR business logic: batch image/PDF recognition, multi-threaded, model hot-swap.
    """
    def __init__(self, status_callback: Callable[[str], None]):
        """Initialize with a status callback for UI progress updates."""
        self.status_callback = status_callback
        # Initialize default OCR model
        log.info("Initializing OCRLogic with default model")
        self.model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)

    def run(self, files: List[str], save_txt: bool, merge_txt: bool, output_img: bool = False, file_time_callback=None, pdf_progress_callback=None, max_workers: int = 4):
        """
        Batch image/PDF recognition entry point with multi-threading.

        Args:
            files: List of file paths to recognize.
            save_txt: Whether to save results as txt.
            merge_txt: Whether to merge all results into one txt.
            output_img: Whether to output annotated images.
            file_time_callback: Callback for per-file processing time.
            pdf_progress_callback: Callback for PDF page progress.
            max_workers: Max thread count, default 4.
        """
        import concurrent.futures
        start_time = time.time()
        all_text = [None] * len(files)  # For ordered result merging
        def process_one(idx_file):
            idx, file = idx_file
            ext = os.path.splitext(file)[1].lower()
            self.status_callback(f"Processing: {os.path.basename(file)} ({idx+1}/{len(files)})")
            t0 = time.time()
            text = ""
            if ext == ".pdf":
                # Convert PDF to images then OCR
                if pdf_to_images is None:
                    raise RuntimeError("pymupdf is not installed, cannot process PDF files. Please install pymupdf first.")
                images = pdf_to_images(file, dpi=300)
                text = self._ocr_images(images, file, save_txt, merge_txt, output_img=output_img, is_pdf=True, pdf_progress_callback=pdf_progress_callback, max_workers=max_workers)
            else:
                # Regular image recognition, compatible with unicode paths
                try:
                    if file.lower().endswith('.bmp'):
                        img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
                    else:
                        with open(file, 'rb') as fimg:
                            img_array = np.frombuffer(fimg.read(), np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except Exception as e:
                    self.status_callback(f"Image read failed: {file}, error: {e}")
                    log.error("Image read failed: {}, error: {}", file, e)
                    if file_time_callback:
                        file_time_callback(idx, 0)
                    return (idx, "")
                if img is None:
                    self.status_callback(f"File is not a valid image: {file}")
                    log.warning("File is not a valid image: {}", file)
                    if file_time_callback:
                        file_time_callback(idx, 0)
                    return (idx, "")
                text = self._ocr_image(img, file, save_txt, output_img=output_img)
            t1 = time.time()
            if file_time_callback:
                file_time_callback(idx, t1-t0)
            self.status_callback(f"{os.path.basename(file)} done in {t1-t0:.2f}s")
            if len(files) > 1:
                avg = (t1 - start_time) / (idx + 1)
                self.status_callback(f"Completed {idx+1}/{len(files)}, avg {avg:.2f}s per file")
            return (idx, text)
        # Multi-threaded processing, results filled by index to preserve order
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one, (idx, file)) for idx, file in enumerate(files)]
            for future in concurrent.futures.as_completed(futures):
                idx, text = future.result()
                all_text[idx] = text
        # Merge and write txt
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
        log.info("Batch recognition done: {} files, total time: {:.2f}s", len(files), elapsed)
        if files:
            out_dir = self._get_output_dir(files[0])
            self.status_callback(f"Done in {elapsed:.2f}s, saved to: {out_dir}")
        else:
            self.status_callback(f"Done in {elapsed:.2f}s")

    def _ocr_images(self, images, pdf_path, save_txt, merge_txt, output_img=False, is_pdf=False, pdf_progress_callback=None, max_workers: int = 4):
        """
        Batch OCR on PDF page images with multi-threading.

        Args:
            images: PDF page images as numpy arrays.
            pdf_path: Original PDF path.
            save_txt: Whether to save txt.
            merge_txt: Whether to merge txt (unused).
            output_img: Whether to output annotated images.
            pdf_progress_callback: Page progress callback.
            max_workers: Max thread count, default 4.
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
        # Multi-threaded page recognition, merged by page order
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
        """OCR a single image, optionally saving txt and annotated image."""
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
        """Convert OCR result to plain text, handling detection-only results."""
        # Robustness check for empty or malformed results
        if not result or not isinstance(result, list) or not result[0] or not isinstance(result[0], list):
            return "[No content detected]"
        lines = []
        for box in result[0]:
            # Handle detection-only results without recognition text
            if isinstance(box, list) and len(box) == 2 and isinstance(box[1], (list, tuple)) and len(box[1]) >= 1:
                lines.append(str(box[1][0]))
            elif isinstance(box, list) and (isinstance(box[0], (list, tuple)) or isinstance(box[0], float)):
                # Detection box only, no recognition text
                lines.append("[Unrecognized] " + str(box))
            else:
                lines.append(str(box))
        return "\n".join(lines)

    def _get_output_dir(self, file_path):
        """Get output directory, creating it if needed."""
        base_dir = os.path.dirname(file_path)
        out_dir = os.path.join(base_dir, "Output_OCR")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def set_model(self, model_name, use_gpu=False):
        """
        Switch OCR model with hot-swap support. All models use ppocrv5 dictionary.

        Args:
            use_gpu: Whether to enable GPU inference.
        """
        import os
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
            use_gpu=use_gpu,
            det_model_dir=det_model_dir,
            cls_model_dir=cls_model_dir,
            rec_char_dict_path=rec_char_dict_path
        )
        if rec_model_dir and os.path.exists(rec_model_dir):
            ocr_kwargs["rec_model_dir"] = rec_model_dir
        try:
            self.model = ONNXPaddleOcr(**ocr_kwargs)
            log.info("Model switched successfully: {}", model_name)
            if use_gpu:
                try:
                    providers = self.model.session.get_providers() if hasattr(self.model, 'session') else []
                    if not any('CUDA' in p for p in providers):
                        msg = ("No GPU available, falling back to CPU. Check CUDA/cuDNN setup.")
                        if hasattr(self, 'ui_ref') and hasattr(self.ui_ref, 'update_gpu_status'):
                            self.ui_ref.update_gpu_status(msg)
                        if hasattr(self, 'status_callback'):
                            self.status_callback("[Warning] No GPU detected, switched to CPU. Check CUDA/cuDNN setup.")
                except Exception:
                    msg = ("GPU detection error, possibly missing CUDA/cuDNN or onnxruntime-gpu. Falling back to CPU.")
                    if hasattr(self, 'ui_ref') and hasattr(self.ui_ref, 'update_gpu_status'):
                        self.ui_ref.update_gpu_status(msg)
                    if hasattr(self, 'status_callback'):
                        self.status_callback("[Warning] GPU detection failed, switched to CPU. Check CUDA/cuDNN setup.")
        except Exception as e:
            log.error("Model switch failed: {}, error: {}", model_name, e)
            if use_gpu:
                msg = f"GPU init failed, falling back to CPU. Check CUDA/cuDNN setup. Error: {e}"
                if hasattr(self, 'ui_ref') and hasattr(self.ui_ref, 'update_gpu_status'):
                    self.ui_ref.update_gpu_status(msg)
                if hasattr(self, 'status_callback'):
                    self.status_callback("[Warning] GPU init failed, switched to CPU. Check CUDA/cuDNN setup.")
                ocr_kwargs["use_gpu"] = False
                self.model = ONNXPaddleOcr(**ocr_kwargs)
            else:
                raise
