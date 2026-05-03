import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def imread_unicode(path: str) -> Optional[np.ndarray]:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class LayoutMarkdownConverter:
    """RapidDoc based document-to-Markdown converter.

    The public methods keep the previous OnnxOCR wrapper API, but the implementation now
    delegates layout analysis, reading order, OCR, and table restoration to RapidDoc.
    """

    def __init__(
        self,
        layout_model_type: str = "pp_doclayoutv2",
        layout_model_path: Optional[str] = None,
        layout_engine_cfg: Optional[Dict] = None,
        layout_conf_thresh: float = 0.4,
        layout_iou_thresh: float = 0.5,
        table_model_type: str = "unet_slanet_plus",
        table_model_path: Optional[str] = None,
        table_engine_cfg: Optional[Dict] = None,
        ocr_kwargs: Optional[Dict] = None,
        formula_enable: bool = False,
        table_enable: bool = True,
        lang: str = "ch",
    ) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.model_dir = self.base_dir / "models"
        self.rapid_doc_model_dir = self.model_dir / "rapid_doc"
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.lang = lang

        os.environ.setdefault("MINERU_DEVICE_MODE", "cpu")
        os.environ.setdefault("USE_DOC_ORIENTATION_CLASSIFY", "false")

        self.layout_model_path = layout_model_path or str(
            self.rapid_doc_model_dir / "layout" / "pp_doclayoutv2.onnx"
        )
        self.table_paths = self._build_table_paths(table_model_path)
        self.ocr_kwargs = ocr_kwargs or {}

        self.engine = self._build_engine(
            layout_model_type=layout_model_type,
            layout_conf_thresh=layout_conf_thresh,
            layout_iou_thresh=layout_iou_thresh,
            layout_engine_cfg=layout_engine_cfg or {},
            table_model_type=table_model_type,
            table_engine_cfg=table_engine_cfg or {},
        )

    def convert_file(
        self,
        file_path: str,
        output_md_path: Optional[str] = None,
        assets_dir: Optional[str] = None,
        pdf_dpi: int = 220,
    ) -> Dict:
        del pdf_dpi
        suffix = Path(file_path).suffix.lower()
        if suffix not in IMAGE_SUFFIXES and suffix != ".pdf":
            raise ValueError(f"Unsupported file type: {suffix}. Supported: PDF and common image files.")

        source_name = Path(file_path).stem or "document"
        if output_md_path is None:
            out_dir = Path(file_path).parent / "Output_Markdown"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_md_path = str(out_dir / f"{source_name}.md")

        return self._convert_input(
            file_path,
            output_md_path=output_md_path,
            assets_dir=assets_dir,
            source_name=source_name,
        )

    def convert_images(
        self,
        images: List[np.ndarray],
        output_md_path: str,
        assets_dir: Optional[str] = None,
        source_name: str = "document",
    ) -> Dict:
        if not images:
            raise ValueError("images cannot be empty.")

        temp_dir = Path(output_md_path).parent / f".{Path(output_md_path).stem}_rapid_doc_input"
        temp_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for idx, img in enumerate(images, start=1):
            if img is None or not isinstance(img, np.ndarray):
                raise ValueError("images must contain decoded OpenCV images.")
            img_path = temp_dir / f"{source_name}_page_{idx:03d}.png"
            cv2.imencode(".png", img)[1].tofile(str(img_path))
            image_paths.append(str(img_path))

        try:
            if len(image_paths) == 1:
                return self._convert_input(
                    image_paths[0],
                    output_md_path=output_md_path,
                    assets_dir=assets_dir,
                    source_name=source_name,
                )

            page_results = []
            md_parts = [f"# {source_name}"]
            output_path = Path(output_md_path)
            assets_path = Path(assets_dir) if assets_dir else output_path.parent / f"{output_path.stem}_assets"
            assets_path.mkdir(parents=True, exist_ok=True)
            start = time.time()
            for idx, image_path in enumerate(image_paths, start=1):
                page_output = output_path.parent / f"{output_path.stem}_page_{idx:03d}.md"
                result = self._convert_input(
                    image_path,
                    output_md_path=str(page_output),
                    assets_dir=str(assets_path),
                    source_name=f"{source_name}_page_{idx:03d}",
                )
                page_results.append(result)
                md_parts.append(f"## Page {idx}")
                md_parts.append(result["markdown"])

            markdown = "\n\n".join(part for part in md_parts if part.strip()).strip() + "\n"
            output_path.write_text(markdown, encoding="utf-8")
            return {
                "markdown": markdown,
                "markdown_path": str(output_path),
                "assets_dir": str(assets_path),
                "pages": page_results,
                "processing_time": time.time() - start,
                "engine": "RapidDoc",
            }
        finally:
            # Keep the generated markdown/assets, but remove temporary page images.
            for image_path in image_paths:
                try:
                    Path(image_path).unlink()
                except OSError:
                    pass
            try:
                temp_dir.rmdir()
            except OSError:
                pass

    def _convert_input(
        self,
        input_path: str,
        output_md_path: str,
        assets_dir: Optional[str],
        source_name: str,
    ) -> Dict:
        start = time.time()
        output_path = Path(output_md_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        assets_path = Path(assets_dir) if assets_dir else output_path.parent / f"{output_path.stem}_assets"
        assets_path.mkdir(parents=True, exist_ok=True)

        result = self.engine(
            input_path,
            output_dir=None,
            image_dir_name=assets_path.name,
            image_output_mode="url",
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
            lang=self.lang,
            f_dump_middle_json=True,
            f_dump_content_list=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
        )

        for rel_path, data in result.images.items():
            rel = Path(rel_path.replace("\\", "/"))
            if rel.parts and rel.parts[0] == assets_path.name:
                rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(rel.name)
            save_path = assets_path / rel
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(data)

        output_path.write_text(result.markdown, encoding="utf-8")
        return {
            "markdown": result.markdown,
            "markdown_path": str(output_path),
            "assets_dir": str(assets_path),
            "pages": self._build_pages(result.middle_json),
            "content_list": result.content_list_json,
            "processing_time": time.time() - start,
            "engine": "RapidDoc",
        }

    def _build_engine(
        self,
        layout_model_type: str,
        layout_conf_thresh: float,
        layout_iou_thresh: float,
        layout_engine_cfg: Dict,
        table_model_type: str,
        table_engine_cfg: Dict,
    ):
        from onnxocr.rapid_doc import RapidDoc
        from onnxocr.rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
        from onnxocr.rapid_doc.model.table.rapid_table_self import EngineType as TableEngineType
        from onnxocr.rapid_doc.model.table.rapid_table_self import ModelType as TableModelType

        layout_type_map = {
            "pp_doclayoutv2": LayoutModelType.PP_DOCLAYOUTV2,
            "pp_doclayoutv3": LayoutModelType.PP_DOCLAYOUTV3,
            "pp_doclayout_plus_l": LayoutModelType.PP_DOCLAYOUT_PLUS_L,
            "pp_doclayout_l": LayoutModelType.PP_DOCLAYOUT_L,
            "pp_doclayout_m": LayoutModelType.PP_DOCLAYOUT_M,
            "pp_doclayout_s": LayoutModelType.PP_DOCLAYOUT_S,
        }
        table_type_map = {
            "unet_slanet_plus": TableModelType.UNET_SLANET_PLUS,
            "slanet_plus": TableModelType.SLANETPLUS,
            "unet": TableModelType.UNET,
            "q_cls": TableModelType.Q_CLS,
        }

        layout_config = {
            "model_type": layout_type_map.get(layout_model_type, LayoutModelType.PP_DOCLAYOUTV2),
            "model_dir_or_path": self.layout_model_path,
            "conf_thresh": layout_conf_thresh,
            "iou_thresh": layout_iou_thresh,
        }
        if layout_engine_cfg:
            layout_config["engine_cfg"] = layout_engine_cfg

        ocr_config = {
            "Det.model_path": str(self.model_dir / "ppocrv5" / "det" / "det.onnx"),
            "Rec.model_path": str(self.model_dir / "ppocrv5" / "rec" / "rec.onnx"),
            "Cls.model_path": str(self.model_dir / "ppocrv5" / "cls" / "cls.onnx"),
            "Rec.rec_keys_path": str(self.model_dir / "ppocrv5" / "ppocrv5_dict.txt"),
            "Global.use_cls": False,
            "use_det_mode": "ocr",
            "seal_enable": False,
        }
        for key, value in self.ocr_kwargs.items():
            if "." in key:
                ocr_config[key] = value

        table_config = {
            "model_type": table_type_map.get(table_model_type, TableModelType.UNET_SLANET_PLUS),
            "engine_type": TableEngineType.ONNXRUNTIME,
            "cls.model_type": TableModelType.Q_CLS,
            "cls.model_dir_or_path": self.table_paths["q_cls"],
            "unet.model_dir_or_path": self.table_paths["unet"],
            "slanet_plus.model_dir_or_path": self.table_paths["slanet_plus"],
            "use_word_box": False,
        }
        if table_model_type == "slanet_plus":
            table_config["model_dir_or_path"] = self.table_paths["slanet_plus"]
        if table_engine_cfg:
            table_config["engine_cfg"] = table_engine_cfg

        return RapidDoc(
            layout_config=layout_config,
            ocr_config=ocr_config,
            table_config=table_config,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
            lang=self.lang,
            pdf_pages_batch=1,
        )

    def _build_table_paths(self, table_model_path: Optional[str]) -> Dict[str, str]:
        table_dir = self.rapid_doc_model_dir / "table"
        return {
            "q_cls": str(table_dir / "q_cls.onnx"),
            "unet": str(table_dir / "unet.onnx"),
            "slanet_plus": table_model_path or str(table_dir / "slanet-plus.onnx"),
        }

    @staticmethod
    def _build_pages(middle_json) -> List[Dict]:
        if not middle_json:
            return []
        pages = []
        for idx, page in enumerate(middle_json.get("pdf_info", []), start=1):
            boxes, class_names, scores = [], [], []
            for item in page.get("layout_dets", []) or []:
                poly = item.get("poly") or []
                if len(poly) >= 8:
                    xs = poly[0::2]
                    ys = poly[1::2]
                    boxes.append([min(xs), min(ys), max(xs), max(ys)])
                    class_names.append(str(item.get("original_label", item.get("category_id", "layout"))))
                    scores.append(float(item.get("score", 0.0)))
            pages.append(
                {
                    "page_index": idx,
                    "layout": {
                        "boxes": boxes,
                        "class_names": class_names,
                        "scores": scores,
                    },
                    "raw_layout": page.get("layout_dets", []),
                    "spans": page.get("para_blocks", []),
                }
            )
        return pages
