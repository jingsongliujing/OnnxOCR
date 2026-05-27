from __future__ import annotations

import importlib
import io
import os
import re
import sys
import types
import zipfile


class FakeOCRLogic:
    def __init__(self, status_callback):
        self.status_callback = status_callback
        self.model_name = None

    def set_model(self, model_name):
        self.model_name = model_name

    def run(self, file_paths, save_txt, merge_txt, output_img=False):
        for file_path in file_paths:
            output_dir = os.path.join(os.path.dirname(file_path), "Output_OCR")
            os.makedirs(output_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{stem}.txt")
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write("fake ocr text")


class FakeModelRegistry:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu


def load_webui(monkeypatch, tmp_path):
    fake_ocr_module = types.ModuleType("onnxocr.ocr_images_pdfs")
    setattr(fake_ocr_module, "OCRLogic", FakeOCRLogic)

    fake_api_module = types.ModuleType("onnxocr.api_utils")
    setattr(fake_api_module, "ModelRegistry", FakeModelRegistry)
    setattr(fake_api_module, "decode_base64_image", lambda image: image)
    setattr(fake_api_module, "format_ocr_results", lambda result: result)

    fake_visualization_module = types.ModuleType("onnxocr.visualization")
    setattr(fake_visualization_module, "draw_layout_analysis", lambda *args, **kwargs: None)
    setattr(fake_visualization_module, "draw_plate_recognition", lambda *args, **kwargs: None)
    setattr(fake_visualization_module, "draw_table_recognition", lambda *args, **kwargs: None)
    setattr(fake_visualization_module, "image_to_base64", lambda image: "")

    monkeypatch.setitem(sys.modules, "onnxocr.ocr_images_pdfs", fake_ocr_module)
    monkeypatch.setitem(sys.modules, "onnxocr.api_utils", fake_api_module)
    monkeypatch.setitem(sys.modules, "onnxocr.visualization", fake_visualization_module)
    sys.modules.pop("webui", None)

    webui = importlib.import_module("webui")
    monkeypatch.setattr(webui, "RESULT_ROOT", str(tmp_path))
    return webui


def test_ocr_download_url_uses_unguessable_result_id(monkeypatch, tmp_path):
    webui = load_webui(monkeypatch, tmp_path)
    client = webui.app.test_client()

    response = client.post(
        "/ocr",
        data={
            "model_name": "PP-OCRv5",
            "files": (io.BytesIO(b"not a real image"), "sample.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    result_id = payload["zip_url"].rsplit("/", 1)[1]
    assert re.fullmatch(r"[A-Za-z0-9_-]{22,}", result_id)
    assert not re.fullmatch(r"\d{8}_\d{6}", result_id)

    download_response = client.get(payload["zip_url"])
    assert download_response.status_code == 200
    assert f"ocr_txt_{result_id}.zip" in download_response.headers["Content-Disposition"]


def test_predictable_timestamp_download_ids_are_rejected(monkeypatch, tmp_path):
    webui = load_webui(monkeypatch, tmp_path)
    client = webui.app.test_client()
    timestamp = "20260101_010203"
    legacy_dir = tmp_path / timestamp
    legacy_dir.mkdir()
    with zipfile.ZipFile(legacy_dir / f"ocr_txt_{timestamp}.zip", "w") as zip_file:
        zip_file.writestr("private.txt", "private result")

    response = client.get(f"/download/{timestamp}")

    assert response.status_code == 404


def test_layout_markdown_download_rejects_predictable_timestamp_ids(monkeypatch, tmp_path):
    webui = load_webui(monkeypatch, tmp_path)
    client = webui.app.test_client()
    timestamp = "20260101_010203"
    legacy_dir = tmp_path / timestamp
    legacy_dir.mkdir()
    with zipfile.ZipFile(
        legacy_dir / f"layout_markdown_{timestamp}.zip", "w"
    ) as zip_file:
        zip_file.writestr("private.md", "# private result")

    response = client.get(f"/download_layout_markdown/{timestamp}")

    assert response.status_code == 404
