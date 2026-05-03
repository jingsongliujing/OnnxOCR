import os
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import requests
from tqdm import tqdm
from loguru import logger


@dataclass
class DownloadFileInput:
    file_url: str
    save_path: Union[str, Path]
    sha256: Optional[str] = None

CPU_MODEL = [
    # layout
    "pp_doclayoutv2.onnx",
    # formula
    "pp_formulanet_plus_m.onnx",
    # ocr
    "ch_PP-OCRv5_rec_mobile_infer.onnx",
    "ch_PP-OCRv5_mobile_det.onnx",
    "ch_PP-OCRv4_rec_infer.onnx",
    "ch_ppocr_mobile_v2.0_cls_infer.onnx",
    "ppocrv5_dict.txt",
    "FZYTK.TTF",
    # table
    "paddle_cls.onnx",
    "q_cls.onnx",
    "unet.onnx",
    "slanet-plus.onnx",
]

GPU_MODEL = [
    # layout
    "pp_doclayoutv2.onnx",
    # formula
    "pp_formulanet_plus_m.pth",
    "pp_formulanet_plus_m_inference.yml"
    # ocr
    "ch_PP-OCRv5_det_mobile_infer.pth",
    "ch_PP-OCRv5_rec_mobile_infer.pth",
    "ch_ptocr_mobile_v2.0_cls_infer.pth",
    "ppocrv5_dict.txt",
    "FZYTK.TTF",
    # table
    "paddle_cls.onnx",
    "q_cls.onnx",
    "unet.onnx",
    "slanet-plus.onnx",
]

NPU_MODEL = [
    # layout
    "pp_doclayoutv2.onnx",
    "doclayout_yolo_docstructbench_imgsz1024.onnx",
    # formula
    "pp_formulanet_plus_m.pth",
    "pp_formulanet_plus_m_inference.yml"
    # ocr
    "ch_PP-OCRv5_det_mobile_infer.pth",
    "ch_PP-OCRv5_rec_mobile_infer.pth",
    "ch_ptocr_mobile_v2.0_cls_infer.pth",
    "ppocrv5_dict.txt",
    "FZYTK.TTF",
    # table
    "paddle_cls.onnx",
    "q_cls.onnx",
    "unet.onnx",
    "slanet-plus.onnx",
]

device_mode = os.getenv("MINERU_DEVICE_MODE", "cpu")

class DownloadFile:
    BLOCK_SIZE = 1024  # 1 KiB
    REQUEST_TIMEOUT = 60

    @classmethod
    def run(cls, input_params: DownloadFileInput):
        if device_mode.startswith('cuda'):
            default_model = GPU_MODEL
        elif device_mode == 'npu':
            default_model = NPU_MODEL
        elif device_mode == 'all':
            default_model = set(GPU_MODEL + CPU_MODEL)
        else:
            default_model = CPU_MODEL
        if not any(k in input_params.file_url for k in default_model):
            return

        save_path = Path(input_params.save_path)

        cls._ensure_parent_dir_exists(save_path)
        if cls._should_skip_download(save_path, input_params.sha256):
            return

        response = cls._make_http_request(input_params.file_url)
        cls._save_response_with_progress(response, save_path)

    @staticmethod
    def _ensure_parent_dir_exists(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _should_skip_download(
        cls, path: Path, expected_sha256: Optional[str]
    ) -> bool:
        if not path.exists():
            return False

        if expected_sha256 is None:
            logger.info("File exists (no checksum verification): {}", path)
            return True

        if cls.check_file_sha256(path, expected_sha256):
            logger.info("File exists and is valid: {}", path)
            return True

        logger.warning("File exists but is invalid, redownloading: {}", path)
        return False

    @classmethod
    def _make_http_request(cls, url: str) -> requests.Response:
        logger.info("Initiating download: {}", url)
        try:
            response = requests.get(url, stream=True, timeout=cls.REQUEST_TIMEOUT)
            response.raise_for_status()  # Raises HTTPError for 4XX/5XX
            return response
        except requests.RequestException as e:
            logger.error("Download failed: {}", url)
            raise DownloadFileException(f"Failed to download {url}") from e

    @classmethod
    def _save_response_with_progress(
        cls, response: requests.Response, save_path: Path
    ) -> None:
        total_size = int(response.headers.get("content-length", 0))
        logger.info("Download size: {:.2f}MB", total_size / 1024 / 1024)

        with tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            disable=not cls.check_is_atty(),
        ) as progress_bar:
            with open(save_path, "wb") as output_file:
                for chunk in response.iter_content(chunk_size=cls.BLOCK_SIZE):
                    progress_bar.update(len(chunk))
                    output_file.write(chunk)

        logger.info("Successfully saved to: {}", save_path)

    @staticmethod
    def check_file_sha256(file_path: Union[str, Path], gt_sha256: str) -> bool:
        return get_file_sha256(file_path) == gt_sha256

    @staticmethod
    def check_is_atty() -> bool:
        try:
            is_interactive = sys.stderr.isatty()
        except AttributeError:
            return False
        return is_interactive


class DownloadFileException(Exception):
    pass


def get_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    with open(file_path, "rb") as file:
        sha_signature = hashlib.sha256()
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            sha_signature.update(chunk)

    return sha_signature.hexdigest()
