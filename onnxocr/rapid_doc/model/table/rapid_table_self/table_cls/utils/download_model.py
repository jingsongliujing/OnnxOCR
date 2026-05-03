import io
from pathlib import Path
from typing import Optional, Union

import requests
from tqdm import tqdm

from .logger import get_logger

logger = get_logger("DownloadModel")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "models"


class DownloadModel:
    @classmethod
    def download(
        cls,
        model_full_url: Union[str, Path],
        save_dir: Union[str, Path, None] = None,
        save_model_name: Optional[str] = None,
    ) -> str:
        if save_dir is None:
            save_dir = DEFAULT_MODEL_DIR

        save_dir.mkdir(parents=True, exist_ok=True)

        if save_model_name is None:
            save_model_name = Path(model_full_url).name

        save_file_path = save_dir / save_model_name
        if save_file_path.exists():
            # logger.debug("%s already exists", save_file_path)
            return str(save_file_path)

        try:
            logger.info("Download %s to %s", model_full_url, save_dir)
            file = cls.download_as_bytes_with_progress(model_full_url, save_model_name)
            cls.save_file(save_file_path, file)
        except Exception as exc:
            raise DownloadModelError from exc
        return str(save_file_path)

    @staticmethod
    def download_as_bytes_with_progress(
        url: Union[str, Path], name: Optional[str] = None
    ) -> bytes:
        resp = requests.get(str(url), stream=True, allow_redirects=True, timeout=180)
        total = int(resp.headers.get("content-length", 0))
        bio = io.BytesIO()
        with tqdm(
            desc=name, total=total, unit="b", unit_scale=True, unit_divisor=1024
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=65536):
                pbar.update(len(chunk))
                bio.write(chunk)
        return bio.getvalue()

    @staticmethod
    def save_file(save_path: Union[str, Path], file: bytes):
        with open(save_path, "wb") as f:
            f.write(file)


class DownloadModelError(Exception):
    pass
