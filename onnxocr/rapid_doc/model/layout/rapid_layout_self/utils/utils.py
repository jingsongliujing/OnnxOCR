import hashlib
import importlib
from pathlib import Path
from typing import Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf


def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def read_yaml(file_path: Union[str, Path]) -> DictConfig:
    return OmegaConf.load(file_path)


def get_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    with open(file_path, "rb") as file:
        sha_signature = hashlib.sha256()
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            sha_signature.update(chunk)

    return sha_signature.hexdigest()


def save_img(save_path: Union[str, Path], img: np.ndarray):
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(save_path), img)


def is_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def import_package(name, package=None):
    try:
        module = importlib.import_module(name, package=package)
        return module
    except ModuleNotFoundError:
        return None
