# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import hashlib
import importlib
from pathlib import Path
from typing import List, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf


def format_ocr_results(
    ocr_results: Tuple[np.ndarray, Tuple[str], Tuple[float]], img_h: int, img_w: int
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    rec_res = list(zip(ocr_results[1], ocr_results[2]))

    bboxes = np.array(ocr_results[0])
    min_coords = bboxes[..., :2].min(axis=1)
    max_coords = bboxes[..., :2].max(axis=1)

    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, [img_w, img_h])
    dt_boxes = np.hstack([min_coords, max_coords])
    return dt_boxes, rec_res


def save_img(save_path: Union[str, Path], img: np.ndarray):
    cv2.imwrite(str(save_path), img)


def save_txt(save_path: Union[str, Path], txt: str):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(txt)


def import_package(name, package=None):
    try:
        module = importlib.import_module(name, package=package)
        return module
    except ModuleNotFoundError:
        return None


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


def is_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False
