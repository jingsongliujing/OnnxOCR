# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..utils.logger import Logger
from ..utils.typings import EngineType
from ..utils.utils import import_package

logger = Logger(logger_name=__name__).get_log()


class InferSession(ABC):
    cur_dir = Path(__file__).resolve().parent.parent
    MODEL_URL_PATH = cur_dir / "configs" / "default_models.yaml"
    ENGINE_CFG_PATH = cur_dir / "configs" / "engine_cfg.yaml"

    model_info = OmegaConf.load(MODEL_URL_PATH)
    DEFAULT_MODEL_PATH = cur_dir / "models"

    engine_cfg = OmegaConf.load(ENGINE_CFG_PATH)

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def _verify_model(model_path: Union[str, Path, None]):
        if model_path is None:
            raise ValueError("model_path is None!")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")

    @abstractmethod
    def have_key(self, key: str = "character") -> bool:
        pass

    @property
    def characters(self):
        return self.get_character_list()

    @abstractmethod
    def get_character_list(self, key: str = "character") -> List[str]:
        pass

    @staticmethod
    def update_params(cfg: DictConfig, params: Dict[str, Any]) -> DictConfig:
        for k, v in params.items():
            OmegaConf.update(cfg, k, v)
        return cfg


def get_engine(engine_type: EngineType):
    logger.info("Using engine_name: %s", engine_type.value)

    if engine_type == EngineType.ONNXRUNTIME:
        if not import_package(engine_type.value):
            raise ImportError(f"{engine_type.value} is not installed.")

        from .onnxruntime import OrtInferSession

        return OrtInferSession

    elif engine_type == EngineType.OPENVINO:
        if not import_package(engine_type.value):
            raise ImportError(f"{engine_type.value} is not installed.")

        from .openvino import OpenVINOInferSession

        return OpenVINOInferSession

    raise ValueError(f"Unsupported engine: {engine_type.value}")
