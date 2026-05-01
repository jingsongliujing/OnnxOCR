# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import abc
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..utils import EngineType, Logger, import_package, read_yaml

logger = Logger(logger_name=__name__).get_log()


class InferSession(abc.ABC):
    cur_dir = Path(__file__).resolve().parent.parent
    ENGINE_CFG_PATH = cur_dir / "engine_cfg.yaml"
    engine_cfg = read_yaml(ENGINE_CFG_PATH)

    @abc.abstractmethod
    def __init__(self, config):
        pass

    @abc.abstractmethod
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

    @abc.abstractmethod
    def have_key(self, key: str = "character") -> bool:
        pass

    @staticmethod
    def update_params(cfg: DictConfig, params: Dict[str, Any]):
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

    if engine_type == EngineType.TORCH:
        if not import_package(engine_type.value):
            raise ImportError(f"{engine_type.value} is not installed")

        from .torch import TorchInferSession

        return TorchInferSession

    raise ValueError(f"Unsupported engine: {engine_type.value}")
