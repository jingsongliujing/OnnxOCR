# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import abc
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..utils import EngineType, Logger, read_yaml

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
        from .onnxruntime import OrtInferSession

        return OrtInferSession

    raise ValueError(f"Unsupported engine: {engine_type.value}. OnnxOCR ships ONNXRuntime only.")
