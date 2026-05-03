import os
import traceback
from pathlib import Path
from typing import Any, List

import numpy as np
from omegaconf import DictConfig
from onnxocr.inference_engine import GraphOptimizationLevel, InferenceSession, SessionOptions

from ...model_handler.utils import ModelProcessor
from ...utils.logger import Logger
from ...utils.typings import RapidLayoutInput
from ..base import InferSession
from .provider_config import ProviderConfig


class OrtInferSession(InferSession):
    def __init__(self, cfg: RapidLayoutInput):
        self.logger = Logger(logger_name=__name__).get_log()

        if cfg.model_dir_or_path is None:
            model_path = ModelProcessor.get_model_path(cfg.model_type)
        else:
            model_path = Path(cfg.model_dir_or_path)

        self._verify_model(model_path)
        self.logger.info(f"Using {model_path}")

        engine_cfg = self.update_params(
            self.engine_cfg[cfg.engine_type.value], cfg.engine_cfg
        )

        sess_opt = self._init_sess_opts(engine_cfg)

        provider_cfg = ProviderConfig(engine_cfg=engine_cfg)
        self.session = InferenceSession(
            model_path,
            sess_options=sess_opt,
            providers=provider_cfg.get_ep_list(),
        )
        provider_cfg.verify_providers(self.session.get_providers())

    @staticmethod
    def _init_sess_opts(cfg: DictConfig) -> SessionOptions:
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = cfg.enable_cpu_mem_arena
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cpu_nums = os.cpu_count()
        intra_op_num_threads = cfg.get("intra_op_num_threads", -1)
        if intra_op_num_threads != -1 and 1 <= intra_op_num_threads <= cpu_nums:
            sess_opt.intra_op_num_threads = intra_op_num_threads

        inter_op_num_threads = cfg.get("inter_op_num_threads", -1)
        if inter_op_num_threads != -1 and 1 <= inter_op_num_threads <= cpu_nums:
            sess_opt.inter_op_num_threads = inter_op_num_threads

        return sess_opt

    def __call__(self, input_content: np.ndarray, scale_factor: np.ndarray = None) -> Any:
        if scale_factor is not None:
            input_names = self.get_input_names()
            input_dict = {}
            if "image" in input_names:
                input_dict["image"] = input_content
            if "scale_factor" in input_names:
                input_dict["scale_factor"] = scale_factor
            if "im_shape" in input_names:
                h, w = input_content.shape[-2:]
                input_dict["im_shape"] = np.array([[h, w]], dtype=np.float32)
        else:
            input_dict = dict(zip(self.get_input_names(), [input_content]))
        try:
            return self.session.run(self.get_output_names(), input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(self) -> List[str]:
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        return [v.name for v in self.session.get_outputs()]

    @property
    def characters(self):
        return self.get_character_list()

    def get_character_list(self, key: str = "character") -> List[str]:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in meta_dict.keys():
            return True
        return False


class ONNXRuntimeError(Exception):
    pass
