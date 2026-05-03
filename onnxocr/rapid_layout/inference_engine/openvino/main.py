# -*- encoding: utf-8 -*-
import traceback
from pathlib import Path
from typing import Any, List

import numpy as np

try:
    from openvino import Core, Tensor
except ImportError:
    from openvino.runtime import Core, Tensor

from ...model_handler.utils import ModelProcessor
from ...utils.logger import logger
from ...utils.typings import RapidLayoutInput
from ..base import InferSession
from .device_config import OpenVINOConfig


class OpenVINOInferSession(InferSession):
    def __init__(self, cfg: RapidLayoutInput):
        if cfg.model_dir_or_path is None:
            model_path = ModelProcessor.get_model_path(cfg.model_type)
        else:
            model_path = Path(cfg.model_dir_or_path)

        self._verify_model(model_path)
        logger.info(f"Using {model_path}")

        core = Core()
        self.model = core.read_model(model=str(model_path))
        self.input_tensors = self.model.inputs
        self.output_tensors = self.model.outputs

        engine_cfg = self.update_params(
            self.engine_cfg[cfg.engine_type.value], cfg.engine_cfg
        )
        device = engine_cfg.get("device", "CPU")
        ov_config = OpenVINOConfig(engine_cfg)
        core.set_property(device, ov_config.get_config())
        self.compiled_model = core.compile_model(self.model, device_name=device)
        self.infer_request = self.compiled_model.create_infer_request()

    def __call__(self, input_content: np.ndarray) -> Any:
        if not isinstance(input_content, list):
            input_content = [input_content]

        if len(input_content) != len(self.input_tensors):
            raise OpenVINOError(
                f"The number of inputs ({len(input_content)}) does not match the number of model inputs ({len(self.input_tensors)})."
            )

        try:
            for input_tensor, input_content in zip(self.input_tensors, input_content):
                input_tensor_name = input_tensor.get_any_name()
                self.infer_request.set_tensor(input_tensor_name, Tensor(input_content))
            self.infer_request.infer()

            outputs = []
            for output_tensor in self.output_tensors:
                output_tensor_name = output_tensor.get_any_name()
                output = self.infer_request.get_tensor(output_tensor_name).data
                outputs.append(output)

            return outputs

        except Exception as e:
            error_info = traceback.format_exc()
            raise OpenVINOError(error_info) from e

    def get_input_names(self) -> List[str]:
        return [tensor.get_any_name() for tensor in self.model.inputs]

    def get_output_names(self) -> List[str]:
        return [tensor.get_any_name() for tensor in self.model.outputs]

    @property
    def characters(self):
        return self.get_character_list()

    def get_character_list(self, key: str = "character") -> List[str]:
        framework_info = self.get_rt_info_framework()
        if framework_info is None:
            return []

        val = framework_info[key] if key in framework_info else None
        if val is None or not hasattr(val, "value"):
            return []

        value = getattr(val, "value", None)
        if value is None:
            return []

        return value.splitlines()

    def have_key(self, key: str = "character") -> bool:
        try:
            framework_info = self.get_rt_info_framework()
            return framework_info is not None and key in framework_info
        except (AttributeError, TypeError, KeyError):
            return False

    def get_rt_info_framework(self):
        rt_info = self.model.get_rt_info()
        if "framework" not in rt_info:
            return None
        return rt_info["framework"]


class OpenVINOError(Exception):
    pass
