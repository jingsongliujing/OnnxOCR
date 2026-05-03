from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .logger import Logger

logger = Logger(logger_name=__name__).get_log()


class ModelType(Enum):
    PP_FORMULANET_PLUS_S = "pp_formulanet_plus_s"
    PP_FORMULANET_PLUS_M = "pp_formulanet_plus_m"
    PP_FORMULANET_PLUS_L = "pp_formulanet_plus_l"


class EngineType(Enum):
    ONNXRUNTIME = "onnxruntime"
    # OPENVINO = "openvino"
    TORCH = "torch"


@dataclass
class RapidFormulaInput:
    model_type: ModelType = ModelType.PP_FORMULANET_PLUS_M
    model_dir_or_path: Union[str, Path, None] = None
    dict_keys_path: Union[str, Path, None] = None #yml字典路径（torch使用）

    engine_type: EngineType = EngineType.ONNXRUNTIME
    engine_cfg: dict = field(default_factory=dict)


@dataclass
class RapidFormulaOutput:
    img: Optional[np.ndarray] = None
    rec_formula: Optional[str] = None
    elapse: Optional[float] = None
