from typing import Any, List

import numpy as np

from ..inference_engine.base import InferSession
from ..utils.logger import Logger
from ..utils.typings import RapidFormulaInput, RapidFormulaOutput
from .pp_formulanet_plus import PPFormulaNetPlusModelHandler


class ModelHandler:
    def __init__(self, cfg: RapidFormulaInput, session: InferSession):
        self.logger = Logger(logger_name=__name__).get_log()
        self.model_processors = self._init_handler(cfg, session)

    def _init_handler(self, cfg: RapidFormulaInput, session: InferSession) -> Any:
        model_type = cfg.model_type.value
        # self.logger.info(f"{model_type} contains {session.characters}")

        if model_type.startswith("pp_formulanet"):
            if model_type.endswith("_l"):
                target_size = (768, 768)
            else:
                target_size = (384, 384)
            return PPFormulaNetPlusModelHandler(
                session.characters, session, target_size
            )

        raise ValueError(f"{model_type.value} is not supported!")

    def __call__(self, img_list: List[np.ndarray]) -> List[RapidFormulaOutput]:
        return self.model_processors(img_list)
