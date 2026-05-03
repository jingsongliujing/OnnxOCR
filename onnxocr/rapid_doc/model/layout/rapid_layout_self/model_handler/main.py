from typing import Any, List

import numpy as np

from ..inference_engine.base import InferSession
from ..utils.logger import Logger
from ..utils.typings import RapidLayoutInput, RapidLayoutOutput
from .pp_doclayout import PPDocLayoutModelHandler
from .doc_layout import DocLayoutModelHandler


class ModelHandler:
    def __init__(self, cfg: RapidLayoutInput, session: InferSession):
        self.logger = Logger(logger_name=__name__).get_log()
        self.model_processors = self._init_handler(cfg, session)

    def _init_handler(self, cfg: RapidLayoutInput, session: InferSession) -> Any:
        model_type = cfg.model_type.value
        self.logger.info(f"{model_type} contains {session.characters}")

        if model_type.startswith("pp_doc") or model_type.startswith("rt_detr"):
            return PPDocLayoutModelHandler(
                session.characters, cfg.conf_thresh, cfg.iou_thresh, session, cfg.model_type, cfg.layout_shape_mode
            )

        if model_type.startswith("doclayout"):
            return DocLayoutModelHandler(
                session.characters, cfg.conf_thresh, cfg.iou_thresh, session
            )

        raise ValueError(f"{model_type.value} is not supported!")

    def __call__(self, img_list: List[np.ndarray]) -> List[RapidLayoutOutput]:
        return self.model_processors(img_list)
