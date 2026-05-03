import time
from typing import List

import numpy as np

from ...inference_engine.base import InferSession
from ...utils.typings import RapidFormulaOutput
from ..base import BaseModelHandler
from .post_process import PPPostProcess
from .pre_process import PPPreProcess


class PPFormulaNetPlusModelHandler(BaseModelHandler):
    def __init__(self, character_dict, session: InferSession, target_size):
        self.img_size = target_size
        self.pp_preprocess = PPPreProcess(img_size=self.img_size)
        self.pp_postprocess = PPPostProcess(character_dict)

        self.session = session

    def __call__(self, ori_img_list: List[np.ndarray]) -> List[RapidFormulaOutput]:
        s1 = time.perf_counter()
        # 1、前置处理
        img_inputs = self.preprocess(ori_img_list)
        img_inputs = np.concatenate(img_inputs, axis=0) #拼接batch
        # 2、推理
        batch_preds = self.session(img_inputs)
        # 3、后处理
        batch_preds = [p.reshape([-1]) for p in batch_preds[0]]
        rec_formula_list = self.pp_postprocess(batch_preds)
        result_list = []
        for i, rec_formula in enumerate(rec_formula_list):
            elapse = time.perf_counter() - s1
            result = RapidFormulaOutput(img=ori_img_list[i], rec_formula=rec_formula, elapse=elapse)
            result_list.append(result)
        return result_list

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return self.pp_preprocess(image)

    def postprocess(self, preds):
        return self.pp_postprocess(preds)
