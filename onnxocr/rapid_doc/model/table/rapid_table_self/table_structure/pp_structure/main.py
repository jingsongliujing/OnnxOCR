# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, Tuple

import numpy as np

from ...inference_engine.base import get_engine
from ...utils.typings import EngineType
from .post_process import TableLabelDecode
from .pre_process import TablePreprocess


class PPTableStructurer:
    def __init__(self, cfg: Dict[str, Any]):
        if cfg["engine_type"] is None:
            cfg["engine_type"] = EngineType.ONNXRUNTIME

        self.session = get_engine(cfg["engine_type"])(cfg)
        self.cfg = cfg

        self.preprocess_op = TablePreprocess()

        character = self.session.get_character_list()
        self.postprocess_op = TableLabelDecode(character, cfg)

    def __call__(
        self, ori_imgs: List[np.ndarray]
    ) -> Tuple[List[str], List[np.ndarray]]:
        imgs, shape_lists = self.preprocess_op(ori_imgs)

        bbox_preds, struct_probs = self.session(imgs.copy())

        table_structs, cell_bboxes = self.postprocess_op(
            bbox_preds, struct_probs, shape_lists, ori_imgs
        )
        return table_structs, cell_bboxes
