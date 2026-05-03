# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .utils import mkdir
from .vis import VisTable


class EngineType(Enum):
    ONNXRUNTIME = "onnxruntime"
    TORCH = "torch"
    OPENVINO = "openvino"

class ModelType(Enum):
    SLANETPLUS = "slanet_plus"
    UNITABLE = "unitable"
    UNET = "unet" # 有线表格unet
    UNET_SLANET_PLUS = "unet_slanet_plus"  # 有线表格使用unet，无线表格使用slanet_plus
    UNET_UNITABLE = "unet_unitable"  # 有线表格使用unet，无线表格使用unitable
    PADDLE_CLS = "paddle_cls"
    Q_CLS = "q_cls"


@dataclass
class RapidTableInput:
    model_type: Optional[ModelType] = ModelType.SLANETPLUS
    model_dir_or_path: Union[str, Path, None, Dict[str, str]] = None

    engine_type: Optional[EngineType] = None
    engine_cfg: dict = field(default_factory=dict)

    use_ocr: bool = True
    ocr_params: dict = field(default_factory=dict)


@dataclass
class RapidTableOutput:
    imgs: List[np.ndarray] = field(default_factory=list)
    pred_htmls: List[str] = field(default_factory=list)
    cell_bboxes: List[np.ndarray] = field(default_factory=list)
    logic_points: List[np.ndarray] = field(default_factory=list)
    elapse: float = 0.0

    def vis(
        self,
        save_dir: Union[str, Path],
        save_name: str,
        indexes: Tuple[int, ...] = (0,),
    ) -> List[np.ndarray]:
        vis = VisTable()

        save_dir = Path(save_dir)
        mkdir(save_dir)

        results = []
        for idx in indexes:
            save_one_dir = save_dir / str(idx)
            mkdir(save_one_dir)

            save_html_path = save_one_dir / f"{save_name}.html"
            save_drawed_path = save_one_dir / f"{save_name}_vis.jpg"
            save_logic_points_path = save_one_dir / f"{save_name}_col_row_vis.jpg"

            vis_img = vis(
                self.imgs[idx],
                self.pred_htmls[idx],
                self.cell_bboxes[idx],
                self.logic_points[idx],
                save_html_path,
                save_drawed_path,
                save_logic_points_path,
            )
            results.append(vis_img)
        return results
