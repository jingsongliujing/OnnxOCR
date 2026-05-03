# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import dataclasses
from typing import List, Optional

from .inference_engine.base import get_engine
from .model_handler import ModelHandler, ModelProcessor
from .utils.load_image import InputType, LoadImage
from .utils.typings import ModelType, RapidLayoutInput, RapidLayoutOutput
from .utils.utils import is_url


class RapidLayout:
    def __init__(self, cfg: Optional[RapidLayoutInput] = None, **kwargs):
        """初始化布局检测引擎。

        Args:
            cfg: 可选，完整配置；若为 None 则仅用 kwargs 构造配置。

        Kwargs（与 RapidLayoutInput 字段一致，传入时会覆盖 cfg 中同名字段）:
            model_type: 模型类型，ModelType 或 str（如 "pp_layout_cdla"），默认 PP_LAYOUT_CDLA。
            model_dir_or_path: 模型目录或单文件路径，str | Path | None，默认 None（按 model_type 自动解析）。
            engine_type: 推理引擎，EngineType 或 str（"onnxruntime" | "openvino"），默认 onnxruntime。
            engine_cfg: 引擎额外配置，dict，默认 {}。
            conf_thresh: 框置信度阈值 [0, 1]，默认 0.5。
            iou_thresh: IoU 阈值 [0, 1]，默认 0.5。
        """
        if cfg is None:
            cfg = RapidLayoutInput(**RapidLayoutInput.normalize_kwargs(kwargs))
        elif kwargs:
            cfg = dataclasses.replace(cfg, **RapidLayoutInput.normalize_kwargs(kwargs))

        if not cfg.model_dir_or_path:
            cfg.model_dir_or_path = ModelProcessor.get_model_path(cfg.model_type)

        self.session = get_engine(cfg.engine_type)(cfg)
        self.model_handler = ModelHandler(cfg, self.session)

        self.load_img = LoadImage()

    def __call__(self, img_content: InputType) -> RapidLayoutOutput:
        img = self.load_img(img_content)
        result = self.model_handler(img)
        return result


def parse_args(arg_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="Path to image for layout.")
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default=ModelType.PP_LAYOUT_CDLA.value,
        choices=[v.value for v in ModelType],
        help="Support model type",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="Box threshold, the range is [0, 1]",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="IoU threshold, the range is [0, 1]",
    )
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        help="Wheter to visualize the layout results.",
    )
    args = parser.parse_args(arg_list)
    return args


def main(arg_list: Optional[List[str]] = None):
    args = parse_args(arg_list)

    input_args = RapidLayoutInput(
        model_type=ModelType(args.model_type),
        iou_thresh=args.iou_thresh,
        conf_thresh=args.conf_thresh,
    )
    layout_engine = RapidLayout(input_args)

    results = layout_engine(args.img_path)
    print(results)

    if args.vis:
        save_path = "layout_vis.jpg"
        if not is_url(args.img_path):
            save_path = args.img_path.resolve().parent / "layout_vis.jpg"
        results.vis(save_path)


if __name__ == "__main__":
    main()
