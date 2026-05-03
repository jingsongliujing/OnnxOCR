import argparse
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from .inference_engine.base import get_engine
from .model_handler import ModelHandler, ModelProcessor
from .utils.load_image import LoadImage
from .utils.typings import ModelType, RapidLayoutInput, RapidLayoutOutput, PP_DOCLAYOUT_L_Threshold, \
    PP_DOCLAYOUT_PLUS_L_Threshold, PP_DOCLAYOUTV2_Threshold
from .utils.utils import is_url


class RapidLayout:
    def __init__(self, cfg: Optional[RapidLayoutInput] = None):
        if cfg is None:
            cfg = RapidLayoutInput()

        if not cfg.conf_thresh:
            if cfg.model_type == ModelType.PP_DOCLAYOUT_PLUS_L:
                cfg.conf_thresh = PP_DOCLAYOUT_PLUS_L_Threshold
            elif cfg.model_type == ModelType.PP_DOCLAYOUTV2:
                cfg.conf_thresh = PP_DOCLAYOUTV2_Threshold
            elif cfg.model_type == ModelType.PP_DOCLAYOUTV3:
                cfg.conf_thresh = 0.3
            elif cfg.model_type == ModelType.PP_DOCLAYOUT_L:
                cfg.conf_thresh = PP_DOCLAYOUT_L_Threshold
            else:
                cfg.conf_thresh = 0.5

        if not cfg.model_dir_or_path:
            cfg.model_dir_or_path = ModelProcessor.get_model_path(cfg.model_type)

        self.session = get_engine(cfg.engine_type)(cfg)
        self.model_handler = ModelHandler(cfg, self.session)

        self.load_img = LoadImage()

    def __call__(
        self, img_contents: List[Union[str, np.ndarray, bytes, Path]], batch_size: int = 1, tqdm_enable=False
    ) -> List[RapidLayoutOutput]:

        # 先读取所有图片
        img_contents = [self.load_img(img_content) for img_content in img_contents]
        batch_results = []
        with tqdm(total=len(img_contents), desc="Layout Predict", disable=not tqdm_enable) as pbar:
            # 分批处理
            for i in range(0, len(img_contents), batch_size):
                batch_imgs = img_contents[i:i + batch_size]
                results = self.model_handler(batch_imgs)
                batch_results.extend(results)
                pbar.update(len(batch_imgs))  # 用实际处理的数量更新进度条

        return batch_results


def parse_args(arg_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="Path to image for layout.")
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default=ModelType.PP_DOCLAYOUT_L.value,
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
        results[0].vis(save_path)


if __name__ == "__main__":
    main()
