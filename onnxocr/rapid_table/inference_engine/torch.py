# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Union

import numpy as np
import torch
from tokenizers import Tokenizer

from ..table_structure.unitable.unitable_modules import Encoder, GPTFastDecoder
from ..utils.logger import Logger
from .base import InferSession

root_dir = Path(__file__).resolve().parent.parent


class TorchInferSession(InferSession):
    def __init__(self, cfg) -> None:
        self.logger = Logger(logger_name=__name__).get_log()

        engine_cfg = self.update_params(
            self.engine_cfg[cfg["engine_type"].value], cfg["engine_cfg"]
        )

        self.device = torch.device(
            f"cuda:{engine_cfg.gpu_id}"
            if torch.cuda.is_available() and engine_cfg.use_cuda
            else "cpu"
        )

        model_info = cfg["model_dir_or_path"]
        self.encoder = self._init_model(model_info["encoder"], Encoder)
        self.decoder = self._init_model(model_info["decoder"], GPTFastDecoder)
        self.vocab = self._init_vocab(model_info["vocab"])

    def _init_model(self, model_path, model_class):
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval().to(self.device)
        return model

    def _init_vocab(self, vocab_path: Union[str, Path]):
        return Tokenizer.from_file(str(vocab_path))

    def __call__(self, img: np.ndarray):
        raise NotImplementedError(
            "Inference logic is not implemented for TorchInferSession."
        )

    def have_key(self, key: str = "character") -> bool:
        return False


class TorchInferError(Exception):
    pass
