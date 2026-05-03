# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
from pathlib import Path
from typing import Dict, Union

from ..utils import DownloadFile, DownloadFileInput, Logger, ModelType, mkdir, read_yaml


class ModelProcessor:
    logger = Logger(logger_name=__name__).get_log()

    cur_dir = Path(__file__).resolve().parent
    root_dir = cur_dir.parent
    DEFAULT_MODEL_PATH = root_dir / "default_models.yaml"

    DEFAULT_MODEL_DIR = Path(os.getenv('RAPID_MODELS_DIR', root_dir / "models"))
    mkdir(DEFAULT_MODEL_DIR)

    model_map = read_yaml(DEFAULT_MODEL_PATH)

    @classmethod
    def get_model_path(cls, model_type: ModelType) -> Union[str, Dict[str, str]]:
        if model_type == ModelType.UNITABLE:
            return cls.get_multi_models_dict(model_type)
        return cls.get_single_model_path(model_type)

    @classmethod
    def get_single_model_path(cls, model_type: ModelType) -> str:
        model_info = cls.model_map[model_type.value]
        save_model_path = (
            cls.DEFAULT_MODEL_DIR / Path(model_info["model_dir_or_path"]).name
        )
        download_params = DownloadFileInput(
            file_url=model_info["model_dir_or_path"],
            sha256=model_info["SHA256"],
            save_path=save_model_path,
            logger=cls.logger,
        )
        DownloadFile.run(download_params)

        return str(save_model_path)

    @classmethod
    def get_multi_models_dict(cls, model_type: ModelType) -> Dict[str, str]:
        model_info = cls.model_map[model_type.value]

        results = {}

        model_root_dir = model_info["model_dir_or_path"]
        save_model_dir = cls.DEFAULT_MODEL_DIR / Path(model_root_dir).name
        for file_name, sha256 in model_info["SHA256"].items():
            save_path = save_model_dir / file_name

            download_params = DownloadFileInput(
                file_url=f"{model_root_dir}/{file_name}",
                sha256=sha256,
                save_path=save_path,
                logger=cls.logger,
            )
            DownloadFile.run(download_params)
            results[Path(file_name).stem] = str(save_path)

        return results
