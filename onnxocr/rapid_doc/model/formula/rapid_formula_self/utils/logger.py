# -*- encoding: utf-8 -*-
# @Author: Jocker1212
# @Contact: xinyijianggo@gmail.com
import logging
from typing import Optional

import colorlog


class Logger:

    def __init__(self, logger_name: Optional[str] = None, log_level=logging.DEBUG):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(levelname)s] %(asctime)s [RapidTable] %(filename)s:%(lineno)d: %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

            console_handler.setLevel(log_level)
            self.logger.addHandler(console_handler)

    def get_log(self):
        return self.logger
