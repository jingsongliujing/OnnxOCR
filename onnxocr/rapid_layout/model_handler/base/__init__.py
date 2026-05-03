# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from abc import ABC, abstractmethod


class BaseModelHandler(ABC):
    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass
