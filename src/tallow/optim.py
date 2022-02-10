from typing import Type

from torch import optim


class Optimizer:
    def __init__(self, optim_class: Type[optim.Optimizer], **kwargs) -> None:
        self._opt_class = optim_class
        self._kwargs = kwargs

    def build(self, params) -> optim.Optimizer:
        return self._opt_class(params, self._kwargs)
