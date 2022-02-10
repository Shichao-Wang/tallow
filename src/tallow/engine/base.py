import abc
import numbers
from typing import Dict, Union

from torch import nn
from torch.optim import Optimizer
from torch.utils import data as td

from ..metrics import TorchMetricProtocol


class Engine(metaclass=abc.ABCMeta):
    @property
    def model(self) -> nn.Module:
        raise NotImplementedError()

    @abc.abstractmethod
    def train(
        self,
        train_data: td.DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        num_steps: int,
        metric: TorchMetricProtocol,
        val_data: td.DataLoader = None,
        epoch_size: int = None,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def test(
        self,
        test_data: td.DataLoader,
        metric: TorchMetricProtocol,
    ) -> Dict[str, numbers.Number]:
        raise NotImplementedError()
