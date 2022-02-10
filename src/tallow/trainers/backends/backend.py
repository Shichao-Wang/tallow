from typing import TypeVar

from torch import nn

from tallow.data.datasets.dataset import Dataset

T_co = TypeVar("T_co", covariant=True)


class Backend:
    def move_to_device(self, obj: T_co) -> T_co:
        return obj

    def setup_dataset(self, dataset: Dataset):
        return dataset

    def setup_module(self, model: nn.Module) -> nn.Module:
        pass
