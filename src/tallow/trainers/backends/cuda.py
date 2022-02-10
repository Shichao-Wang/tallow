from typing import Mapping, Sequence

import torch
from torch import nn

from tallow.data import datasets

from .backend import Backend, T_co


class CudaBackend(Backend):
    def __init__(self, device: str) -> None:
        super().__init__()
        self._device = torch.device(device)

    def setup_module(self, module: nn.Module):
        return module.to(self._device)

    def setup_dataset(self, dataset: datasets.Dataset):
        if isinstance(dataset, CudaDataset):
            if dataset.device == self._device:
                return dataset
            else:
                return CudaDataset(dataset.dataset, self._device)
        else:
            return CudaDataset(dataset, self._device)


class CudaDataset(datasets.TransformDataset):
    def __init__(
        self, dataset: datasets.Dataset[T_co], device: torch.device
    ) -> None:
        super().__init__(dataset, move_to_device, device)
        self._device = device

    @property
    def dataset(self):
        return self._dataset

    @property
    def device(self):
        return self._device


def move_to_device(obj: T_co, device: torch.device) -> T_co:
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, Sequence):
        return obj.__class__([move_to_device(item, device) for item in obj])
    elif isinstance(obj, Mapping):
        return obj.__class__(
            {k: move_to_device(v, device) for k, v in obj.items()}
        )
    else:
        return obj
