import logging
from typing import TYPE_CHECKING, List, Mapping, Sequence

import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from tallow.data import datasets

from .backend import Backend, T_co

if TYPE_CHECKING:
    from tallow.trainers.trainer import TrainContext


logger = logging.getLogger(__name__)


class AbstractNativeBackend(Backend):
    def __init__(self, amp: bool = False) -> None:
        super().__init__()
        self._amp_enabled = amp
        self._grad_scaler = GradScaler(enabled=self._amp_enabled)

    def setup_module(self, module: nn.Module) -> nn.Module:
        if self._amp_enabled:
            module.forward = autocast(self._amp_enabled)(module.forward)
        return module

    def backward(self, loss: torch.Tensor):
        r = self._grad_scaler.scale(loss).backward()
        return r

    def step_and_zero_grad(self, ctx: "TrainContext"):
        if ctx.grad_clipper:
            self._grad_scaler.unscale_(ctx.optim)
            ctx.grad_clipper.clip(ctx.model.parameters())

        r = self._grad_scaler.step(ctx.optim)
        self._grad_scaler.update()
        ctx.optim.zero_grad()
        return r


class SingletonCudaBackend(AbstractNativeBackend):
    def __init__(self, device: torch.device, amp: bool = False) -> None:
        super().__init__(amp)
        self._device = device

    def setup_module(self, module: nn.Module):
        m = super().setup_module(module)
        return m.to(self._device)

    def setup_dataset(self, dataset: datasets.Dataset):
        if isinstance(dataset, CudaDataset):
            if dataset.device == self._device:
                return dataset
            else:
                return CudaDataset(dataset.dataset, self._device)
        else:
            return CudaDataset(dataset, self._device)

    def backward(self, loss: torch.Tensor):
        return super().backward(loss)

    def step_and_zero_grad(self, ctx: "TrainContext"):
        return super().step_and_zero_grad(ctx)


class DistributedCudaBackend(Backend):
    def __init__(self, devices: List[torch.device]) -> None:
        super().__init__()
        self._devices = devices


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
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, Sequence):
        return obj.__class__([move_to_device(item, device) for item in obj])
    elif isinstance(obj, Mapping):
        return obj.__class__(
            {k: move_to_device(v, device) for k, v in obj.items()}
        )
    else:
        return obj
