import torch

from .backend import Backend
from .pytorch_native import DistributedCudaBackend, SingletonCudaBackend

__all__ = ["Backend", "SingletonCudaBackend", "auto_select"]


def auto_select(plugin: str = None, amp: bool = False):
    if plugin is None:
        # fallback to native
        num_devices = torch.cuda.device_count()
        if num_devices == 0:
            raise ValueError()
        if num_devices == 1:
            device = torch.device("cuda:0")
            return SingletonCudaBackend(device, amp)
        else:
            raise NotImplementedError()
            devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]
            return DistributedCudaBackend(devices, amp)
