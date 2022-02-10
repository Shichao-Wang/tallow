import torch

from .backend import Backend
from .cuda import CudaBackend


def auto_select():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        return CudaBackend(device)
    else:
        raise ValueError()


__all__ = ["Backend", "CudaBackend", "auto_select"]
