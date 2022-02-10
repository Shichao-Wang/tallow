import random
import sys
from typing import Dict, List, Protocol, TypeVar, Union, runtime_checkable

import numpy
import torch
from torch import LongTensor, Tensor, nn

TorchDevice = TypeVar("TorchDevice", int, str, torch.device)


def seed_all(seed: int = 0, cudnn: bool = False):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    if cudnn:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f"Seed Everything to {seed}", file=sys.stderr)


def move_module_to_device(
    module: nn.Module,
    device: Union[TorchDevice, List[TorchDevice]],
):
    if isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, List):
        # data parallel
        raise NotImplementedError()
    else:
        raise ValueError()
    return module.to(device=device)


def seq_length_to_mask(
    seq_lengths: Union[List[int], LongTensor], max_length: int = None
) -> Tensor:
    if isinstance(seq_lengths, Tensor):
        assert seq_lengths.dim() == 1
        # noinspection PyArgumentList
        batch_size = seq_lengths.size(0)
        max_length = max_length or torch.max(seq_lengths)
    elif isinstance(seq_lengths, List):
        batch_size = len(seq_lengths)
        max_length = max(seq_lengths)
    else:
        raise TypeError()
    range_tensor: Tensor = torch.arange(max_length).expand(batch_size, -1)

    return torch.lt(
        range_tensor, torch.unsqueeze(torch.as_tensor(seq_lengths), dim=-1)
    )


@runtime_checkable
class SupportStateDict(Protocol):
    def state_dict(self):
        pass

    def load_state_dict(self, state: Dict):
        pass
