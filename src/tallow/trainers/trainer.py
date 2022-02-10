from typing import Dict, TypedDict

import torch
import torchmetrics as tm
from torch import nn

from tallow.common.supports import SupportsStateDict


class TrainerStateDict(TypedDict):
    model: Dict[str, torch.Tensor]
    criterion: Dict[str, torch.Tensor]
    optimizer: Dict[str, torch.Tensor]
    num_epochs: int
    num_batches: int


class Trainer(SupportsStateDict[TrainerStateDict]):
    # attributes
    model: nn.Module
    metric: tm.Metric
