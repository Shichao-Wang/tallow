from numbers import Number
from typing import Dict

import torch
import torchmetrics as tm
from torch import nn

from tallow.datasets import Dataset
from tallow.nn.misc import metric_forward, module_forward

from ..trainer import Trainer
from . import Hook


class EvaluateHook(Hook):
    def __init__(
        self, datasets: Dict[str, Dataset], metric: tm.Metric = None
    ) -> None:
        super().__init__()
        self._datasets = datasets
        self._metric = metric

    def on_train_begin(self, trainer: Trainer):
        trainer.optimizer
        if self._metric is None:
            self._metric = trainer.metric.clone()

    def on_valid_end(self, trainer: Trainer):
        for key in self._datasets:
            metric_values = evaluate_model(
                trainer.model, self._datasets[key], self._metric
            )
            trainer.logger.info(
                "Evaluate %s: %s", key, format_metric_dict(metric_values)
            )


def format_metric_dict(d: Dict[str, Number]):
    string = ""
    for k, v in d.items():
        string += f"|{k} {v:.4f}"

    if string:
        string += "|"

    return string


@torch.no_grad()
def evaluate_model(
    model: nn.Module, dataset: Dataset, metric: tm.Metric
) -> Dict[str, Number]:
    metric.reset()

    _prev = model.training
    model.eval()
    for model_inputs in dataset:
        model_outputs = module_forward(model, **model_inputs)
        metric_forward(metric, **model_inputs, **model_outputs)
    model.train(_prev)

    metric_tensors = metric.compute()
    if isinstance(metric_tensors, torch.Tensor):
        metric_name = metric.__class__.__name__.lower()
        metric_tensors = {metric_name: metric_tensors}

    metric_values = {}
    for key in metric_tensors:
        metric_values[key] = metric_tensors[key].item()
    return metric_values
