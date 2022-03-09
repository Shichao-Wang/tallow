import sys
from typing import Mapping

import torch
import torchmetrics as tm
from pandas import DataFrame, Series
from tqdm import tqdm

from tallow import backends
from tallow.data.datasets import Dataset
from tallow.nn import forwards


class Evaluator:
    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        metric: tm.Metric,
        backend: backends.Backend = None,
    ) -> None:
        self._datasets = datasets
        self._metric = metric

        if backend is None:
            backend = backend or backends.auto_select()
        self._backend = backend

        for k in self._datasets:
            self._datasets[k] = self._backend.setup_dataset(self._datasets[k])
        self._metric = self._backend.setup_module(self._metric)

    @torch.no_grad()
    def execute(self, model: torch.nn.Module) -> DataFrame:
        model = self._backend.setup_module(model)

        model.eval()
        dataframe = DataFrame()
        for key in self._datasets:
            metric_values = run_evaluate_dataset(
                model, self._datasets[key], self._metric
            )
            dataframe[key] = Series(metric_values)
        return dataframe


def run_evaluate_dataset(
    model: torch.nn.Module, dataset: Dataset, metric: tm.Metric
):
    metric.reset()
    for model_inputs in tqdm(
        dataset,
        desc="Evaluating",
        leave=False,
        file=sys.stderr,
        dynamic_ncols=True,
    ):
        model_outputs = forwards.module_forward(model, **model_inputs)
        forwards.metric_forward(metric, **model_inputs, **model_outputs)

    metric_tensors = metric.compute()
    if isinstance(metric_tensors, torch.Tensor):
        metric_name = metric.__class__.__name__.lower()
        metric_tensors = {metric_name: metric_tensors}

    metric_values = {}
    for key in metric_tensors:
        metric_values[key] = metric_tensors[key].item()
    return metric_values
