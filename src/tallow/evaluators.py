from typing import Dict

import torch
import torchmetrics as tm
from pandas import DataFrame, Series
from torch import nn

from tallow.data import datasets
from tallow.nn import forwards
from tallow.trainers.backends.backend import Backend

from .trainers import SupervisedTrainer


class Evaluator:
    def __init__(
        self, model: nn.Module, metric: tm.Metric, backend: Backend
    ) -> None:
        self._model = model
        self._metric = metric
        self._backend = backend

    def execute(self, datasets: Dict[str, datasets.Dataset]) -> DataFrame:
        self._backend.setup_module(self._model)
        self._backend.setup_module(self._metric)
        self._model.eval()

        dataframe = self._do_evaluate(datasets)

        return dataframe

    def _do_evaluate(self, datasets: Dict[str, datasets.Dataset]):
        dataframe = DataFrame()
        for key in datasets:
            metric_values = self._do_evaluate_dataset(datasets[key])
            dataframe[key] = Series(metric_values)
        return dataframe

    @torch.no_grad()
    def _do_evaluate_dataset(self, dataset: datasets.Dataset):
        dataset = self._backend.setup_dataset(dataset)

        self._metric.reset()
        for model_inputs in dataset:
            model_outputs = forwards.module_forward(self._model, **model_inputs)
            forwards.metric_forward(
                self._metric, **model_inputs, **model_outputs
            )

        metric_tensors = self._metric.compute()
        if isinstance(metric_tensors, torch.Tensor):
            metric_name = self._metric.__class__.__name__.lower()
            metric_tensors = {metric_name: metric_tensors}

        metric_values = {}
        for key in metric_tensors:
            metric_values[key] = metric_tensors[key].item()
        return metric_values


def trainer_evaluator(
    trainer: SupervisedTrainer,
    metric: tm.Metric = None,
):
    metric = metric or trainer.metric
    return Evaluator(
        model=trainer.model, metric=metric, backend=trainer.backend
    )
