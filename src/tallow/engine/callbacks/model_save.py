import itertools
import operator
import os
import sys
from typing import Dict, Iterable, Literal, NamedTuple, NoReturn

import torch
from torch import nn

from ..base import Engine
from .base import Callback

DEFAULT_MODEL_NAME_TEMPLATE = "epoch_{epoch}.pt"


def parse_metric_value(metric_string: str):
    if metric_string.startswith(("+", "-")):
        op_string, metric_name = metric_string[0], metric_string[1:]
    else:
        op_string = "-"
        metric_name = metric_string
    return op_string, metric_name


METRIC_OPERATOR = {"+": operator.ge, "-": operator.le}
EPOCH_OPERATOR = {"first": max, "last": min}


class ModelSaveKey(NamedTuple):
    metric: float
    epoch: int


def _make_key_pop_func(
    epoch_save: Literal["first", "last"],
    metric_save: Literal["+", "-"],
):
    metric_cmp = METRIC_OPERATOR[metric_save]
    epoch_pop = EPOCH_OPERATOR[epoch_save]

    def key_pop_func(
        keys: Iterable[ModelSaveKey], new: ModelSaveKey
    ) -> ModelSaveKey:
        total_keys = itertools.chain(keys, [new])
        metric_pop_keys = [key for key in total_keys if metric_cmp(new, key)]
        pop_key = epoch_pop(metric_pop_keys, key=operator.itemgetter(1))
        return pop_key

    return key_pop_func


class ModelSaveCallback(Callback):
    first = "first"
    last = "last"

    def __init__(
        self,
        save_folder_path: str,
        model_template: str,
        metric_value: str = "loss",
        n: int = 1,
        same_value: Literal["first", "last"] = "first",
        save_on_raise: bool = True,
    ):
        self._save_folder_path = save_folder_path
        self._model_template = model_template
        self._save_on_raise = save_on_raise
        self._same_value = same_value
        self._n = n

        op_char, self._metric_name = parse_metric_value(metric_value)
        self._key_pop_func = _make_key_pop_func(same_value, op_char)

        self._saved_models: Dict[ModelSaveKey, str] = {}

        os.makedirs(save_folder_path, exist_ok=True)

    def _save(self, model: nn.Module, epoch: int, **kwargs):
        metric_value = float(self._metric_name.format(**kwargs))
        # metric_value = kwargs[self._metric_namespace][self._metric_name]
        new_key = ModelSaveKey(metric_value, epoch)

        key_to_remove = self._key_pop_func(self._saved_models, new_key)
        # if is a better result or the list is not full.
        if key_to_remove != new_key or len(self._saved_models) < self._n:
            model_name = self._model_template.format(epoch=epoch, **kwargs)
            model_save_file = os.path.join(self._save_folder_path, model_name)
            save_model(model, model_save_file)
            self._saved_models[new_key] = model_save_file

            print(f"Save new model with `{new_key}`", file=sys.stderr)

        if len(self._saved_models) > self._n:
            os.remove(self._saved_models[key_to_remove])
            del self._saved_models[key_to_remove]

            print(f"Remove out-dated model `{key_to_remove}`", file=sys.stderr)

    def on_epoch_end(self, engine: Engine, **kwargs):
        self._save(engine.model, **kwargs)

    def on_exception(self, engine: Engine, **kwargs):
        if self._save_on_raise:
            self._save(engine.model, **kwargs)

    def get_best_model_path(self) -> str:
        key = max(self._saved_models, key=operator.attrgetter("metric"))
        return self._saved_models[key]


def save_model(model: nn.Module, save_file: str, state_dict: bool = True):
    save_obj = model.state_dict() if state_dict else model
    torch.save(save_obj, save_file)


class BestModelSaveCallback(Callback):
    def __init__(self, model_save_path: str) -> None:
        super().__init__()
        self._model_save_path = model_save_path

    def on_train_begin(self, *, engine: Engine, **kwargs) -> NoReturn:
        engine.model.load_state_dict()
        return super().on_train_begin(engine=engine, **kwargs)
