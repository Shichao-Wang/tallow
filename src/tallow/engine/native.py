import copy
import itertools as it
import numbers
import sys
from typing import Dict, Iterable, Iterator, List, Mapping, TypeVar

import torch
import tqdm
from torch import nn, optim
from torch.utils import data as td

from tallow.engine import history

from .. import metrics, misc
from .. import nn as thnn
from . import base, callbacks, utils

T_co = TypeVar("T_co", contravariant=True)


class InfiniteIterable(Iterable[T_co]):
    def __init__(self, iterable: Iterable[T_co]) -> None:
        self._iterable = iterable

    def __iter__(self) -> Iterator[T_co]:
        while True:
            yield from self._iterable


def ichunked(iterator: Iterator[T_co], chunk_size: int):
    MARK = object()
    while True:
        head = next(iterator, MARK)
        if head is MARK:
            break
        yield it.chain([head], it.islice(iterator, chunk_size - 1))


class NativeEngine(base.Engine):
    def __init__(self, model: nn.Module, device: misc.TorchDevice) -> None:
        super().__init__()
        self._device = device
        self._model: nn.Module = model.to(self._device)

    @property
    def model(self):
        return self._model

    def train(
        self,
        train_data: td.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        num_steps: int,
        metric: metrics.TorchMetricProtocol,
        val_data: td.DataLoader = None,
        epoch_size: int = None,
        callback_list: List[callbacks.Callback] = None,
    ):
        optimizer = prepare_optimizer(optimizer, self._device)
        cb_mgr = callbacks.Manager(callback_list or [])
        epoch_size = epoch_size or len(train_data)

        infinite_batches = InfiniteIterable(train_data)
        train_batches = it.islice(infinite_batches, 0, num_steps)
        enum_train_data = enumerate(train_batches)

        epoch, step = 0, 0  # initial value
        hist = []
        cb_mgr.on_train_begin(engine=self)
        for epoch, enum_epoch_data in enumerate(
            ichunked(enum_train_data, epoch_size)
        ):
            enum_epoch_tqdm = tqdm.tqdm(
                enum_epoch_data,
                desc=f"Epoch {epoch}",
                total=epoch_size,
            )
            cb_mgr.on_epoch_begin(engine=self, epoch=epoch)
            self._model.train()
            for step, model_inputs in enum_epoch_tqdm:
                model_inputs = move_to_device(model_inputs, self._device)
                cb_mgr.on_step_begin(
                    engine=self, **model_inputs, epoch=epoch, step=step
                )
                model_outputs = thnn.module_forward(self._model, **model_inputs)
                losses = thnn.module_forward(
                    criterion, **model_outputs, **model_inputs
                )
                optimizer.zero_grad()
                cb_mgr.on_backward_begin(
                    engine=self, epoch=epoch, step=step, losses=losses
                )
                torch.sum(losses).backward()
                cb_mgr.on_backward_end(engine=self, epoch=epoch, step=step)
                optimizer.step()
                cb_mgr.on_step_end(
                    engine=self,
                    epoch=epoch,
                    step=step,
                    **model_inputs,
                    **model_outputs,
                )

            identified_string = f"Step={step} Epoch={epoch}"
            print(identified_string, file=sys.stderr)

            metric_outputs = {}
            if isinstance(val_data, Mapping):
                for name, data in val_data.items():
                    val_metric_dict = self.test(data, metric)
                    val_metric_string = utils.format_dict(val_metric_dict)
                    print(
                        f"{name}:\t".capitalize() + val_metric_string,
                        file=sys.stderr,
                    )
                    metric_outputs[name] = val_metric_dict

            else:
                val_metric_dict = self.test(val_data, metric)
                val_metric_string = utils.format_dict(val_metric_dict)
                print("Val:\t" + val_metric_string, file=sys.stderr)
                metric_outputs["val"] = val_metric_dict

            cb_mgr.on_epoch_end(
                engine=self, epoch=epoch, step=step, **metric_outputs
            )
            record = history.Record(
                step=step, epoch=epoch, metrics=metric_outputs
            )
            hist.append(record)

        cb_mgr.on_train_end(engine=self)

        return hist

    @torch.no_grad()
    def test(
        self, test_data: td.DataLoader, metric: metrics.TorchMetricProtocol
    ) -> Dict[str, numbers.Number]:
        test_metric = copy.deepcopy(metric).to(self._device)

        test_data_tqdm = tqdm.tqdm(test_data, leave=False)

        self._model.eval()
        for model_inputs in test_data_tqdm:
            model_inputs = move_to_device(model_inputs, self._device)
            model_outputs = thnn.module_forward(self._model, **model_inputs)
            _ = thnn.metric_forward(
                test_metric, **model_outputs, **model_inputs
            )

        test_metric_dict = {
            k: t.item() for k, t in test_metric.compute().items()
        }
        test_metric_dict.setdefault(
            "epoch_time", test_data_tqdm.format_dict["elapsed"]
        )
        return test_metric_dict


def move_to_device(obj, device: misc.TorchDevice):
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, Mapping):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, List):
        return [move_to_device(item, device) for item in obj]
    else:
        return obj
        # raise ValueError(f"Unknown type {type(obj)}")


def prepare_optimizer(optimizer: optim.Optimizer, device: misc.TorchDevice):
    state_dict = optimizer.state_dict()
    device_state_dict = move_to_device(state_dict, device)
    optimizer.load_state_dict(device_state_dict)
    return optimizer
