import copy
import dataclasses
import itertools as it
import sys
from numbers import Number
from typing import Dict, Iterable, Mapping, Union

import accelerate
import more_itertools as mit
import torch
from accelerate import utils as xlr_utils
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import tallow as tl

from ..metrics import TorchMetricProtocol
from ..nn import misc
from . import callbacks, history
from .base import Engine


def recursively_apply(
    func,
    data,
    *args,
    test_type=xlr_utils.is_torch_tensor,
    error_on_other_type=False,
    **kwargs,
):
    """
    Recursively apply a function on a data structure that is a nested list/tuple/dictionary of a given base type.

    Args:
        func (:obj:`callable`):
            The function to recursively apply.
        data (nested list/tuple/dictionary of :obj:`main_type`):
            The data on which to apply :obj:`func`
        *args:
            Positional arguments that will be passed to :obj:`func` when applied on the unpacked data.
        main_type (:obj:`type`, `optional`, defaults to :obj:`torch.Tensor`):
            The base type of the objects to which apply :obj:`func`.
        error_on_other_type (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to return an error or not if after unpacking :obj:`data`, we get on an object that is not of type
            :obj:`main_type`. If :obj:`False`, the function will leave objects of types different than :obj:`main_type`
            unchanged.
        **kwargs:
            Keyword arguments that will be passed to :obj:`func` when applied on the unpacked data.

    Returns:
        The same data structure as :obj:`data` with :obj:`func` applied to every object of type :obj:`main_type`.
    """
    if test_type(data):
        return func(data, *args, **kwargs)
    elif isinstance(data, (tuple, list)):
        return xlr_utils.honor_type(
            data,
            (
                recursively_apply(
                    func,
                    o,
                    *args,
                    test_type=test_type,
                    error_on_other_type=error_on_other_type,
                    **kwargs,
                )
                for o in data
            ),
        )
    elif isinstance(data, dict):
        return type(data)(
            **{
                k: recursively_apply(
                    func,
                    v,
                    *args,
                    test_type=test_type,
                    error_on_other_type=error_on_other_type,
                    **kwargs,
                )
                for k, v in data.items()
            }
        )
    elif error_on_other_type:
        raise TypeError(
            f"Can't apply {func.__name__} on object of type {type(data)}, only of nested list/tuple/dicts of objects "
            f"that satisfy {test_type.__name__}."
        )
    return data


xlr_utils.recursively_apply = recursively_apply


class AccelerateEngine(Engine):
    def __init__(
        self,
        model: nn.Module,
        split_batches: bool = False,
        dispatch_batches: bool = False,
        amp: bool = False,
        callback_list: Iterable[callbacks.Callback] = (),
    ) -> None:
        """
        :param: amp: auto mix presicion
        """
        self._xlr = accelerate.Accelerator(
            device_placement=True,
            split_batches=split_batches,
            fp16=amp,
            dispatch_batches=dispatch_batches,
        )
        self._model = self._xlr.prepare_model(model)
        self._cb_mgr = callbacks.Manager(callback_list)

    @property
    def model(self):
        return self._xlr.unwrap_model(self._model)

    def prepare_data(self, data):
        if isinstance(data, tl.IterableDataset):
            return data
        else:
            return self._xlr.prepare_data_loader(data)

    def train(
        self,
        train_data: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        num_steps: int,
        metric: TorchMetricProtocol,
        val_data: Union[DataLoader, Mapping[str, DataLoader]],
        epoch_size: int = None,
    ):

        epoch_size = epoch_size or len(train_data)
        optimizer = self._xlr.prepare_optimizer(optimizer)
        train_data = self._xlr.prepare_data_loader(train_data)

        batches = it.chain.from_iterable(it.repeat(train_data))
        train_batches = it.islice(batches, 0, num_steps)
        enum_train_data = enumerate(train_batches)

        epoch, step = 0, 0  # initial value
        hist = history.History()
        self._cb_mgr.on_train_begin(engine=self)
        for epoch, enum_epoch_data in enumerate(
            mit.ichunked(enum_train_data, epoch_size)
        ):
            enum_epoch_tqdm = tqdm(
                enum_epoch_data,
                desc=f"Epoch {epoch}",
                total=epoch_size,
            )
            self._cb_mgr.on_epoch_begin(engine=self, epoch=epoch)
            self._model.train()
            for step, model_inputs in enum_epoch_tqdm:
                if dataclasses.is_dataclass(model_inputs):
                    model_inputs = dataclasses.asdict(model_inputs)
                self._cb_mgr.on_step_begin(
                    engine=self, **model_inputs, epoch=epoch, step=step
                )
                model_outputs = misc.module_forward(self._model, **model_inputs)
                losses = misc.module_forward(
                    criterion, **model_outputs, **model_inputs
                )
                loss_value = torch.mean(losses).item()
                optimizer.zero_grad()
                self._cb_mgr.on_backward_begin(
                    engine=self, epoch=epoch, step=step, losses=losses
                )
                self._xlr.backward(torch.sum(losses))
                self._cb_mgr.on_backward_end(
                    engine=self, epoch=epoch, step=step
                )
                optimizer.step()
                self._cb_mgr.on_step_end(
                    engine=self,
                    epoch=epoch,
                    step=step,
                    **model_inputs,
                    **model_outputs,
                )
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                enum_epoch_tqdm.set_postfix({"loss": loss_value})

            identified_string = f"Step={step} Epoch={epoch}"
            print(identified_string, file=sys.stderr)

            metric_outputs = {}
            if isinstance(val_data, Mapping):
                for name, data in val_data.items():
                    val_metric_dict = self.test(data, metric)
                    val_metric_string = format_dict(val_metric_dict)
                    print(
                        f"{name}:\t".capitalize() + val_metric_string,
                        file=sys.stderr,
                    )
                    metric_outputs[name] = val_metric_dict

            else:
                val_metric_dict = self.test(val_data, metric)
                val_metric_string = format_dict(val_metric_dict)
                print("Val:\t" + val_metric_string, file=sys.stderr)
                metric_outputs["val"] = val_metric_dict

            self._cb_mgr.on_epoch_end(
                engine=self, epoch=epoch, step=step, **metric_outputs
            )
            hist.append(step, epoch, metric_outputs)

        self._cb_mgr.on_train_end(engine=self)

        return history

    @torch.no_grad()
    def test(
        self, test_loader: DataLoader, metric: TorchMetricProtocol
    ) -> Dict[str, Number]:
        test_data = self._xlr.prepare_data_loader(test_loader)
        test_metric = copy.deepcopy(metric).to(self._xlr.device)

        test_data_tqdm = tqdm(test_data, leave=False)

        self._model.eval()
        for model_inputs in test_data_tqdm:
            if dataclasses.is_dataclass(model_inputs):
                model_inputs = dataclasses.asdict(model_inputs)
            model_outputs = misc.module_forward(self._model, **model_inputs)
            metric_value_dict = misc.metric_forward(
                test_metric, **model_outputs, **model_inputs
            )

        test_metric_dict = {
            k: t.item() for k, t in test_metric.compute().items()
        }
        test_metric_dict.setdefault(
            "epoch_time", test_data_tqdm.format_dict["elapsed"]
        )
        return test_metric_dict
