import contextlib
import dataclasses
import logging
import sys
from typing import Dict, Optional, TypedDict

import torch
import torchmetrics
from tqdm import tqdm

from tallow.backends.backend import Backend
from tallow.backends.grad_clipper import GradClipper
from tallow.common.supports import SupportsStateDict
from tallow.data.datasets import Dataset
from tallow.nn import forwards

from .hooks.early_stop import EarlyStopHook
from .hooks.hook import HookManager, Hooks
from .signals import StopTraining

logger = logging.getLogger(__name__)

# class Trainer(SupportsStateDict[TrainerStateDict]):
#     # attributes
#     model: nn.Module
#     val_metric: tm.Metric


class TrainerStateDict(TypedDict):
    model: Dict[str, torch.Tensor]
    criterion: Dict[str, torch.Tensor]
    optimizer: Dict[str, torch.Tensor]
    num_epochs: int
    num_batches: int


@dataclasses.dataclass()
class TrainContext(SupportsStateDict[TrainerStateDict]):
    model: torch.nn.Module
    criterion: torch.nn.Module
    optim: torch.optim.Optimizer
    grad_clipper: Optional[GradClipper] = None

    num_epochs: int = 0
    num_batches: int = 0

    def state_dict(self) -> TrainerStateDict:
        return TrainerStateDict(
            model=self.model.state_dict(),
            criterion=self.criterion.state_dict(),
            optimizer=self.optim.state_dict(),
            num_epochs=self.num_epochs,
            num_batches=self.num_batches,
        )

    def load_state_dict(self, state_dict: TrainerStateDict):
        self.model.load_state_dict(state_dict["model"])
        self.criterion.load_state_dict(state_dict["criterion"])
        self.optim.load_state_dict(state_dict["optimizer"])
        self.num_epochs = state_dict["num_epochs"]
        self.num_batches = state_dict["num_batches"]


class Trainer:
    def __init__(
        self,
        backend: Backend,
        max_epochs: int,
        grad_clipper: GradClipper,
        num_to_accumulate: int,
        hook_mgr: HookManager,
    ) -> None:
        self.backend = backend
        self._max_epochs = max_epochs
        self._grad_clipper = grad_clipper
        self._num_to_accumulate = num_to_accumulate
        self.hook_mgr = hook_mgr

    def execute(
        self,
        model: torch.nn.Module,
        train_data: Dataset,
        criterion: torch.nn.Module,
        optimizer: torch.nn.Module,
    ) -> TrainerStateDict:
        # following should be `self.backend.launch`
        return self._execute_impl(model, train_data, criterion, optimizer)

    def _execute_impl(
        self,
        model: torch.nn.Module,
        train_data: Dataset,
        criterion: torch.nn.Module,
        optimizer: torch.nn.Module,
    ) -> TrainerStateDict:
        tr_ctx = TrainContext(model, criterion, optimizer, self._grad_clipper)
        try:
            self.hook_mgr.call_hook(Hooks.ON_TRAIN_BEGIN, tr_ctx)
            self._try_train(tr_ctx, train_data)
        # except StopTraining as e:
        #     tr_ctx = e.ctx
        finally:
            self.hook_mgr.call_hook(Hooks.ON_TRAIN_END, tr_ctx)
            return tr_ctx.state_dict()

    def _try_train(self, ctx: TrainContext, dataset: Dataset):
        dataset = self.backend.setup_dataset(dataset)
        ctx.model = self.backend.setup_module(ctx.model)
        while True:
            if (
                self._max_epochs is not None
                and ctx.num_epochs >= self._max_epochs
            ):
                raise StopTraining(ctx)

            ctx.num_epochs += 1
            self.hook_mgr.call_hook(Hooks.ON_EPOCH_BEGIN, ctx)
            ctx = self.train_epoch(ctx, dataset)
            self.hook_mgr.call_hook(Hooks.ON_EPOCH_END, ctx)

            # self.hook_mgr.call_hook(Hooks.ON_VALID_BEGIN, ctx)
            # self.eval_dataset(ctx, self.val_data)
            # self.hook_mgr.call_hook(Hooks.ON_VALID_END, ctx)

    def train_epoch(self, ctx: TrainContext, dataset: Dataset):
        with training(ctx.model):
            tqdm_prog = tqdm(
                dataset,
                desc=f"Epoch: {ctx.num_epochs}",
                leave=False,
                file=sys.stderr,
                dynamic_ncols=True,
            )
            for model_inputs in tqdm_prog:
                ctx.num_batches += 1
                self.hook_mgr.call_hook(Hooks.ON_BATCH_BEGIN, ctx)

                self.train_batch(ctx, model_inputs)

                self.hook_mgr.call_hook(Hooks.ON_BATCH_END, ctx)

        return ctx

    def train_batch(self, ctx: TrainContext, model_inputs: Dict):
        model_outputs = forwards.module_forward(ctx.model, model_inputs)
        loss: torch.Tensor = forwards.module_forward(
            ctx.criterion, model_inputs, model_outputs
        )
        self.backend.backward(loss)
        if ctx.num_batches % self._num_to_accumulate == 0:
            self.backend.step_and_zero_grad(ctx)
        return loss


class TrainerBuilder:
    def __init__(
        self,
        save_folder_path: str,
        backend: Backend,
        max_epochs: int = None,
        grad_clipper: GradClipper = None,
        num_to_accumulate: int = 1,
    ) -> None:
        self._save_folder_path = save_folder_path
        self._backend = backend
        self._max_epochs = max_epochs
        self._grad_clipper = grad_clipper
        self._num_to_accumulate = num_to_accumulate
        self._hooks = []

    def build(self) -> Trainer:
        return Trainer(
            self._backend,
            self._max_epochs,
            self._grad_clipper,
            self._num_to_accumulate,
            HookManager(self._hooks),
        )

    def earlystop(
        self,
        val_data: Dataset,
        val_metric: torchmetrics.Metric,
        patient: int,
        prefix: str = None,
        metric_field: str = None,
    ):
        self._hooks.append(
            EarlyStopHook(
                self._save_folder_path,
                val_data,
                val_metric,
                patient,
                prefix,
                self._backend,
                metric_field,
            )
        )
        return self


@contextlib.contextmanager
def training(model: torch.nn.Module):
    prev = model.training
    model.train(True)
    with torch.set_grad_enabled(True):
        yield
    model.train(prev)


@contextlib.contextmanager
def evaluating(model: torch.nn.Module):
    prev = model.training
    model.train(False)
    with torch.set_grad_enabled(False):
        yield
    model.train(prev)
