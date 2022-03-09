# import contextlib
# import logging
# import os
# import sys
# import warnings
# from typing import Callable, Iterable, List, Optional, Tuple

# import torch
# import torchmetrics as tm
# from torch import nn, optim
# from tqdm import tqdm

# from tallow import backends
# from tallow.data import datasets
# from tallow.nn import forwards
# from tallow.trainers.hooks.early_stop import EarlyStopHook

# from .hooks import Hook, HookException, HookManager, Hooks
# from .signals import StopTraining
# from .trainer import Trainer, TrainerStateDict

# logger = logging.getLogger(__name__)


# class SupervisedTrainer(Trainer):
#     """
#     The stateful object.
#     """

#     def __init__(
#         self,
#         model: nn.Module,
#         criterion: nn.Module,
#         init_optim: Callable[[nn.Module], Tuple[optim.Optimizer, Callable]],
#         val_metric: tm.Metric,
#         backend: backends.Backend = None,
#         hooks: List[Hook] = None,
#     ) -> None:
#         self.model = model
#         self.criterion = criterion
#         self._init_optim = init_optim
#         self._optimizer: Optional[optim.Optimizer] = None
#         self.val_metric = val_metric

#         self.backend = backend or backends.auto_select()
#         self._hook_mgr = HookManager(hooks or [])

#         # state variables
#         self._num_batches = 0
#         self._num_epochs = 0

#         # print some trainer information
#         # 1. Model architecture
#         # 2. Hooks order

#     def state_dict(self):
#         return TrainerStateDict(
#             model=self.model.state_dict(),
#             criterion=self.criterion.state_dict(),
#             optimizer=self.optimizer.state_dict(),
#             num_epochs=self.num_epochs,
#             num_batches=self.num_batches,
#         )

#     def load_state_dict(self, state_dict: TrainerStateDict):
#         self.model.load_state_dict(state_dict["model"])
#         self.criterion.load_state_dict(state_dict["criterion"])
#         self.optimizer.load_state_dict(state_dict["optimizer"])
#         self._num_batches = state_dict["num_batches"]
#         self._num_epochs = state_dict["num_epochs"]

#     def execute(
#         self,
#         train_data: datasets.Dataset,
#         eval_data: datasets.Dataset = None,
#         max_epochs: int = None,
#         dry_validate: bool = False,
#     ):
#         self._setup_model_and_optimizer()
#         self.backend.setup_module(self.val_metric)
#         train_data = self.backend.setup_dataset(train_data)
#         eval_data = self.backend.setup_dataset(eval_data)

#         self._hook_mgr.call_hook(Hooks.ON_TRAIN_BEGIN, self)
#         if dry_validate:
#             self._do_validate(eval_data)
#         try:
#             self._try_train(train_data, eval_data, max_epochs)
#         except StopTraining:  # correct exit
#             pass
#         except HookException:  # goes here for malfunction
#             raise
#         finally:
#             self._hook_mgr.call_hook(Hooks.ON_TRAIN_END, self)

#     def _setup_model_and_optimizer(self):
#         self.backend.setup_module(self.model)
#         init_outputs = self._init_optim(self.model)
#         if isinstance(init_outputs, optim.Optimizer):
#             self._optimizer = init_outputs
#             self._clip_norm = None
#         else:
#             self._optimizer, self._clip_norm = init_outputs
#             assert isinstance(self._optimizer, optim.Optimizer)
#             assert isinstance(self._clip_norm, Callable)

#     def _try_train(
#         self,
#         train_data: datasets.Dataset,
#         eval_data: datasets.Dataset,
#         max_epochs: int = None,
#     ):
#         while True:
#             if max_epochs is not None and self.num_epochs >= max_epochs:
#                 raise StopTraining

#             # train an epoch
#             self._do_epoch_train(train_data)
#             # validate
#             if eval_data is None:
#                 warnings.warn(
#                     "Evalulation dataset is not provided, "
#                     "use training dataset as fallback."
#                 )
#                 eval_data = train_data

#             self._do_validate(eval_data)

#     def _do_epoch_train(self, dataset: datasets.Dataset):
#         _prev = self.model.training
#         self.model.train()

#         self._num_epochs += 1
#         self._hook_mgr.call_hook(Hooks.ON_EPOCH_BEGIN, self)

#         self.epoch_train(dataset)

#         # batch_avg_loss = torch.cat(batch_losses, dim=-1).mean().item()
#         # logger.info(f"Avg batch loss: {batch_avg_loss}")

#         self._hook_mgr.call_hook(Hooks.ON_EPOCH_END, self)
#         self.model.train(_prev)

#     def epoch_train(self, dataset: datasets.Dataset):
#         for model_inputs in self.iter_batch(dataset):
#             model_outputs = forwards.module_forward(self.model, **model_inputs)
#             loss: torch.Tensor = forwards.module_forward(
#                 self.criterion, **model_outputs, **model_inputs
#             )
#             self._do_optimize(loss)

#     def _do_optimize(self, loss: torch.Tensor):
#         self.optimizer.zero_grad()
#         loss.backward()
#         if self._clip_norm:
#             self._clip_norm()

#         self.optimizer.step()

#     @torch.no_grad()
#     def _do_validate(self, dataset: datasets.Dataset):
#         with evaluating(self.model) as model:
#             self.val_metric.reset()
#             self._hook_mgr.call_hook(Hooks.ON_VALID_BEGIN, self)

#             for model_inputs in tqdm(
#                 dataset, desc="Validating", leave=False, dynamic_ncols=True
#             ):
#                 model_outputs = forwards.module_forward(model, **model_inputs)
#                 forwards.metric_forward(
#                     self.val_metric, **model_inputs, **model_outputs
#                 )

#             value = torch.Tensor.item(self.val_metric.compute())
#             metric_name = self.val_metric.__class__.__name__.lower()
#             logger.info(
#                 "Epoch %d |Valid %s %f", self.num_epochs, metric_name, value
#             )

#             self._hook_mgr.call_hook(Hooks.ON_VALID_END, self)

#     def iter_batch(self, batches: Iterable):
#         for batch in tqdm(
#             batches,
#             desc=f"Epoch {self.num_epochs}",
#             leave=False,
#             file=sys.stderr,
#             dynamic_ncols=True,
#         ):
#             self._num_batches += 1
#             self._hook_mgr.call_hook(Hooks.ON_BATCH_BEGIN, self)
#             yield batch
#             self._hook_mgr.call_hook(Hooks.ON_BATCH_END, self)

#     @property
#     def optimizer(self):
#         return self._optimizer

#     @property
#     def num_epochs(self):
#         return self._num_epochs

#     @property
#     def num_batches(self):
#         return self._num_batches


# @contextlib.contextmanager
# def training(model: nn.Module):
#     prev = model.training
#     model.train(True)
#     with torch.set_grad_enabled(True):
#         yield model
#     model.train(prev)


# @contextlib.contextmanager
# def evaluating(model: nn.Module):
#     prev = model.training
#     model.train(False)
#     with torch.set_grad_enabled(False):
#         yield model
#     model.train(prev)


# class EarlystopTrainer(SupervisedTrainer):
#     def __init__(
#         self,
#         save_folder_path: str,
#         model: nn.Module,
#         criterion: nn.Module,
#         init_optim: Callable[[nn.Module], Tuple[optim.Optimizer, Callable]],
#         val_metric: tm.Metric,
#         patient: int,
#     ) -> None:
#         self.earlystop_hook = EarlyStopHook(save_folder_path, patient)
#         hooks = [self.earlystop_hook]
#         super().__init__(
#             model,
#             criterion,
#             init_optim,
#             val_metric,
#             backends.auto_select(),
#             hooks,
#         )

#     @property
#     def best_model_path(self):
#         return self.earlystop_hook.best_model_path
