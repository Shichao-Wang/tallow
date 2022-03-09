import logging
import math
import re
from typing import TYPE_CHECKING, Optional

import torch
import torchmetrics

from tallow.backends import Backend
from tallow.evaluators import Evaluator
from tallow.trainers.signals import StopTraining

from .hook import Hook
from .model_save import DiskManager

if TYPE_CHECKING:
    from tallow.trainers.trainer import TrainContext


logger = logging.getLogger(__name__)

DECIMAL_RE = r"(\d+)(\.\d+)?"
MODEL_FILE_RE = re.compile(
    r"(.*[/\\])?earlystop-e(?P<epoch>\d+)_b(?P<batch>\d+)_m(?P<value>%s)\.pt$"
    % DECIMAL_RE
)
MODEL_FILE_TEMPLATE = "earlystop-e{epoch}_b{batch}_m{value}.pt"


"""
e10_b30_m0.9650.earlystop.pt
"""


def parse_value_from_filename(filename: str) -> Optional[int]:
    matches = MODEL_FILE_RE.match(filename)
    if matches is None:
        return None
    value_string = matches.group("value")
    try:
        return float(value_string)
    except ValueError:
        return None


EARLYSTOP_DATASET = "earlystop_dataset"
EARLYSTOP_METRIC = "earlystop_metric"


class EarlyStopHook(Hook):
    def __init__(
        self,
        save_folder_path: str,
        dataset: torch.nn.Module,
        metric: torchmetrics.Metric,
        patient: int,
        prefix: str = None,
        backend: Backend = None,
        metric_field: str = None,
    ) -> None:
        super().__init__()
        self._disk_mgr = DiskManager(
            save_folder_path, 1, sort_key=parse_value_from_filename
        )
        self._evaluator = Evaluator(
            {EARLYSTOP_DATASET: dataset}, metric, backend
        )
        self._patient = patient
        self._prefix = prefix

        if metric_field is None:
            metric_field = metric.__class__.__name__.lower()
        self._metric_field = metric_field

        best_value: Optional[float] = None
        if self._disk_mgr.saved_models:
            best_value = parse_value_from_filename(
                self._disk_mgr.saved_models[-1]
            )
        self._best_value = best_value

        self._tol = 0

    def on_train_begin(self, ctx: "TrainContext"):
        if len(self._disk_mgr.saved_models) != 0:
            best_model_path = self._disk_mgr.saved_models[-1]
            logger.info("Restore checkpoint from %s" % best_model_path)
            state_dict = self._disk_mgr.load(best_model_path)
            ctx.load_state_dict(state_dict)
        return super().on_train_begin(ctx)

    def on_epoch_end(self, ctx: "TrainContext"):
        metric_dataframe = self._evaluator.execute(ctx.model)
        metric_value = metric_dataframe[EARLYSTOP_DATASET][self._metric_field]
        logger.info(
            "Epoch %d Evaluate: %s: %f",
            ctx.num_epochs,
            self._metric_field,
            metric_value,
        )

        if math.isnan(metric_value) or math.isinf(metric_value):
            raise StopTraining(ctx)

        if self._best_value and self._best_value > metric_value:
            logger.debug(
                "Tolerance increase. current: %f best: %f",
                metric_value,
                self._best_value,
            )
            self._tol += 1
        else:
            self._best_value = metric_value
            self._tol = 0
            logger.info(f"Better val: {self._best_value}")

            model_file = MODEL_FILE_TEMPLATE.format(
                epoch=ctx.num_epochs,
                batch=ctx.num_batches,
                value=self._best_value,
            )
            if self._prefix:
                model_file = self._prefix + "-" + model_file
            self._disk_mgr.save_or_substitute(ctx.state_dict(), model_file)

        if self._tol == self._patient:
            logger.debug("Early stop training")
            raise StopTraining(ctx)
