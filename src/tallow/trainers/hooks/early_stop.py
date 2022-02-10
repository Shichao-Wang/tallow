import logging
import re
from typing import Optional

import torch

from ..signals import StopTraining
from ..trainer import Trainer
from .hook import Hook
from .model_save import ModelSaveManager

logger = logging.getLogger(__name__)

DECIMAL_RE = r"(\d+)(\.\d+)?"
MODEL_FILE_RE = re.compile(
    r"(.*[/\\])?e(?P<epoch>\d+)_b(?P<batch>\d+)_m(?P<value>%s)\.earlystop\.pt"
    % DECIMAL_RE
)
MODEL_FILE_TEMPLATE = "e{epoch}_b{batch}_m{value}.earlystop.pt"


def parse_value_from_filename(filename: str) -> int:
    matches = MODEL_FILE_RE.match(filename)
    assert matches, ""
    value_string = matches.group("value")
    return float(value_string)


class EarlyStopSaver(ModelSaveManager):
    """
    e10_b30_m0.9650.earlystop.pt
    """

    def __init__(self, save_folder_path: str) -> None:
        super().__init__(
            save_folder_path, 1, value_fn=parse_value_from_filename
        )

    def _get_save_file(self, trainer: Trainer) -> str:
        state_dict = trainer.state_dict()
        metric_tensor: torch.Tensor = trainer.metric.compute()
        return MODEL_FILE_TEMPLATE.format(
            epoch=state_dict["num_epochs"],
            batch=state_dict["num_batches"],
            value=metric_tensor.item(),
        )

    def _is_save_file(self, file: str) -> bool:
        return MODEL_FILE_RE.match(file)


class EarlyStopHook(Hook):
    def __init__(self, save_folder_path: str, patient: int) -> None:
        super().__init__()
        self._saver = EarlyStopSaver(save_folder_path)
        self._patient = patient

        best_tensor = None
        if self.best_model_path:
            best_tensor = parse_value_from_filename(self.best_model_path)
        self._best_tensor = best_tensor

        self._tol = 0

    @property
    def best_model_path(self) -> Optional[str]:
        if self._saver.saved_models:
            return self._saver.saved_models[-1]

    def on_valid_end(self, trainer: Trainer):
        metric_tensor: torch.Tensor = trainer.metric.compute()
        if self._best_tensor and self._best_tensor > metric_tensor:
            self._tol += 1
        else:
            self._best_tensor = metric_tensor
            self._tol = 0
            self._saver.save(trainer)
            logger.info(f"Best val: {self._best_tensor.item()}")

        if self._tol == self._patient:
            raise StopTraining
