import os
from typing import NoReturn

import torch

import molurus.logger

from ..base import Engine
from .base import Callback

logger = molurus.logger.get_logger(__name__)


class CheckpointCallback(Callback):
    """
    Used for save the trainer trainer.

    If `save_path` is not empty, trainer will retrieve the checkpoint trainer at the begin of training
    """

    def __init__(self, save_path: str, delete_on_finish: bool = False):
        self._save_path = save_path
        self._delete_on_finish = delete_on_finish

    def on_train_begin(self, engine: Engine, global_step: int) -> NoReturn:
        if os.path.exists(self._save_path):
            states = torch.load(
                self._save_path, map_location=lambda store, loc: store
            )
            engine.model.load_state_dict(states["model_state_dict"])
            engine._optimizer.load_state_dict(states["optimizer_state_dict"])
            global_steps = states["steps"]
            logger.info(
                f"CheckpointCallback@on_train_begin: "
                f"Load trainer trainer from {self._save_path}."
            )

    def on_epoch_end(self, engine: Engine, global_step: int) -> NoReturn:
        states = {
            "model_state_dict": engine.model.state_dict(),
            "optimizer_state_dict": engine.optimizer.state_dict(),
            "steps": global_step,
        }
        torch.save(states, self._save_path)
        logger.info(
            f"CheckpointCallback@on_epoch_end: "
            f"Checkpoint saved at {self._save_path}"
        )

    def on_train_end(self, engine: Engine) -> NoReturn:
        if self._delete_on_finish:
            if os.path.exists(self._save_path):
                os.remove(self._save_path)
                logger.info(
                    "CheckpointCallback@on_train_end: "
                    "Delete checkpoint on finish."
                    f"{self._save_path}"
                )
