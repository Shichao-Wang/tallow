import abc
import errno
import os
import warnings
from numbers import Number
from typing import Callable

import sortedcollections
import torch

from ..trainer import Trainer


class ModelSaveManager:
    def __init__(
        self,
        save_folder_path: str,
        num_saves: int,
        value_fn: Callable[[str], Number],  # larger better
    ) -> None:
        assert os.path.exists(save_folder_path), "Create Folder First"
        self._save_folder_path = save_folder_path
        self._num_saves = num_saves

        self.saved_models = sortedcollections.SortedList(key=value_fn)
        for file in os.listdir(self._save_folder_path):
            if not self._is_save_file(file):
                continue
            model_path = os.path.join(self._save_folder_path, file)
            self.saved_models.add(model_path)

        if len(self.saved_models) > self._num_saves:
            warnings.warn("Found but")

    @abc.abstractmethod
    def _get_save_file(self, trainer: Trainer) -> str:
        pass

    @abc.abstractmethod
    def _is_save_file(self, file: str) -> bool:
        pass

    def save(self, trainer: Trainer):
        save_file = self._get_save_file(trainer)
        save_path = os.path.join(self._save_folder_path, save_file)
        state_dict = trainer.state_dict()
        try:
            torch.save(state_dict, save_path)
        except OSError as e:
            if e.errno == errno.ENOSPC:
                raise
            raise
        else:
            self.saved_models.add(save_path)

        if len(self.saved_models) > self._num_saves:
            model_to_remove = self.saved_models.pop(0)
            os.remove(model_to_remove)
        return save_path

    def load(self, trainer: Trainer):
        best_model_path = self.saved_models[-1]
        if not os.path.exists(best_model_path):
            raise FileNotFoundError("")  # Extreme error

        state_dict = torch.load(best_model_path)
        trainer.load_state_dict(state_dict)
