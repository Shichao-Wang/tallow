import enum
from typing import Collection

from ..trainer import Trainer


class Hooks(enum.Enum):
    ON_TRAIN_BEGIN = "on_train_begin"
    ON_EPOCH_BEGIN = "on_epoch_begin"
    ON_BATCH_BEGIN = "on_batch_begin"
    ON_BATCH_END = "on_batch_end"
    ON_EPOCH_END = "on_epoch_end"
    ON_VALID_BEGIN = "on_valid_begin"
    ON_VALID_END = "on_valid_end"
    ON_TRAIN_END = "on_train_end"


class HookException(Exception):
    pass


class Hook:
    def on_train_begin(self, trainer: Trainer):
        pass

    def on_epoch_begin(self, trainer: Trainer):
        pass

    def on_batch_begin(self, trainer: Trainer):
        pass

    def on_batch_end(self, trainer: Trainer):
        pass

    def on_epoch_end(self, trainer: Trainer):
        pass

    def on_valid_begin(self, trainer: Trainer):
        pass

    def on_valid_end(self, trainer: Trainer):
        pass

    def on_train_end(self, trainer: Trainer):
        pass


class HookManager:
    def __init__(self, hooks: Collection[Hook]) -> None:
        self._hooks = hooks

    def call_hook(self, hook: Hooks, trainer: Trainer):
        for h in self._hooks:
            hook_fn = getattr(h, hook.value)
            hook_fn(trainer)
