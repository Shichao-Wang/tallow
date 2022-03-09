import enum
from typing import TYPE_CHECKING, Collection

if TYPE_CHECKING:
    from tallow.trainers import TrainContext


class Hooks(enum.Enum):
    ON_TRAIN_BEGIN = "on_train_begin"
    ON_EPOCH_BEGIN = "on_epoch_begin"
    ON_BATCH_BEGIN = "on_batch_begin"
    ON_BATCH_END = "on_batch_end"
    ON_EPOCH_END = "on_epoch_end"
    # ON_VALID_BEGIN = "on_valid_begin"
    # ON_VALID_END = "on_valid_end"
    ON_TRAIN_END = "on_train_end"


class HookException(Exception):
    pass


class Hook:
    def on_train_begin(self, ctx: "TrainContext"):
        pass

    def on_epoch_begin(self, ctx: "TrainContext"):
        pass

    def on_batch_begin(self, ctx: "TrainContext"):
        pass

    def on_batch_end(self, ctx: "TrainContext"):
        pass

    def on_epoch_end(self, ctx: "TrainContext"):
        pass

    # def on_valid_begin(self, ctx: "TrainContext"):
    #     pass

    # def on_valid_end(self, ctx: "TrainContext"):
    #     pass

    def on_train_end(self, ctx: "TrainContext"):
        pass


class HookManager:
    def __init__(self, hooks: Collection[Hook]) -> None:
        self._hooks = hooks

    def call_hook(self, hook: Hooks, ctx: "TrainContext"):
        for h in self._hooks:
            hook_fn = getattr(h, hook.value)
            hook_fn(ctx)
