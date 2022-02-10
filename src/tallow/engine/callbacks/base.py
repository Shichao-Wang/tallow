from __future__ import annotations

from abc import ABCMeta
from typing import NoReturn

from ..base import Engine


class Callback(metaclass=ABCMeta):
    def on_train_begin(self, *, engine: Engine, **kwargs) -> NoReturn:
        pass

    def on_epoch_begin(self, *, engine: Engine, **kwargs) -> NoReturn:
        pass

    def on_step_begin(self, *, engine: Engine, **kwargs) -> NoReturn:
        pass

    def on_backward_begin(self, *, engine: Engine, **kwargs) -> NoReturn:
        pass

    def on_backward_end(self, *, engine: Engine, **kwargs) -> NoReturn:
        pass

    def on_step_end(self, *, engine: Engine, **kwargs) -> NoReturn:
        pass

    def on_epoch_end(self, *, engine: Engine, **kwargs) -> NoReturn:
        pass

    def on_train_end(self, *, engine: Engine, **kwargs) -> NoReturn:
        pass

    def on_exception(self, *, engine: Engine, **kwargs) -> NoReturn:
        raise
