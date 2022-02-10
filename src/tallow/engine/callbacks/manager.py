from __future__ import annotations

from functools import wraps
from typing import Callable, Iterable, List

import molurus

from .base import Callback


def dispatch(func: Callable, remotes: List[Callback]):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        for remote in remotes:
            cb_fn = getattr(remote, func.__name__)
            cb_kwargs = molurus.build_fn_kwargs(cb_fn, **kwargs)
            cb_fn(*args, **cb_kwargs)

    return _wrapper


class CallbackManager(Callback):
    def __init__(self, callbacks: Iterable[Callback]):
        super(CallbackManager, self).__init__()
        self._callbacks = list(callbacks)

        self.on_train_begin = dispatch(self.on_train_begin, self._callbacks)
        self.on_epoch_begin = dispatch(self.on_epoch_begin, self._callbacks)
        self.on_step_begin = dispatch(self.on_step_begin, self._callbacks)
        self.on_backward_begin = dispatch(
            self.on_backward_begin, self._callbacks
        )
        self.on_backward_end = dispatch(self.on_backward_end, self._callbacks)
        self.on_step_end = dispatch(self.on_step_end, self._callbacks)
        self.on_epoch_end = dispatch(self.on_epoch_end, self._callbacks)
        self.on_train_end = dispatch(self.on_train_end, self._callbacks)
        self.on_exception = dispatch(self.on_exception, self._callbacks)

    def append_callback(self, cb: Callback):
        return self._callbacks.append(cb)
