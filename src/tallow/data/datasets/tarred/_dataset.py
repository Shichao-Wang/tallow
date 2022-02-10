import os
import tarfile
from typing import Callable, Iterator, Mapping, TypeVar

from .. import dataset, utils
from ..utils import example
from . import handlers

T_co = TypeVar("T_co", contravariant=True)


def tarfile_group_examples(
    tar: tarfile.TarFile,
) -> Iterator[Mapping[str, tarfile.TarInfo]]:
    current = None
    cur_key = ""
    for ti in tar:
        if ti is None:
            break

        key, file = ti.name.split("/", maxsplit=1)
        if current is None:
            cur_key = key
            current = {file: ti}
        elif key != cur_key:
            yield current
            del current

            cur_key = key
            current = {file: ti}
        else:
            current[file] = ti

    if current:
        yield current


class TarfileDataset(dataset.Dataset[T_co]):
    def __init__(
        self,
        tar_file: str,
        additional_handlers: Mapping[str, handlers.LoadsFn] = None,
        on_exception: Callable[[Exception], bool] = utils.handlers.reraise,
    ) -> None:
        super().__init__()
        self._tar_file = tar_file
        assert os.path.isfile(self._tar_file)
        assert tarfile.is_tarfile(self._tar_file)
        self._handlers = dict(additional_handlers or {})
        self._on_exception = on_exception

        default_handlers = handlers.get_default_loads()
        for ext, handler in default_handlers.items():
            self._handlers.setdefault(ext, handler)

    def __iter__(self) -> Iterator[T_co]:
        tf = tarfile.open(self._tar_file)
        for example_ti in tarfile_group_examples(tf):
            try:
                ex = example.Example()
                for field, ti in example_ti.items():
                    key, ext = field.split(".", maxsplit=1)
                    data = tf.extractfile(ti)
                    handler = self._handlers[ext]
                    value = handler(data)
                    ex[key] = value
                yield ex

            except Exception as e:
                if self._on_exception(e):
                    continue
                else:
                    break
            finally:
                del ex
