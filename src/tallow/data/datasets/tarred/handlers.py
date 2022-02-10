import io
import json
import pickle
import warnings
from typing import Any, BinaryIO, Callable, MutableMapping, Tuple

__all__ = [
    "LoadsFn",
    "DumpsFn",
    "register_handler",
    "get_default_loads",
    "get_default_dumps",
]

LoadsFn = Callable[[BinaryIO], Any]
DumpsFn = Callable[[Any], bytes]

_registry: MutableMapping[str, Tuple[LoadsFn, DumpsFn]] = {}


def register_handler(*exts: str, loads_fn: LoadsFn, dumps_fn: DumpsFn):
    for ext in exts:
        if ext in _registry:
            warnings.warn("")
        _registry[ext] = (loads_fn, dumps_fn)


def get_default_loads():
    return {ext: loads for ext, (loads, _) in _registry.items()}


def get_default_dumps():
    return {ext: dumps for ext, (_, dumps) in _registry.items()}


def int_load(data: BinaryIO) -> int:
    return int(data.read().decode("utf-8"))


def int_dump(integer: int):
    return str(integer).encode("utf-8")


register_handler("cls", "id", loads_fn=int_load, dumps_fn=int_dump)


def json_dump(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


register_handler("json", loads_fn=json.load, dumps_fn=json_dump)
register_handler("pickle", loads_fn=pickle.load, dumps_fn=pickle.dumps)

try:
    import torch

    def torch_load(data: BinaryIO):
        return torch.load(data)

    def torch_dump(tensor: torch.Tensor):
        data = io.BytesIO()
        torch.save(tensor, data)
        return data.getvalue()

    register_handler("pt", loads_fn=torch_load, dumps_fn=torch_dump)
except ImportError:
    pass

try:
    import numpy

    def numpy_load(data: BinaryIO):
        # https://github.com/numpy/numpy/issues/7989
        buf = io.BytesIO()
        buf.write(data.read())
        buf.seek(0)
        return numpy.load(buf)

    def numpy_dump(array: numpy.ndarray):
        data = io.BytesIO()
        numpy.save(data, array)
        return data.getvalue()

    register_handler("npy", loads_fn=numpy_load, dumps_fn=numpy_dump)
except ImportError:
    pass
