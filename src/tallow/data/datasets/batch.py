from typing import Any, Dict, List, Mapping, Text, TypeVar

import torch
import torch.utils.data
from torch import Tensor

FieldData = TypeVar("FieldData", Tensor, Any)

BatchElement = TypeVar("BatchElement", Tensor, List)

Instance = Dict[str, FieldData]
"""
Instance is a container of data point.
It performs as a dict object.
"""


def recursively_apply(obj, method, *args, **kwargs):
    if hasattr(obj, method):
        return getattr(obj, method)(*args, **kwargs)
    elif isinstance(obj, List):
        return obj.__class__(
            [recursively_apply(item, method, *args, **kwargs) for item in obj]
        )
    elif isinstance(obj, Mapping):
        return obj.__class__(
            {
                key: recursively_apply(value, method, *args, **kwargs)
                for key, value in obj.items()
            }
        )
    else:
        return obj


class Batch(Dict[str, BatchElement]):
    def pin_memory(self):
        return Batch(
            {
                key: recursively_apply(value, "pin_memory")
                for key, value in self.items()
            }
        )
        # return self._apply_fields("pin_memory")

    def to(self, device: str):
        return Batch(
            {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in self.items()
            }
        )


class DefaultCollator:
    def __init__(self, depth: int) -> None:
        self._depth = depth

    def collate(self, data_list: List[Dict[str, FieldData]]) -> Batch:
        collated = auto_collate(data_list)
        return Batch(collated)


def default_collate(data_list: List[Dict[str, FieldData]]) -> Batch:
    return DefaultCollator(1).collate(data_list)


def auto_collate(data_list: List):
    if len(data_list) == 0:
        return data_list
    ele = data_list[0]
    # premitive types
    if isinstance(ele, torch.Tensor):  # official implementation
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in data_list])
            storage = ele.storage()._new_shared(numel)
            out = ele.new(storage)
        # return torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True)
        return torch.stack(data_list, 0, out=out)
    elif isinstance(ele, (int, float)):
        return torch.tensor(data_list)
    elif isinstance(ele, Text):
        return data_list
    # collection types
    elif isinstance(ele, List):
        collated = [auto_collate(e) for e in data_list]
        collated_ele = collated[0]
        if isinstance(collated_ele, torch.Tensor):
            if len(collated_ele.size()):
                return torch.nn.utils.rnn.pad_sequence(
                    collated, batch_first=True
                )
            else:
                return torch.stack(collated)
        else:
            return collated
    elif isinstance(ele, tuple) and hasattr(ele, "_fields"):  # namedtuple
        tup_class = type(ele)
        return tup_class(*(auto_collate([item for item in zip(*data_list)])))

    elif isinstance(ele, Mapping):
        ret = {k: auto_collate([d[k] for d in data_list]) for k in ele}
        return ret
    else:
        return data_list
