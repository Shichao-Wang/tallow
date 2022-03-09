import json
from typing import TypeVar

try:
    import h5py
except ImportError:

    class H5Dataset:
        def __init__(self, *args, **kwargs) -> None:
            raise
            pass


import more_itertools as mit
import torch

from . import dataset

T = TypeVar("T", contravariant=True)


class H5Dataset(dataset.MapDataset[T]):
    def __init__(self, h5_file: str, *, shuffle: bool = True) -> None:
        self._h5_file = h5py.File(h5_file)

        def __length_fn(name: str, ds: h5py.Dataset):
            return ds.size[0]

        lengths = self._h5_file.visititems(__length_fn)
        assert mit.all_equal(lengths)
        super().__init__(lengths[0], shuffle=shuffle)

    def __getitem__(self, index: int) -> T:
        item = {}
        field: str
        ds: h5py.Dataset
        for field, ds in self._h5_file.items():
            ds_type = ds.attrs["type"]
            value = ds[index][:]
            if ds_type == "tensor":
                item[field] = torch.from_numpy(value)
            elif ds_type == "json":
                item[field] = json.loads(value)
            else:
                raise ValueError()

        return item
