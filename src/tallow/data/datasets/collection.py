import pickle
import sys
import timeit
from typing import Collection, TypeVar

from . import dataset

T_co = TypeVar("T_co")


class CollectionDataset(dataset.MapDataset[T_co]):
    def __init__(
        self, items: Collection[T_co], *, shuffle: bool = True
    ) -> None:
        super().__init__(len(items), shuffle=shuffle)
        self._items = items

    def __getitem__(self, index: int) -> T_co:
        return self._items[index]


class PickleDataset(CollectionDataset[T_co]):
    def __init__(self, pickle_file: str, *, shuffle: bool = True) -> None:
        print(
            "Loading Pickle file %s" % pickle_file,
            end="\t",
            file=sys.stderr,
            flush=True,
        )
        with open(pickle_file, "rb") as fp:
            start = timeit.default_timer()
            objs = pickle.load(fp)
            end = timeit.default_timer()
            print("%f" % (end - start), file=sys.stderr)
        super().__init__(objs, shuffle=shuffle)
