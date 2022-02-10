import abc
import random
from typing import Iterable, Iterator, TypeVar

from torch.utils import data

from . import utils

T_co = TypeVar("T_co", contravariant=True)


class Dataset(data.IterableDataset[T_co], utils.ShortcutsMixin):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError()


class SizedDataset(Dataset[T_co]):
    def __init__(self, length: int) -> None:
        super().__init__()
        self._length = length

    def __len__(self):
        return self._length


class MapDataset(SizedDataset[T_co]):
    def __init__(
        self,
        length: int,
        *,
        shuffle: int = True,
    ) -> None:
        super().__init__(length)
        self._shuffle = shuffle

    @abc.abstractmethod
    def __getitem__(self, index: int) -> T_co:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[T_co]:
        indexes = list(range(len(self)))
        if self._shuffle:
            random.shuffle(indexes)

        for index in indexes:
            yield self[index]


class ChainDataset(Dataset[T_co]):
    def __init__(
        self, datasets: Iterable[Dataset[T_co]], shuffle: bool = False
    ) -> None:
        super().__init__()
        self._datasets = datasets
        self._shuffle = shuffle

    def __iter__(self) -> Iterator[T_co]:
        if self._shuffle:
            random.shuffle(self._datasets)
        for ds in self._datasets:
            yield from ds

    def __len__(self) -> int:
        return sum(len(ds) for ds in self._datasets)
