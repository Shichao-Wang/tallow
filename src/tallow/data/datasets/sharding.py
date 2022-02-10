import itertools
import multiprocessing
import random
import time
from typing import Iterable, Iterator, TypeVar

from . import dataset

T_co = TypeVar("T_co", contravariant=True)


def worker_loop(
    worker_id: int,
    num_workers: int,
    chunk_size: int,
    job_queue: multiprocessing.Queue,
    data_queue: multiprocessing.Queue,
):
    try:
        buf = []
        while True:
            dataset = job_queue.get()
            for item in dataset:
                if len(buf) == chunk_size:
                    data_queue.put(buf)
                    buf = []
                buf.append(item)
            if buf:
                data_queue.put(buf)
            data_queue.put(None)
    except KeyboardInterrupt:
        pass


def monitor_worker(
    job_queue: multiprocessing.Queue, data_queue: multiprocessing.Queue
):
    while True:
        print(f"Job queue: {job_queue.qsize():d}")
        print(f"Data queue: {data_queue.qsize():d}")
        time.sleep(1)


class ShardingDataset(dataset.Dataset[T_co]):
    def __init__(
        self,
        datasets: Iterable[dataset.Dataset[T_co]],
        num_workers: int,
        chunk_size: int = 1,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self._datasets = datasets
        self._num_workers = num_workers
        self._chunk_size = chunk_size
        self._shuffle = shuffle

        self._job_queue = multiprocessing.Queue()
        self._data_queue = multiprocessing.Queue(2 * num_workers)

        self._workers = [
            multiprocessing.Process(
                target=worker_loop,
                args=(
                    i,
                    num_workers,
                    self._chunk_size,
                    self._job_queue,
                    self._data_queue,
                ),
                daemon=True,
            )
            for i in range(num_workers)
        ]
        for w in self._workers:
            w.start()

    def __iter__(self) -> Iterator[T_co]:
        random.shuffle(self._datasets)

        for ds in self._datasets:
            self._job_queue.put(ds)

        unfinished_jobs = len(self._datasets)
        while unfinished_jobs:
            data = self._data_queue.get()
            if data is None:
                unfinished_jobs -= 1
            else:
                yield from data
                del data

    def __len__(self):
        return sum(len(ds) for ds in self._datasets)


class ShardableChainDataset(dataset.Dataset[T_co]):
    def __init__(
        self, datasets: Iterable[dataset.Dataset[T_co]], shuffle: bool = True
    ) -> None:
        super().__init__()
        self._datasets = datasets
        self._shuffle = shuffle

    def __iter__(self) -> Iterator[T_co]:
        from torch.utils import data

        if self._shuffle:
            random.shuffle(self._datasets)

        info = data.get_worker_info()
        if info is None:
            datasets = self._datasets
        else:
            worker_id, num_workers = info.id, info.num_workers
            datasets = self._datasets[worker_id::num_workers]

        chain = itertools.chain.from_iterable(datasets)
        yield from chain

    def __len__(self) -> int:
        return sum(len(ds) for ds in self._datasets)


class PytorchShardingDataset(dataset.Dataset[T_co]):
    def __init__(
        self,
        datasets: Iterable[dataset.Dataset[T_co]],
        num_workers: int,
        chunk_size: int,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        from torch.utils import data

        dataset = ShardableChainDataset(datasets, shuffle)
        self._loader = data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=chunk_size,
            collate_fn=lambda x: x,
            pin_memory=True,
        )

    def __iter__(self) -> Iterator[T_co]:
        for data in self._loader:
            yield from data
