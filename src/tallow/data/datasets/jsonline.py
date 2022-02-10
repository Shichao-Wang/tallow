import json
from typing import Iterator, TypeVar

from . import dataset

T_co = TypeVar("T_co", contravariant=True)


class JsonlineDataset(dataset.SizedDataset[T_co]):
    def __init__(self, jsonl_file: str) -> None:
        length = sum([1 for _ in open(jsonl_file)])
        super().__init__(length)
        self._jsonl_file = jsonl_file

    def __iter__(self) -> Iterator[T_co]:
        with open(self._jsonl_file) as fp:
            examples = map(json.loads, fp)
            yield from examples
