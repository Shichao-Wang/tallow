import abc
import typing
from typing import Counter, Dict, List, Mapping, Union

import numpy


class SpecialTokensMixin:
    @property
    def pad_tok(self):
        raise NotImplementedError()

    @property
    def pad_id(self):
        raise NotImplementedError()

    @property
    def unk_tok(self):
        raise NotImplementedError()

    @property
    def unk_id(self):
        raise NotImplementedError()


class AbstractVocab(SpecialTokensMixin, Mapping, metaclass=abc.ABCMeta):
    # typing overloads
    @typing.overload
    def to_id(self, tokens: List[str]) -> List[int]:
        pass

    @typing.overload
    def to_id(self, token: str) -> int:
        pass

    @typing.overload
    def to_token(self, indexes: List[int]) -> List[str]:
        pass

    @typing.overload
    def to_token(self, index: int) -> str:
        pass

    # abstract methods
    @abc.abstractmethod
    def to_id(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_token(self, indexes: Union[int, List[int]]) -> Union[str, List[str]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError()

    # partial implemented
    def __call__(self, tokens: List[str]) -> numpy.ndarray:
        token_ids = self.to_id(tokens)
        return numpy.asarray(token_ids)


def _set_property(obj, name: str, value):
    shadow_name = f"_{name}"
    setattr(obj, shadow_name, value)
    prop = property(lambda self: getattr(self, shadow_name))
    setattr(obj.__class__, name, prop)


class Vocab(AbstractVocab):
    def __init__(
        self,
        token_to_index: Dict[str, int],
        preserved_tokens: Mapping[str, str] = None,
    ) -> None:
        super().__init__()
        self._token_to_index = token_to_index
        self._index_to_tokens = {
            index: token for token, index in token_to_index.items()
        }
        for name, token in preserved_tokens.items():
            assert token in self._token_to_index
            _set_property(self, name + "_tok", token)
            _set_property(self, name + "_id", self.to_id(token))

    def to_id(self, tokens: Union[str, List[str]]):
        if isinstance(tokens, str):
            token = tokens
            if token in self._token_to_index:
                return self._token_to_index[token]
            else:
                return self.unk_id
        return [self.to_id(tok) for tok in tokens]

    def to_token(self, indexes: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(indexes, int):
            index = indexes
            return self._index_to_tokens[index]

        else:
            return [self.to_token(index) for index in indexes]

    def __iter__(self):
        return iter(self._token_to_index)

    def __getitem__(self, item: str):
        return self._token_to_index[item]

    def __len__(self):
        return len(self._token_to_index)


def counter_vocab(counter: Counter, preserved_tokens: Mapping[str, str]):
    prepend_tokens = []
    for token in preserved_tokens.values():
        if token not in counter:
            prepend_tokens.append(token)

    token_list = [tok for tok, _ in counter.most_common()]
    token_list = prepend_tokens + token_list
    token_to_index = {tok: index for index, tok in enumerate(token_list)}
    return Vocab(token_to_index, preserved_tokens)
