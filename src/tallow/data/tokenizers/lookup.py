import warnings
from typing import Counter, Dict, List, Mapping, OrderedDict, Union

import tqdm

from . import tokenizer

DEFAULT_PRESERVED_TOKENS = {"unk": "[UNK]", "pad": "[PAD]"}


def _set_property(obj, name: str, value):
    shadow_name = f"_{name}"
    setattr(obj, shadow_name, value)
    prop = property(lambda self: getattr(self, shadow_name))
    setattr(obj.__class__, name, prop)


class LookupTokenizer(tokenizer.Tokenizer):
    def __init__(
        self,
        token_to_index: Mapping[str, int],
        casefold: str = True,
        max_length: int = None,
        pad_to_max_length: bool = False,
        preserved_tokens: Mapping[str, str] = None,
    ) -> None:
        super().__init__()
        self._token_to_index = token_to_index
        self._index_to_tokens = {
            index: token for token, index in token_to_index.items()
        }
        self._casefold = casefold
        self._max_length = max_length
        self._pad_to_max_length = pad_to_max_length

        self._preserved_tokens = preserved_tokens or DEFAULT_PRESERVED_TOKENS

        for name, token in preserved_tokens.items():
            assert token in self._token_to_index
            _set_property(self, name + "_tok", token)
            _set_property(self, name + "_id", self.to_id(token))
        if casefold:
            for token in token_to_index:
                if (
                    token.casefold() != token
                    and token not in self._preserved_tokens.values()
                ):
                    warnings.warn("Case fold tokenizer receive uncased tokens")

    def get_vocab(self) -> Dict[str, int]:
        return self._token_to_index

    def tokenize(
        self,
        words: List[str],
    ) -> List[str]:
        tokens = words
        if self._casefold:
            tokens = [
                token
                if token in self._preserved_tokens.values()
                else token.casefold()
                for token in tokens
            ]
        if self._max_length:
            tokens = tokens[: self._max_length]
        if self._pad_to_max_length:
            tokens = tokens + [self.pad_tok] * (self._max_length - len(tokens))
        return tokens

    def to_id(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            token = tokens

            if token in self._token_to_index:
                return self._token_to_index[token]
            else:
                return self.unk_id

        return [self.to_id(token) for token in tokens]

    def to_token(self, indexes: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(indexes, int):
            index = indexes
            return self._index_to_tokens[index]

        else:
            return [self.to_token(index) for index in indexes]


class CounterLookupTokenizer(LookupTokenizer):
    def __init__(
        self,
        counter: Counter,
        casefold: bool = True,
        max_size: int = None,
        min_freq: int = 0,
        max_length: int = None,
        pad_to_max_length: bool = False,
        preserved_tokens: Mapping[str, str] = None,
    ) -> None:

        valid_tokens = []
        token: str
        for token, cnt in counter.most_common(max_size):
            if cnt < min_freq:
                break
            if token in preserved_tokens.values():
                continue
            if casefold:
                token = token.casefold()
            valid_tokens.append(token)

        total_tokens = (
            list(OrderedDict.fromkeys(preserved_tokens.values())) + valid_tokens
        )
        token_to_index = {
            token: index for index, token in enumerate(total_tokens)
        }

        super().__init__(
            token_to_index=token_to_index,
            casefold=casefold,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            preserved_tokens=preserved_tokens,
        )


class Word2VecTokenizer(LookupTokenizer):
    def __init__(
        self,
        word2vec_file: str,
        casefold: bool = True,
        max_length: int = None,
        pad_to_max_length: bool = False,
        *,
        preserved_tokens: Mapping[str, str] = None,
    ) -> None:
        token_to_index = {}
        with open(word2vec_file) as fp:
            for lineno, line in tqdm.tqdm(
                enumerate(fp), "Loading word2vec words"
            ):
                word, *_ = str.split(line, sep=" ")
                token_to_index[word] = lineno

        super().__init__(
            token_to_index,
            casefold=casefold,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            preserved_tokens=preserved_tokens,
        )
