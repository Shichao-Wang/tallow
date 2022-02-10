# import os
from typing import Dict, List

import numpy
import transformers

from . import tokenizer

# os.environ["TOKENIZERS_PARALLELISM"] = "true"


class HuggingfaceTokenizer(tokenizer.Tokenizer):
    def __init__(
        self,
        pretrained_path: str,
        add_special_tokens: bool = True,
        max_length: int = None,
        pad_to_max_length: bool = False,
        *,
        init_kwargs: Dict = None
    ) -> None:
        super().__init__()
        init_kwargs = init_kwargs or {}
        self._hf_tokenizer: transformers.PreTrainedTokenizer = (
            transformers.AutoTokenizer.from_pretrained(
                pretrained_path, **init_kwargs
            )
        )
        self._add_special_tokens = add_special_tokens
        self._max_length = max_length
        self._pad_to_max_length = pad_to_max_length

    @property
    def vocab_size(self) -> int:
        return self._hf_tokenizer.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        return self._hf_tokenizer.get_vocab()

    def tokenize(
        self,
        words: List[str],
    ) -> List[str]:
        tokens = self._hf_tokenizer.tokenize(
            words,
            is_split_into_words=True,
            add_special_tokens=self._add_special_tokens,
            truncation=self._max_length is not None,
            max_length=self._max_length,
            padding="max_length" if self._pad_to_max_length else False,
        )
        return tokens

    def __call__(
        self,
        words: List[str],
    ) -> numpy.ndarray:
        tokenize_outputs = self._hf_tokenizer(
            words,
            is_split_into_words=True,
            truncation=self._max_length is not None,
            max_length=self._max_length,
            padding="max_length" if self._pad_to_max_length else False,
            add_special_tokens=self._add_special_tokens,
            return_tensors="np",
        )
        return tokenize_outputs["input_ids"][0]

    def to_token(self, index: int) -> str:
        return self._hf_tokenizer.convert_ids_to_tokens(index)

    def to_id(self, token: str) -> int:
        return self._hf_tokenizer.convert_tokens_to_ids(token)

    def to_tokens(self, indexes: List[int]) -> List[str]:
        return self._hf_tokenizer.convert_ids_to_tokens(indexes)

    def to_ids(self, tokens: List[str]) -> List[int]:
        return self._hf_tokenizer.convert_tokens_to_ids(tokens)

    # special tokens
    @property
    def pad_tok(self):
        return self._hf_tokenizer.pad_token

    @property
    def unk_tok(self):
        return self._hf_tokenizer.unk_token

    @property
    def pad_id(self):
        return self._hf_tokenizer.pad_token_id

    @property
    def unk_id(self):
        return self._hf_tokenizer.unk_token_id


def token_to_words(tokens: List[str]):
    current_word_id = -1
    ret = []
    for token in tokens:
        if not token.startswith("##"):
            current_word_id += 1
        ret.append(current_word_id)

    return ret
