"""Deterministic whitespace word-level tokenizer for DR01 captions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
MAX_LENGTH = 7


def normalize_and_split(text: str) -> list[str]:
    """Lowercase, normalize whitespace, and split only on whitespace."""
    return text.lower().split()


def build_train_vocabulary(captions: Iterable[str]) -> tuple[dict[str, int], Counter[str]]:
    """Build deterministic IDs and frequencies from TRAIN captions only."""
    frequencies: Counter[str] = Counter()
    for caption in captions:
        frequencies.update(normalize_and_split(caption))
    lexical_tokens = sorted(frequencies)
    token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    token_to_id.update(
        {token: len(SPECIAL_TOKENS) + index for index, token in enumerate(lexical_tokens)}
    )
    return token_to_id, frequencies


@dataclass(frozen=True)
class EncodedText:
    tokens: list[str]
    token_ids: list[int]
    attention_mask: list[int]


class WhitespaceTokenizer:
    """Fixed-vocabulary tokenizer with BOS/EOS and optional right padding."""

    def __init__(self, token_to_id: dict[str, int], max_length: int = MAX_LENGTH) -> None:
        if tuple(token_to_id.get(token) for token in SPECIAL_TOKENS) != (0, 1, 2, 3):
            raise ValueError("Special token IDs must be <pad>=0, <bos>=1, <eos>=2, <unk>=3")
        self.token_to_id = dict(token_to_id)
        self.max_length = max_length

    def encode(self, text: str) -> EncodedText:
        lexical_tokens = normalize_and_split(text)
        serialized_tokens = ["<bos>", *lexical_tokens, "<eos>"]
        if len(serialized_tokens) > self.max_length:
            raise ValueError(
                f"Serialized sequence length {len(serialized_tokens)} exceeds {self.max_length}"
            )
        padding = self.max_length - len(serialized_tokens)
        tokens = [*serialized_tokens, *(["<pad>"] * padding)]
        token_ids = [self.token_to_id.get(token, UNK_ID) for token in tokens]
        attention_mask = [1] * len(serialized_tokens) + [0] * padding
        return EncodedText(tokens=tokens, token_ids=token_ids, attention_mask=attention_mask)

