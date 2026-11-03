"""Token counting with the GPT-4o tokenizer (o200k_base).

We count with the same tokenizer that the cost assumption (GPT-4o pricing) is
based on, so that context-budget tables and cost estimates are consistent.
"""
from __future__ import annotations

from functools import lru_cache

import tiktoken

from . import config


@lru_cache(maxsize=1)
def _enc():
    return tiktoken.get_encoding(config.TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_enc().encode(text, disallowed_special=()))


def count_messages(messages: list[dict]) -> int:
    # ~4 tokens of chat-format overhead per message (OpenAI cookbook heuristic)
    return sum(count_tokens(m.get("content") or "") + 4 for m in messages) + 2


def truncate_tokens(text: str, max_tokens: int) -> str:
    ids = _enc().encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return text
    return _enc().decode(ids[:max_tokens])


def encode(text: str) -> list[int]:
    return _enc().encode(text, disallowed_special=())


def decode(ids: list[int]) -> str:
    return _enc().decode(ids)
