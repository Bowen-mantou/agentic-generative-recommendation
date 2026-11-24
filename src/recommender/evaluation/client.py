"""Minimal client protocol shared by the query generator and auditor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from recommender.api.deepseek import JSONCompletion


class ChatJSONClient(Protocol):
    """Structural type satisfied by :class:`recommender.api.deepseek.DeepSeekClient`."""

    @property
    def model(self) -> str: ...

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_json_mode: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> JSONCompletion: ...
