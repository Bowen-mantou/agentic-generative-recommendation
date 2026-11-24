"""DeepSeek per-product structured judgment (Phase 5)."""

from __future__ import annotations

from recommender.judgment.judge import (
    DEFAULT_BATCH_SIZE,
    Judge,
    JudgmentBatchError,
    available_evidence_sources,
)
from recommender.judgment.prompts import JUDGMENT_PROMPT_VERSION

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "JUDGMENT_PROMPT_VERSION",
    "Judge",
    "JudgmentBatchError",
    "available_evidence_sources",
]
