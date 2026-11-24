"""DeepSeek per-product structured judgment (Phase 5)."""

from __future__ import annotations

from recommender.judgment.evidence import (
    DEFAULT_CHARS_PER_TOKEN,
    DEFAULT_DESCRIPTION_MAX_CHARS,
    DEFAULT_MAX_DETAIL_KEYS,
    DEFAULT_MAX_TOTAL_CHARS,
    TRUNCATION_MARKER,
    CatalogLookup,
    DroppedDetail,
    EvidenceBuildResult,
    EvidenceStats,
    JudgmentEvidenceBuilder,
)
from recommender.judgment.judge import (
    DEFAULT_BATCH_SIZE,
    Judge,
    JudgmentBatchError,
    RankedJudgment,
    available_evidence_sources,
)
from recommender.judgment.prompts import JUDGMENT_PROMPT_VERSION

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_DESCRIPTION_MAX_CHARS",
    "DEFAULT_MAX_DETAIL_KEYS",
    "DEFAULT_MAX_TOTAL_CHARS",
    "JUDGMENT_PROMPT_VERSION",
    "TRUNCATION_MARKER",
    "CatalogLookup",
    "DroppedDetail",
    "EvidenceBuildResult",
    "EvidenceStats",
    "Judge",
    "JudgmentBatchError",
    "JudgmentEvidenceBuilder",
    "RankedJudgment",
    "available_evidence_sources",
]
