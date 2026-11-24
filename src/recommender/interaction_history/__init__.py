"""Interaction History: SQLite `InteractionEvent` store for TIGER sequences (Phase 4/6)."""

from __future__ import annotations

from recommender.interaction_history.build import (
    ReviewBuildStats,
    build_interaction_history,
    load_catalog_product_ids,
    normalize_rating,
    normalize_timestamp,
    parse_review_record,
    sequence_length_summary,
)
from recommender.interaction_history.store import InsertStats, InteractionStore

__all__ = [
    "InsertStats",
    "InteractionStore",
    "ReviewBuildStats",
    "build_interaction_history",
    "load_catalog_product_ids",
    "normalize_rating",
    "normalize_timestamp",
    "parse_review_record",
    "sequence_length_summary",
]
