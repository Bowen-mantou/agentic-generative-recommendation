"""User Memory and Interaction History contracts.

Two separate stores, never merged (AGENTS.md §2.2):

* Mem0 semantic memory: long-term preferences, recent interests, conflicts.
* SQLite ``InteractionEvent``: timestamped product events used to build TIGER
  sequences. Amazon Reviews 2023 yields review/rating *proxy* events only.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from recommender.schemas.common import (
    ContractModel,
    EventType,
    MemoryAction,
    MemoryPolarity,
    MemorySourceType,
    MemoryStatus,
    MemoryTargetStore,
    MemoryType,
    NonEmptyStr,
)


class MemorySource(ContractModel):
    source_type: MemorySourceType
    source_ids: list[str] = Field(default_factory=list)


class MemoryRecord(ContractModel):
    """Semantic memory record (stored in Mem0, described here for validation)."""

    memory_id: NonEmptyStr
    user_id: NonEmptyStr
    memory_type: MemoryType
    content: NonEmptyStr
    category: str | None = None
    polarity: MemoryPolarity = MemoryPolarity.NEUTRAL
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: MemorySource
    evidence_count: int = Field(default=0, ge=0)
    created_at: datetime
    last_observed_at: datetime
    is_synthetic: bool
    status: MemoryStatus = MemoryStatus.ACTIVE

    @model_validator(mode="after")
    def _source_ids_match_evidence(self) -> Self:
        explicit_feedback = self.source.source_type is MemorySourceType.EXPLICIT_USER_FEEDBACK
        if self.evidence_count > 0 and explicit_feedback and not self.source.source_ids:
            raise ValueError("explicit feedback memories must cite source_ids")
        return self


class InteractionEvent(ContractModel):
    """A timestamped product interaction used for TIGER sequences."""

    event_id: NonEmptyStr
    user_id: NonEmptyStr
    product_id: NonEmptyStr
    event_type: EventType
    rating: int | None = Field(default=None, ge=1, le=5)
    timestamp: int = Field(ge=0, description="Unix timestamp in seconds (UTC)")
    source: NonEmptyStr
    source_id: str | None = None
    # Deliberately required: synthetic events must always be marked, so they can be
    # filtered from TIGER/evaluation (AGENTS.md §2.2).
    is_synthetic: bool

    @model_validator(mode="after")
    def _rating_only_for_review_or_rating(self) -> Self:
        rating_events = {EventType.REVIEW, EventType.RATING}
        if self.rating is not None and self.event_type not in rating_events:
            raise ValueError("rating is only meaningful for review/rating events")
        return self

    @field_validator("source_id")
    @classmethod
    def _blank_source_id_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class MemoryUpdateOperation(ContractModel):
    """One write decided by the Memory Update rule engine."""

    memory_type: MemoryType
    action: MemoryAction
    content: NonEmptyStr
    source: MemorySourceType
    target_store: MemoryTargetStore
    is_synthetic: bool

    @model_validator(mode="after")
    def _store_matches_content_kind(self) -> Self:
        episodic = self.memory_type is MemoryType.EPISODIC
        if episodic and self.target_store is not MemoryTargetStore.SQLITE_INTERACTION_HISTORY:
            raise ValueError("episodic events must be stored as interaction events")
        return self

    @model_validator(mode="after")
    def _inference_never_becomes_fact(self) -> Self:
        # Model-inferred preferences may only be written as evidence-backed semantic
        # candidates, never as confirmed user facts without explicit feedback.
        inference = self.source is MemorySourceType.SYSTEM_INFERENCE
        long_term = self.memory_type is MemoryType.LONG_TERM_PREFERENCE
        if inference and long_term:
            raise ValueError("system inference cannot create a long-term preference memory")
        return self
