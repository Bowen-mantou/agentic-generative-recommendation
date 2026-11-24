"""Judgment contracts (DeepSeek API, per-product structured judgment).

Judgment answers "how well does this product fit and what is the evidence?" — it never
decides REPLAN/STOP, never re-runs retrieval and never re-checks numeric hard
constraints (AGENTS.md §3, final-stage doc §3.7).
"""

from __future__ import annotations

from pydantic import Field, model_validator
from typing_extensions import Self

from recommender.schemas.common import (
    ConstraintStatus,
    ContractModel,
    EvidenceStatus,
    MatchAssessment,
    NonEmptyStr,
    PreferenceFitLevel,
    UncertaintyLevel,
)


class JudgmentCandidate(ContractModel):
    """Product view handed to Judgment: catalog facts plus the upstream constraint state.

    ``constraint_status`` is read-only context produced by the single Python Constraint
    Evaluator. ``None`` keeps decision Q3 intact: "no hard constraint configured" is not
    the same as ``pass``, and Judgment must never upgrade ``unknown``/``None`` to a
    satisfied hard constraint.
    """

    product_id: NonEmptyStr
    title: str = ""
    price: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    description: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    constraint_status: ConstraintStatus | None = None


class PreferenceFit(ContractModel):
    level: PreferenceFitLevel
    matched_preferences: list[str] = Field(default_factory=list)
    unmatched_preferences: list[str] = Field(default_factory=list)
    unknown_preferences: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _preference_sets_are_disjoint(self) -> Self:
        matched = set(self.matched_preferences)
        unmatched = set(self.unmatched_preferences)
        unknown = set(self.unknown_preferences)
        if matched & unmatched or matched & unknown or unmatched & unknown:
            raise ValueError("a preference may appear in only one of matched/unmatched/unknown")
        return self


class AttributeEvidence(ContractModel):
    """Whether the catalog actually supports an attribute, and from which field."""

    status: EvidenceStatus
    source: str | None = None

    @model_validator(mode="after")
    def _evidence_needs_a_source(self) -> Self:
        if (
            self.status in (EvidenceStatus.SUPPORTED, EvidenceStatus.CONTRADICTED)
            and not self.source
        ):
            raise ValueError(f"status={self.status.value} requires a catalog source field")
        if self.status is EvidenceStatus.UNKNOWN and self.source is not None:
            raise ValueError("status=unknown must not cite a source")
        return self


class JudgmentLLMOutput(ContractModel):
    """Exactly what the Judgment model is allowed to return (validated strictly)."""

    product_id: NonEmptyStr
    preference_fit: PreferenceFit
    attribute_evidence: dict[str, AttributeEvidence] = Field(default_factory=dict)
    match_assessment: MatchAssessment
    uncertainty: UncertaintyLevel
    warnings: list[str] = Field(default_factory=list)


class JudgmentRecord(ContractModel):
    """Validated Judgment output plus non-LLM execution metadata."""

    result: JudgmentLLMOutput
    model: NonEmptyStr
    prompt_version: NonEmptyStr
    latency_ms: float | None = Field(default=None, ge=0.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    # Number of chat attempts that produced this batch (includes a bounded
    # reasoning-truncation compatibility retry); useful for cost/retry auditing.
    attempts: int = Field(default=1, ge=1)
    is_synthetic: bool = False


class JudgmentBatch(ContractModel):
    request_id: NonEmptyStr
    records: list[JudgmentRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_record_per_product(self) -> Self:
        ids = [record.result.product_id for record in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError("each product must have exactly one independent judgment record")
        return self
