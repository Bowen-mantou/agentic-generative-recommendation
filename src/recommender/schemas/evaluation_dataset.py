"""Evaluation dataset contracts for the temporal / profile / query protocol.

Target labels are deliberately a separate model: the sample file never carries the
ground-truth product, so generators and auditors cannot receive it by accident.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from recommender.schemas.common import ContractModel, EvalSplit, NonEmptyStr

Confidence = Literal["high", "medium", "low"]
FactKind = Literal["category", "brand", "attribute", "negative"]
QueryStatus = Literal["generated", "insufficient_profile", "failed"]
AuditStatus = Literal["PASS", "REVIEW", "FAIL", "ERROR"]


class TemporalSample(ContractModel):
    """One pre-cutoff history that predicts the user's next interaction.

    ``history_product_ids`` and ``history_event_ids`` are parallel lists. The target is
    stored only in :class:`TargetLabel` and is never a field of this model.
    """

    sample_id: NonEmptyStr
    user_id: NonEmptyStr
    split: EvalSplit
    cutoff_timestamp: int = Field(ge=0)
    history_product_ids: list[NonEmptyStr] = Field(min_length=2)
    history_event_ids: list[NonEmptyStr] = Field(min_length=2)
    is_synthetic: bool = False
    dataset_version: NonEmptyStr

    @model_validator(mode="after")
    def _parallel_history(self) -> Self:
        if len(self.history_product_ids) != len(self.history_event_ids):
            raise ValueError("history_product_ids and history_event_ids must be aligned")
        if self.is_synthetic and self.split in (EvalSplit.VALIDATION, EvalSplit.TEST):
            raise ValueError("synthetic samples cannot enter validation/test evaluation")
        return self


class TargetLabel(ContractModel):
    """Offline label, isolated from all LLM/profile construction inputs."""

    sample_id: NonEmptyStr
    target_product_id: NonEmptyStr
    target_event_id: NonEmptyStr
    target_timestamp: int = Field(ge=0)
    target_title: str = ""


class ProfileFact(ContractModel):
    fact_id: NonEmptyStr
    kind: FactKind
    value: NonEmptyStr
    support_count: int = Field(ge=1)
    positive_count: int = Field(default=0, ge=0)
    negative_count: int = Field(default=0, ge=0)
    source_event_ids: list[NonEmptyStr] = Field(min_length=1)
    confidence: Confidence


class UserProfile(ContractModel):
    profile_id: NonEmptyStr
    sample_id: NonEmptyStr
    user_id: NonEmptyStr
    cutoff_timestamp: int = Field(ge=0)
    profile_version: NonEmptyStr
    history_length: int = Field(ge=0)
    category_interests: list[ProfileFact] = Field(default_factory=list)
    brand_interests: list[ProfileFact] = Field(default_factory=list)
    attribute_interests: list[ProfileFact] = Field(default_factory=list)
    negative_signals: list[ProfileFact] = Field(default_factory=list)
    unknown_preferences: list[str] = Field(default_factory=list)

    def all_facts(self) -> dict[str, ProfileFact]:
        facts: dict[str, ProfileFact] = {}
        for group in (
            self.category_interests,
            self.brand_interests,
            self.attribute_interests,
            self.negative_signals,
        ):
            for fact in group:
                facts[fact.fact_id] = fact
        return facts


class SyntheticQuery(ContractModel):
    """One generated shopping query. ``used_fact_ids`` must point at profile facts."""

    query_id: NonEmptyStr
    sample_id: NonEmptyStr
    query: str | None = None
    used_fact_ids: list[NonEmptyStr] = Field(default_factory=list)
    status: QueryStatus
    is_synthetic: bool = True
    generator_model: NonEmptyStr
    prompt_version: NonEmptyStr
    generation_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _status_shape(self) -> Self:
        if self.status == "generated" and not (self.query and self.query.strip()):
            raise ValueError("generated queries require non-empty query text")
        if self.status != "generated" and self.query:
            raise ValueError("non-generated queries must not carry query text")
        if not self.is_synthetic:
            raise ValueError("LLM-generated evaluation queries must be marked synthetic")
        return self


class QueryAudit(ContractModel):
    audit_id: NonEmptyStr
    query_id: NonEmptyStr
    sample_id: NonEmptyStr
    groundedness: Literal["pass", "fail"]
    hallucination: bool
    contradiction: bool
    query_quality: Literal["good", "review", "bad"]
    unsupported_claims: list[str] = Field(default_factory=list)
    supported_fact_ids: list[NonEmptyStr] = Field(default_factory=list)
    audit_status: AuditStatus
    reason: str = ""
    auditor_model: NonEmptyStr
    audit_prompt_version: NonEmptyStr
    audit_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fail_is_explicit(self) -> Self:
        if self.audit_status == "PASS" and (self.hallucination or self.contradiction):
            raise ValueError("PASS is incompatible with hallucination/contradiction")
        return self


class QueryValidation(ContractModel):
    query_id: NonEmptyStr
    sample_id: NonEmptyStr
    passed: bool
    checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    target_id_scanned: bool = False
    target_title_scanned: bool = False
