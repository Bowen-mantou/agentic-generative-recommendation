"""Recommendation Generator contracts.

The Generator only expresses Controller-approved products; every reason must be
traceable to catalog facts or Judgment output (final-stage doc §5.4, AGENTS.md §3).
"""

from __future__ import annotations

from pydantic import Field, model_validator
from typing_extensions import Self

from recommender.schemas.common import ContractModel, NonEmptyStr, StopReason


class RecommendationItem(ContractModel):
    rank: int = Field(ge=1)
    product_id: NonEmptyStr
    title: str = ""
    price: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    reasons: list[str] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class RecommendationOutput(ContractModel):
    """Final user-facing recommendation payload.

    ``recommendations`` may contain fewer than K items: padding the list with
    unverified products is forbidden.
    """

    request_id: NonEmptyStr
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    model: str | None = None
    stop_reason: StopReason | None = None
    quality_satisfied: bool | None = None
    budget_exhausted: bool | None = None
    is_synthetic: bool = False

    @model_validator(mode="after")
    def _ranking_is_well_formed(self) -> Self:
        expected = list(range(1, len(self.recommendations) + 1))
        if [item.rank for item in self.recommendations] != expected:
            raise ValueError("recommendation ranks must be contiguous from 1")
        ids = [item.product_id for item in self.recommendations]
        if len(set(ids)) != len(ids):
            raise ValueError("a product may be recommended only once")
        return self

    @model_validator(mode="after")
    def _stop_semantics_are_consistent(self) -> Self:
        if self.stop_reason is StopReason.BUDGET_EXHAUSTED and self.budget_exhausted is False:
            raise ValueError("stop_reason=budget_exhausted requires budget_exhausted=true")
        if self.stop_reason is StopReason.QUALITY_SATISFIED and self.quality_satisfied is False:
            raise ValueError("stop_reason=quality_satisfied requires quality_satisfied=true")
        if not self.recommendations and self.stop_reason is None:
            raise ValueError("an empty recommendation must state why it is empty")
        return self

    @property
    def product_ids(self) -> list[str]:
        return [item.product_id for item in self.recommendations]
