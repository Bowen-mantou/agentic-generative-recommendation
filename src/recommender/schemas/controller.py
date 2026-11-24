"""Adaptive Controller contracts.

The Controller is deterministic Python + LangGraph: it aggregates constraint status,
Judgment results, quality metrics and multi-dimensional budget, then decides
REPLAN / STOP / CLARIFY. Budget exhaustion is a forced stop and is tracked separately
from quality satisfaction (final-stage doc §4.4.1, §4.5; AGENTS.md §2.6).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator
from typing_extensions import Self

from recommender.schemas.common import (
    ContractModel,
    ControllerAction,
    FailedComponent,
    NonEmptyStr,
    StopReason,
)
from recommender.schemas.constraints import ConstraintSummary
from recommender.schemas.fusion import CandidatePool, FusedCandidate
from recommender.schemas.generation import RecommendationOutput
from recommender.schemas.judgment import JudgmentRecord
from recommender.schemas.memory import InteractionEvent
from recommender.schemas.plan import RecommendationPlan
from recommender.schemas.rerank import RankedProduct
from recommender.schemas.retrieval import RecallResult
from recommender.utils.errors import ContractError


class BudgetState(ContractModel):
    """Multi-dimensional budget owned by the Controller, never by the Planner."""

    remaining_replan_rounds: int = Field(default=2, ge=0)
    remaining_retrieval_calls: int = Field(default=6, ge=0)
    remaining_api_calls: int = Field(default=60, ge=0)
    tiger_candidate_budget: int = Field(default=50, ge=0)
    e5_candidate_budget: int = Field(default=100, ge=0)

    # --- read-only predicates -------------------------------------------------
    @property
    def replan_exhausted(self) -> bool:
        return self.remaining_replan_rounds <= 0

    @property
    def retrieval_exhausted(self) -> bool:
        return self.remaining_retrieval_calls <= 0

    @property
    def api_exhausted(self) -> bool:
        return self.remaining_api_calls <= 0

    @property
    def is_exhausted(self) -> bool:
        return self.replan_exhausted or self.retrieval_exhausted or self.api_exhausted

    def can_replan(self) -> bool:
        return not self.is_exhausted

    # --- immutable-ish consumption -------------------------------------------
    def _replace(self, **updates: int) -> BudgetState:
        payload = self.model_dump()
        payload.update(updates)
        return BudgetState(**payload)

    def consume_replan_round(self) -> BudgetState:
        if self.remaining_replan_rounds <= 0:
            raise ContractError("no replan rounds remain")
        return self._replace(remaining_replan_rounds=self.remaining_replan_rounds - 1)

    def consume_retrieval_calls(self, count: int = 1) -> BudgetState:
        if count <= 0:
            raise ContractError("retrieval consumption must be positive")
        if self.remaining_retrieval_calls < count:
            raise ContractError(
                f"not enough retrieval calls: need {count}, have {self.remaining_retrieval_calls}"
            )
        return self._replace(remaining_retrieval_calls=self.remaining_retrieval_calls - count)

    def consume_api_calls(self, count: int = 1) -> BudgetState:
        if count <= 0:
            raise ContractError("API consumption must be positive")
        if self.remaining_api_calls < count:
            raise ContractError(
                f"not enough API calls: need {count}, have {self.remaining_api_calls}"
            )
        return self._replace(remaining_api_calls=self.remaining_api_calls - count)


class QualityMetrics(ContractModel):
    """Aggregated evidence used by the Controller decision rules."""

    valid_product_count: int = Field(ge=0)
    required_count: int = Field(ge=1)
    hard_constraints_satisfied: bool
    mean_rerank_score: float | None = Field(default=None, allow_inf_nan=False)
    match_distribution: dict[str, int] = Field(default_factory=dict)
    evidence_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    unknown_attribute_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    diversity: float | None = Field(default=None, ge=0.0, le=1.0)
    # Decision D8: only Judgments that parsed, passed schema validation and are aligned
    # with the ranked candidate set count towards quality. Fewer than
    # ``QualityThresholds.min_judgments`` valid Judgments can never satisfy quality.
    valid_judgment_count: int = Field(default=0, ge=0)
    quality_satisfied: bool = False


class FailureInfo(ContractModel):
    """Structured record of an exhausted model/API call (decision D4).

    ``failed_component`` is limited to the three model-backed components so a trace
    reader can tell a Planner outage from a Judgment outage without parsing prose.
    The fields are optional on purpose: a failure raised by an older code path still
    yields a valid record, just with fewer details.
    """

    failed_component: FailedComponent
    error_type: str | None = None
    retryable: bool | None = None
    attempts: int | None = Field(default=None, ge=0)
    message: str | None = None


class ControllerDecision(ContractModel):
    action: ControllerAction
    stop_reason: StopReason | None = None
    quality_satisfied: bool = False
    budget_exhausted: bool = False
    reasons: list[str] = Field(default_factory=list)
    current_round: int = Field(default=0, ge=0)
    # Only populated for ``API_FAILED`` stops (decision D4); ``None`` keeps every
    # non-API decision byte-identical to the Phase 5 contract.
    failure: FailureInfo | None = None

    @model_validator(mode="after")
    def _action_semantics(self) -> Self:
        if self.action is ControllerAction.STOP:
            if self.stop_reason is None:
                raise ValueError("STOP requires a stop_reason")
        elif self.stop_reason is not None:
            raise ValueError("stop_reason is only valid for STOP")
        if self.stop_reason is StopReason.QUALITY_SATISFIED and not self.quality_satisfied:
            raise ValueError("quality_satisfied stop requires quality_satisfied=true")
        if self.stop_reason is StopReason.BUDGET_EXHAUSTED and not self.budget_exhausted:
            raise ValueError("budget stop requires budget_exhausted=true")
        if self.action is ControllerAction.REPLAN and self.budget_exhausted:
            raise ValueError("cannot replan after the budget is exhausted")
        if self.stop_reason is StopReason.API_FAILED:
            if self.failure is None:
                raise ValueError("api_failed stop requires failure details (failed_component)")
            if self.quality_satisfied:
                raise ValueError("api_failed stop cannot claim quality_satisfied")
        elif self.failure is not None:
            raise ValueError("failure details are only valid for an api_failed stop")
        return self


class RecommendationState(ContractModel):
    """Serializable LangGraph state (mirrors final-stage doc §4.5).

    LangGraph may persist this object between REPLAN rounds; every nested payload is a
    validated contract model so a resumed run cannot smuggle unvalidated data forward.
    """

    user_id: str | None = None
    user_request: NonEmptyStr
    user_memory: dict[str, Any] = Field(default_factory=dict)
    interaction_history: list[InteractionEvent] = Field(default_factory=list)
    session_history: list[dict[str, Any]] = Field(default_factory=list)

    current_plan: RecommendationPlan | None = None

    tiger_candidates: RecallResult | None = None
    e5_candidates: RecallResult | None = None
    fused_candidates: list[FusedCandidate] = Field(default_factory=list)
    candidate_pool: CandidatePool | None = None

    ranked_products: list[RankedProduct] = Field(default_factory=list)
    judgment_results: list[JudgmentRecord] = Field(default_factory=list)

    constraint_summary: ConstraintSummary | None = None
    valid_product_count: int = Field(default=0, ge=0)
    required_count: int = Field(default=0, ge=0)
    quality_metrics: QualityMetrics | None = None

    action: ControllerAction | None = None
    stop_reason: StopReason | None = None
    quality_satisfied: bool = False
    budget_exhausted: bool = False
    # Full Controller decision including reasons. The final-stage doc §4.5 state sketch
    # only lists action/stop_reason/quality_satisfied/budget_exhausted, which loses the
    # audit trail; keeping the validated decision object here makes the trace complete
    # without inventing parallel scalar fields.
    decision: ControllerDecision | None = None
    budget_state: BudgetState = BudgetState()
    current_round: int = Field(default=0, ge=0)
    max_replan_rounds: int = Field(default=2, ge=0)
    seen_query_hashes: list[str] = Field(default_factory=list)

    request_id: NonEmptyStr
    trace_id: NonEmptyStr

    selected_products: list[str] = Field(default_factory=list)
    recommendation: RecommendationOutput | None = None

    @model_validator(mode="after")
    def _selected_products_come_from_ranked(self) -> Self:
        if not self.selected_products:
            return self
        ranked_ids = {product.product_id for product in self.ranked_products}
        if ranked_ids:
            unknown = set(self.selected_products) - ranked_ids
            if unknown:
                raise ValueError(f"selected_products not in ranked_products: {sorted(unknown)}")
        if len(set(self.selected_products)) != len(self.selected_products):
            raise ValueError("selected_products must be unique")
        return self
