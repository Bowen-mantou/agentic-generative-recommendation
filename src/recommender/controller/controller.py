"""Deterministic Adaptive Controller: REPLAN / STOP / CLARIFY.

No Controller LLM (AGENTS.md §3, final-stage doc §4). The Controller only reads validated
upstream state — the Python Constraint Evaluator verdict, the Reranker ranking and the
Judgment records — and applies explicit, testable rules:

1. a plan that requests clarification yields CLARIFY;
2. an API failure is an explicit degraded STOP, never an empty success;
3. quality satisfied by this round's validated evidence -> STOP ``QUALITY_SATISFIED``
   (with an explicit note while ``controller.quality.calibrated`` is false);
4. otherwise any exhausted budget dimension is a **forced** STOP and is never reported
   as ``quality_satisfied`` (AGENTS.md §2.6);
5. zero valid candidates triggers REPLAN while budget/rounds allow, otherwise
   ``NO_VALID_CANDIDATES``;
6. insufficient quality triggers REPLAN while the loop guard allows, else STOP
   ``MAX_ROUNDS_REACHED``.

Duplicate-query detection lives next to the decision rules but is executed by the
workflow *after* the Replanner returns a new plan (the Controller cannot inspect a plan
that does not exist yet).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

from recommender.observability.logging import get_logger
from recommender.schemas.common import (
    ControllerAction,
    EvidenceStatus,
    MatchAssessment,
    StopReason,
)
from recommender.schemas.constraints import ConstraintSummary
from recommender.schemas.controller import (
    BudgetState,
    ControllerDecision,
    QualityMetrics,
    RecommendationState,
)
from recommender.schemas.judgment import JudgmentRecord
from recommender.schemas.plan import RecommendationPlan
from recommender.schemas.rerank import RankedProduct

if TYPE_CHECKING:
    from recommender.config import ControllerConfig, QualityThresholds

_LOGGER = get_logger(__name__)


def normalize_query(value: str) -> str:
    """Canonical form used for duplicate-query detection."""
    return " ".join(value.split()).casefold()


def query_hash(value: str) -> str:
    """Stable hash of one normalized E5 query (State stores hashes, not raw queries)."""
    return hashlib.sha256(normalize_query(value).encode("utf-8")).hexdigest()


def plan_query_hashes(plan: RecommendationPlan) -> list[str]:
    """Hashes of the E5 queries this plan would execute (empty for TIGER-only plans)."""
    return [query_hash(query) for query in plan.retrieval.e5_faiss.queries]


def duplicate_queries(plan: RecommendationPlan, seen_query_hashes: Sequence[str]) -> list[str]:
    """Return the plan's query hashes that were already executed in an earlier round."""
    seen = set(seen_query_hashes)
    return [digest for digest in plan_query_hashes(plan) if digest in seen]


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_quality(
    *,
    constraint_summary: ConstraintSummary | None,
    ranked_products: Sequence[RankedProduct],
    judgment_results: Sequence[JudgmentRecord],
    required_count: int,
) -> QualityMetrics:
    """Aggregate constraint/rerank/Judgment evidence into :class:`QualityMetrics`.

    Definitions chosen here (documented as uncalibrated in docs/phase5):

    * ``valid_product_count`` comes from the Python Constraint Evaluator only —
      Judgment never overrides a hard constraint;
    * ``hard_constraints_satisfied`` means "no evaluated product failed or left a hard
      constraint unknown and at least one product is valid"; it does **not** claim the
      checks were configured/verified (decision Q3 keeps ``not_configured_count``);
    * ``evidence_coverage`` = share of judged products with at least one supported or
      contradicted attribute; ``None`` means nothing was judged and therefore quality
      cannot be satisfied;
    * ``unknown_attribute_ratio`` = unknown evidence entries / all evidence entries;
    * ``diversity`` is not computed in Phase 5 (the reranker contract carries no item
      similarity), so ``min_diversity`` must stay null until a real metric exists.
    """
    valid_product_count = (
        constraint_summary.valid_product_count if constraint_summary is not None else 0
    )
    hard_constraints_satisfied = bool(
        constraint_summary is not None
        and constraint_summary.fail_count == 0
        and constraint_summary.unknown_count == 0
        and constraint_summary.valid_product_count > 0
    )
    match_distribution = {
        MatchAssessment.DIRECT_MATCH.value: 0,
        MatchAssessment.PARTIAL_MATCH.value: 0,
        MatchAssessment.NO_MATCH.value: 0,
    }
    for record in judgment_results:
        key = record.result.match_assessment.value
        match_distribution[key] = match_distribution.get(key, 0) + 1

    evidence_entries = 0
    unknown_entries = 0
    products_with_evidence = 0
    for record in judgment_results:
        evidences = list(record.result.attribute_evidence.values())
        evidence_entries += len(evidences)
        unknown_entries += sum(1 for item in evidences if item.status is EvidenceStatus.UNKNOWN)
        if any(item.status is not EvidenceStatus.UNKNOWN for item in evidences):
            products_with_evidence += 1

    return QualityMetrics(
        valid_product_count=valid_product_count,
        required_count=required_count,
        hard_constraints_satisfied=hard_constraints_satisfied,
        mean_rerank_score=_mean([product.rerank_score for product in ranked_products]),
        match_distribution=match_distribution,
        evidence_coverage=(
            products_with_evidence / len(judgment_results) if judgment_results else None
        ),
        unknown_attribute_ratio=(
            unknown_entries / evidence_entries if evidence_entries else None
        ),
        diversity=None,
        quality_satisfied=False,
    )


def evaluate_quality(
    metrics: QualityMetrics,
    thresholds: QualityThresholds,
) -> QualityMetrics:
    """Return a copy of ``metrics`` with ``quality_satisfied`` applied.

    Only configured thresholds are enforced. Unconfigured thresholds stay ``None`` and
    are skipped instead of being invented (AGENTS.md §8.8).
    """
    satisfied = True
    if not metrics.hard_constraints_satisfied:
        satisfied = False
    if metrics.valid_product_count < thresholds.required_count:
        satisfied = False
    # Quality must be based on actual Judgment results: no judged product cannot be a
    # quality pass (the workflow always judges before calling the Controller).
    if metrics.evidence_coverage is None:
        satisfied = False
    if thresholds.min_mean_rerank_score is not None and (
        metrics.mean_rerank_score is None
        or metrics.mean_rerank_score < thresholds.min_mean_rerank_score
    ):
        satisfied = False
    if thresholds.max_unknown_attribute_ratio is not None and (
        metrics.unknown_attribute_ratio is not None
        and metrics.unknown_attribute_ratio > thresholds.max_unknown_attribute_ratio
    ):
        satisfied = False
    if thresholds.min_diversity is not None and (
        metrics.diversity is None or metrics.diversity < thresholds.min_diversity
    ):
        satisfied = False
    return metrics.model_copy(update={"quality_satisfied": satisfied})


class Controller:
    """Deterministic decision engine over a validated :class:`RecommendationState`."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        if config is None:
            from recommender.config import ControllerConfig as _ControllerConfig

            config = _ControllerConfig()
        self._config = config

    @property
    def config(self) -> ControllerConfig:
        return self._config

    def initial_budget(self) -> BudgetState:
        """Fresh multi-dimensional budget from ``configs/controller.yaml``."""
        budget = self._config.budget
        return BudgetState(
            remaining_replan_rounds=budget.remaining_replan_rounds,
            remaining_retrieval_calls=budget.remaining_retrieval_calls,
            remaining_api_calls=budget.remaining_api_calls,
        )

    def summarize(self, state: RecommendationState) -> QualityMetrics:
        """Aggregate the current state into quality metrics (pure, no LLM)."""
        metrics = summarize_quality(
            constraint_summary=state.constraint_summary,
            ranked_products=state.ranked_products,
            judgment_results=state.judgment_results,
            required_count=self._config.quality.required_count,
        )
        return evaluate_quality(metrics, self._config.quality)

    def decide(
        self,
        state: RecommendationState,
        *,
        api_failure: str | None = None,
    ) -> ControllerDecision:
        """Apply the decision rules and return a validated decision."""
        metrics = state.quality_metrics or self.summarize(state)
        budget = state.budget_state
        plan = state.current_plan
        round_number = state.current_round
        max_rounds = self._config.loop_guard.max_rounds
        reasons: list[str] = []

        if plan is not None and plan.clarification.required:
            return ControllerDecision(
                action=ControllerAction.CLARIFY,
                quality_satisfied=False,
                budget_exhausted=False,
                reasons=[
                    "plan requires user clarification: "
                    + str(plan.clarification.question or "missing clarification question")
                ],
                current_round=round_number,
            )

        if api_failure is not None:
            stop_reason = (
                StopReason.NO_VALID_CANDIDATES
                if metrics.valid_product_count == 0
                else StopReason.RETRIEVAL_DEGRADED
            )
            reasons.append(f"API failure: {api_failure}")
            reasons.append("degraded stop; no result was fabricated")
            decision = ControllerDecision(
                action=ControllerAction.STOP,
                stop_reason=stop_reason,
                quality_satisfied=False,
                budget_exhausted=False,
                reasons=reasons,
                current_round=round_number,
            )
            return self._log_decision(decision, metrics)

        # Quality is judged from this round's validated evidence. It is checked before
        # the "forced budget stop" rule so a good result reached on the last allowed
        # round is not mislabelled as a budget stop; the reverse direction (a budget
        # stop claiming quality) remains impossible because this branch requires real
        # quality_satisfied evidence.
        if metrics.quality_satisfied:
            reasons.append(
                f"quality satisfied: valid={metrics.valid_product_count} "
                f">= required={metrics.required_count}"
            )
            reasons.append(
                "hard constraints satisfied with no unknown/failed check"
                if metrics.hard_constraints_satisfied
                else "hard constraint status incomplete"
            )
            if not self._config.quality.calibrated:
                reasons.append(
                    "quality thresholds are uncalibrated "
                    "(config.controller.quality.calibrated=false)"
                )
            if budget.is_exhausted:
                reasons.append(
                    "a budget dimension was already at its limit, but quality was met "
                    "before any further budget was required"
                )
            return self._log_decision(
                ControllerDecision(
                    action=ControllerAction.STOP,
                    stop_reason=StopReason.QUALITY_SATISFIED,
                    quality_satisfied=True,
                    budget_exhausted=False,
                    reasons=reasons,
                    current_round=round_number,
                ),
                metrics,
            )

        if budget.is_exhausted:
            exhausted = []
            if budget.replan_exhausted:
                exhausted.append("replan rounds")
            if budget.retrieval_exhausted:
                exhausted.append("retrieval calls")
            if budget.api_exhausted:
                exhausted.append("API calls")
            reasons.append("budget exhausted (" + ", ".join(exhausted) + "); forced stop")
            reasons.append("budget exhaustion is not a quality pass")
            decision = ControllerDecision(
                action=ControllerAction.STOP,
                stop_reason=StopReason.BUDGET_EXHAUSTED,
                quality_satisfied=False,
                budget_exhausted=True,
                reasons=reasons,
                current_round=round_number,
            )
            return self._log_decision(decision, metrics)

        if metrics.valid_product_count == 0:
            reasons.append("no product passes the configured hard constraints")
            if self._can_replan(state, budget, max_rounds):
                return self._log_decision(
                    ControllerDecision(
                        action=ControllerAction.REPLAN,
                        quality_satisfied=False,
                        budget_exhausted=False,
                        reasons=reasons,
                        current_round=round_number,
                    ),
                    metrics,
                )
            return self._log_decision(
                ControllerDecision(
                    action=ControllerAction.STOP,
                    stop_reason=StopReason.NO_VALID_CANDIDATES,
                    quality_satisfied=False,
                    budget_exhausted=False,
                    reasons=reasons,
                    current_round=round_number,
                ),
                metrics,
            )

        reasons.append(
            f"quality insufficient: valid={metrics.valid_product_count}, "
            f"required={metrics.required_count}, "
            f"unknown_attribute_ratio={metrics.unknown_attribute_ratio}"
        )
        if not metrics.hard_constraints_satisfied:
            reasons.append("hard constraints are not fully satisfied/verified")
        if metrics.match_distribution:
            reasons.append(f"match distribution: {metrics.match_distribution}")
        if self._can_replan(state, budget, max_rounds):
            return self._log_decision(
                ControllerDecision(
                    action=ControllerAction.REPLAN,
                    quality_satisfied=False,
                    budget_exhausted=False,
                    reasons=reasons,
                    current_round=round_number,
                ),
                metrics,
            )

        reasons.append(
            f"loop guard reached: round={round_number}, max_rounds={max_rounds}, "
            f"remaining_replan_rounds={budget.remaining_replan_rounds}"
        )
        return self._log_decision(
            ControllerDecision(
                action=ControllerAction.STOP,
                stop_reason=StopReason.MAX_ROUNDS_REACHED,
                quality_satisfied=False,
                budget_exhausted=False,
                reasons=reasons,
                current_round=round_number,
            ),
            metrics,
        )

    def budget_stop(
        self,
        state: RecommendationState,
        *,
        dimension: str,
        reason: str,
    ) -> ControllerDecision:
        """Forced budget STOP when an upcoming node cannot be afforded.

        Used by the workflow when a *partial* budget (e.g. 1 retrieval call left for a
        2-path plan) cannot pay for the next node. The decision is still a budget stop,
        not a quality pass.
        """
        return ControllerDecision(
            action=ControllerAction.STOP,
            stop_reason=StopReason.BUDGET_EXHAUSTED,
            quality_satisfied=False,
            budget_exhausted=True,
            reasons=[f"{dimension} budget cannot afford the next node: {reason}"],
            current_round=state.current_round,
        )

    # -- internals -----------------------------------------------------------
    def _can_replan(
        self,
        state: RecommendationState,
        budget: BudgetState,
        max_rounds: int,
    ) -> bool:
        return (
            budget.remaining_replan_rounds > 0
            and state.current_round < max_rounds
            and state.current_round <= state.max_replan_rounds
        )

    @staticmethod
    def _log_decision(
        decision: ControllerDecision,
        metrics: QualityMetrics,
    ) -> ControllerDecision:
        _LOGGER.info(
            "controller decision",
            extra={
                "action": decision.action.value,
                "stop_reason": decision.stop_reason.value if decision.stop_reason else None,
                "quality_satisfied": decision.quality_satisfied,
                "budget_exhausted": decision.budget_exhausted,
                "current_round": decision.current_round,
                "valid_product_count": metrics.valid_product_count,
                "required_count": metrics.required_count,
                "reasons": decision.reasons,
            },
        )
        return decision
