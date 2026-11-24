"""Phase 5 Controller + workflow tests (deterministic Python, no API, no models).

Covers: quality aggregation, STOP/REPLAN/CLARIFY rules, forced budget stop vs quality
stop, max rounds, no-valid-candidate degradation, decision D4 ``api_failed`` +
``failed_component``, D5 strict NFKC query hashing, D8 ``min_judgments``, workflow
REPLAN loop, planner/judge/retrieval failures, the ``selected_products`` evidence policy
and the D3 LangGraph / pure-Python runtime dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from recommender.config import (
    ControllerBudgetConfig,
    ControllerConfig,
    LoopGuardConfig,
    QualityThresholds,
)
from recommender.controller import (
    Controller,
    LangGraphUnavailableError,
    RecommendationWorkflow,
    WorkflowDependencies,
    WorkflowOptions,
    duplicate_queries,
    evaluate_quality,
    langgraph_available,
    normalize_query,
    plan_query_hashes,
    query_hash,
    summarize_quality,
)
from recommender.judgment import JudgmentBatchError
from recommender.planner import PlanGenerationError, PlannerResult, SystemCapabilities
from recommender.schemas.common import (
    ConstraintStatus,
    ControllerAction,
    EvidenceStatus,
    FailedComponent,
    MatchAssessment,
    PlanMode,
    PreferenceFitLevel,
    RecallPath,
    StopReason,
    UncertaintyLevel,
)
from recommender.schemas.constraints import ConstraintSummary
from recommender.schemas.controller import BudgetState, FailureInfo, RecommendationState
from recommender.schemas.fusion import CandidatePool, FusedCandidate
from recommender.schemas.judgment import (
    AttributeEvidence,
    JudgmentBatch,
    JudgmentLLMOutput,
    JudgmentRecord,
    PreferenceFit,
)
from recommender.schemas.plan import RecommendationPlan
from recommender.schemas.rerank import RankedProduct
from recommender.utils.errors import RetrievalError

pytestmark = pytest.mark.unit


# -- builders ---------------------------------------------------------------


def _config(
    *,
    required_count: int = 2,
    min_valid_candidates: int = 1,
    min_judgments: int = 1,
    max_rounds: int = 3,
    max_replan_rounds: int = 2,
    calibrated: bool = False,
) -> ControllerConfig:
    return ControllerConfig(
        budget=ControllerBudgetConfig(
            remaining_replan_rounds=2,
            remaining_retrieval_calls=6,
            remaining_api_calls=60,
        ),
        quality=QualityThresholds(
            required_count=required_count,
            min_valid_candidates=min_valid_candidates,
            min_judgments=min_judgments,
            calibrated=calibrated,
        ),
        loop_guard=LoopGuardConfig(
            max_replan_rounds=max_replan_rounds,
            max_rounds=max_rounds,
        ),
    )


def _plan(
    *,
    plan_id: str = "plan_001",
    version: int = 1,
    mode: str = "initial",
    parent_plan_id: str | None = None,
    queries: list[str] | None = None,
    clarification_required: bool = False,
    change_summary: list[str] | None = None,
) -> RecommendationPlan:
    return RecommendationPlan.model_validate(
        {
            "schema_version": "1.0",
            "plan_id": plan_id,
            "parent_plan_id": parent_plan_id,
            "plan_version": version,
            "mode": mode,
            "intent": {
                "intent": "product_recommendation",
                "category": "Beauty",
                "product_type": "moisturizer",
                "usage_scenario": "daily_use",
            },
            "constraints": {
                "hard": [
                    {
                        "field": "price",
                        "operator": "lte",
                        "value": 30,
                        "unit": "USD",
                        "source": "current_request",
                    }
                ],
                "soft": [
                    {
                        "attribute": "lightweight",
                        "source": "current_request",
                        "priority": "high",
                    }
                ],
            },
            "preferences": {
                "current": ["lightweight"],
                "historical": [],
                "negative": [],
                "priority": ["price"],
            },
            "retrieval": {
                "strategy": "hybrid_balanced",
                "tiger": {"enabled": False, "candidate_budget": "standard"},
                "e5_faiss": {
                    "enabled": True,
                    "queries": queries or ["daily moisturizer under 30"],
                    "candidate_budget": "standard",
                },
            },
            "ranking": {
                "rerank_query": "daily moisturizer under $30",
                "priority_attributes": ["price"],
            },
            "clarification": {
                "required": clarification_required,
                "question": "您更偏好哪种香型？" if clarification_required else None,
                "reason": "缺少香型偏好" if clarification_required else None,
            },
            "assumptions": [],
            "change_summary": change_summary or [],
        }
    )


def _summary(
    valid: int,
    *,
    unknown: int = 0,
    failed: int = 0,
    fields: tuple[str, ...] = ("price",),
) -> ConstraintSummary:
    return ConstraintSummary(
        evaluated_products=valid + unknown + failed,
        pass_count=valid,
        fail_count=failed,
        unknown_count=unknown,
        not_configured_count=0,
        valid_product_count=valid,
        constraint_fields=list(fields),
    )


def _ranked(product_ids: list[str]) -> list[RankedProduct]:
    return [
        RankedProduct(
            product_id=product_id,
            rerank_score=1.0 - index * 0.05,
            rerank_rank=index + 1,
            fusion_score=1.0 / (index + 1),
            fusion_rank=index + 1,
            retrieval_sources=[RecallPath.E5_FAISS],
        )
        for index, product_id in enumerate(product_ids)
    ]


def _judgment(
    product_id: str,
    *,
    match: MatchAssessment = MatchAssessment.PARTIAL_MATCH,
    uncertainty: UncertaintyLevel = UncertaintyLevel.MEDIUM,
    evidence: dict[str, AttributeEvidence] | None = None,
) -> JudgmentRecord:
    output = JudgmentLLMOutput(
        product_id=product_id,
        preference_fit=PreferenceFit(level=PreferenceFitLevel.HIGH),
        attribute_evidence=evidence or {},
        match_assessment=match,
        uncertainty=uncertainty,
    )
    return JudgmentRecord(
        result=output,
        model="deepseek-flash",
        prompt_version="judgment.v1",
        latency_ms=10.0,
        input_tokens=100,
        output_tokens=50,
    )


def _state(
    *,
    plan: RecommendationPlan | None = None,
    summary: ConstraintSummary | None = None,
    ranked: list[RankedProduct] | None = None,
    judgments: list[JudgmentRecord] | None = None,
    budget: BudgetState | None = None,
    current_round: int = 1,
    max_replan_rounds: int = 2,
) -> RecommendationState:
    return RecommendationState(
        user_request="我想买一款30美元以内的清爽日常面霜",
        current_plan=plan,
        constraint_summary=summary,
        ranked_products=ranked or [],
        judgment_results=judgments or [],
        budget_state=budget or BudgetState(),
        current_round=current_round,
        max_replan_rounds=max_replan_rounds,
        request_id="req_controller",
        trace_id="tr_controller",
    )


def _pool(eligible: list[str]) -> CandidatePool:
    candidates = [
        FusedCandidate(
            product_id=product_id,
            fusion_score=1.0 / (index + 1),
            retrieval_sources=[RecallPath.E5_FAISS],
            source_ranks={"e5_faiss": index + 1},
            constraint_status=ConstraintStatus.PASS,
        )
        for index, product_id in enumerate(eligible)
    ]
    return CandidatePool(
        request_id="req_controller",
        candidates=candidates,
        unverified_candidates=[],
        rejected_candidates=[],
        constraint_summary=_summary(len(eligible)),
    )


def _empty_pool() -> CandidatePool:
    return CandidatePool(
        request_id="req_controller",
        candidates=[],
        unverified_candidates=[],
        rejected_candidates=[],
        constraint_summary=_summary(0),
    )


# -- quality aggregation ----------------------------------------------------


def test_summarize_quality_aggregates_constraints_rerank_and_judgments() -> None:
    judgments = [
        _judgment(
            "p1",
            match=MatchAssessment.DIRECT_MATCH,
            evidence={
                "lightweight": AttributeEvidence(status=EvidenceStatus.SUPPORTED, source="title")
            },
        ),
        _judgment(
            "p2",
            evidence={
                "fragrance_free": AttributeEvidence(status=EvidenceStatus.UNKNOWN, source=None)
            },
        ),
        _judgment(
            "p3",
            match=MatchAssessment.NO_MATCH,
            evidence={
                "lightweight": AttributeEvidence(status=EvidenceStatus.SUPPORTED, source="title")
            },
        ),
    ]
    metrics = summarize_quality(
        constraint_summary=_summary(3),
        ranked_products=_ranked(["p1", "p2", "p3"]),
        judgment_results=judgments,
        required_count=3,
    )
    assert metrics.valid_product_count == 3
    assert metrics.hard_constraints_satisfied is True
    assert metrics.mean_rerank_score == pytest.approx(0.95)
    assert metrics.match_distribution == {"direct_match": 1, "partial_match": 1, "no_match": 1}
    assert metrics.evidence_coverage == pytest.approx(2 / 3)
    assert metrics.unknown_attribute_ratio == pytest.approx(1 / 3)
    assert metrics.diversity is None
    assert metrics.valid_judgment_count == 3
    assert metrics.quality_satisfied is False

    evaluated = evaluate_quality(metrics, _config(required_count=3).quality)
    assert evaluated.quality_satisfied is True


def test_valid_judgment_count_excludes_unaligned_products() -> None:
    """D8: a Judgment for a product outside the ranked set is not valid evidence."""
    metrics = summarize_quality(
        constraint_summary=_summary(2),
        ranked_products=_ranked(["p1", "p2"]),
        judgment_results=[_judgment("p1"), _judgment("p2"), _judgment("p9")],
        required_count=2,
    )
    assert metrics.valid_judgment_count == 2
    # All three records were parsed but two carry no attribute evidence entries.
    assert metrics.evidence_coverage == pytest.approx(0.0)


def test_quality_stop_requires_min_judgments_two_is_not_enough() -> None:
    """D8: 2 valid Judgments must never satisfy quality with min_judgments=3."""
    controller = Controller(_config(required_count=3, min_judgments=3))
    state = _state(
        plan=_plan(),
        summary=_summary(3),
        ranked=_ranked(["p1", "p2", "p3"]),
        judgments=[_judgment("p1"), _judgment("p2")],
    )
    decision = controller.decide(state)
    assert decision.quality_satisfied is False
    assert decision.action is ControllerAction.REPLAN
    assert decision.stop_reason is None
    assert any("min_judgments=3" in reason for reason in decision.reasons)


def test_quality_stop_with_three_min_judgments_satisfies() -> None:
    """D8: 3 valid Judgments meet min_judgments=3 together with the other rules."""
    controller = Controller(_config(required_count=3, min_judgments=3))
    state = _state(
        plan=_plan(),
        summary=_summary(3),
        ranked=_ranked(["p1", "p2", "p3"]),
        judgments=[_judgment("p1"), _judgment("p2"), _judgment("p3")],
    )
    decision = controller.decide(state)
    assert decision.action is ControllerAction.STOP
    assert decision.stop_reason is StopReason.QUALITY_SATISFIED
    assert decision.quality_satisfied is True
    assert any("min_judgments=3" in reason for reason in decision.reasons)


def test_no_configured_constraints_never_counts_as_verified_pass() -> None:
    summary = ConstraintSummary(
        evaluated_products=2,
        pass_count=0,
        fail_count=0,
        unknown_count=0,
        not_configured_count=2,
        valid_product_count=2,
        constraint_fields=[],
    )
    metrics = summarize_quality(
        constraint_summary=summary,
        ranked_products=_ranked(["p1", "p2"]),
        judgment_results=[],
        required_count=2,
    )
    # No check was configured: the aggregate flag describes "no known violation", while
    # the per-product status stays None (decision Q3) and not_configured is preserved.
    assert metrics.hard_constraints_satisfied is True
    assert summary.not_configured_count == 2


# -- controller decisions ---------------------------------------------------


def test_controller_stops_with_quality_satisfied() -> None:
    controller = Controller(_config(required_count=2))
    state = _state(
        plan=_plan(),
        summary=_summary(2),
        ranked=_ranked(["p1", "p2"]),
        judgments=[_judgment("p1"), _judgment("p2")],
    )
    decision = controller.decide(state)
    assert decision.action is ControllerAction.STOP
    assert decision.stop_reason is StopReason.QUALITY_SATISFIED
    assert decision.quality_satisfied is True
    assert decision.budget_exhausted is False
    assert any("uncalibrated" in reason for reason in decision.reasons)


def test_controller_requires_judgment_results_for_quality_stop() -> None:
    controller = Controller(_config(required_count=2))
    state = _state(
        plan=_plan(),
        summary=_summary(2),
        ranked=_ranked(["p1", "p2"]),
        judgments=[],
    )
    decision = controller.decide(state)
    assert decision.action is ControllerAction.REPLAN
    assert decision.stop_reason is None
    assert decision.quality_satisfied is False


def test_controller_replans_when_insufficient_candidates() -> None:
    controller = Controller(_config(required_count=2))
    state = _state(
        plan=_plan(),
        summary=_summary(1),
        ranked=_ranked(["p1"]),
        judgments=[_judgment("p1")],
    )
    decision = controller.decide(state)
    assert decision.action is ControllerAction.REPLAN
    assert decision.stop_reason is None
    assert decision.quality_satisfied is False


def test_controller_budget_exhausted_is_forced_stop_not_quality() -> None:
    controller = Controller(_config(required_count=2))
    state = _state(
        plan=_plan(),
        summary=_summary(1),
        ranked=_ranked(["p1"]),
        judgments=[_judgment("p1")],
        budget=BudgetState(remaining_replan_rounds=0),
    )
    decision = controller.decide(state)
    assert decision.action is ControllerAction.STOP
    assert decision.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert decision.quality_satisfied is False
    assert decision.budget_exhausted is True
    assert any("not a quality pass" in reason for reason in decision.reasons)


def test_controller_quality_on_last_round_is_not_mislabelled_as_budget() -> None:
    controller = Controller(_config(required_count=2))
    state = _state(
        plan=_plan(),
        summary=_summary(2),
        ranked=_ranked(["p1", "p2"]),
        judgments=[_judgment("p1"), _judgment("p2")],
        budget=BudgetState(remaining_replan_rounds=0),
    )
    decision = controller.decide(state)
    assert decision.stop_reason is StopReason.QUALITY_SATISFIED
    assert decision.quality_satisfied is True
    assert decision.budget_exhausted is False
    assert any("already at its limit" in reason for reason in decision.reasons)


def test_controller_stops_at_max_rounds() -> None:
    controller = Controller(_config(required_count=2, max_rounds=2))
    state = _state(
        plan=_plan(),
        summary=_summary(1),
        ranked=_ranked(["p1"]),
        judgments=[_judgment("p1")],
        budget=BudgetState(remaining_replan_rounds=2),
        current_round=2,
        max_replan_rounds=0,
    )
    decision = controller.decide(state)
    assert decision.stop_reason is StopReason.MAX_ROUNDS_REACHED
    assert decision.quality_satisfied is False
    assert any("loop guard" in reason for reason in decision.reasons)


def test_controller_replans_when_no_valid_candidates() -> None:
    controller = Controller(_config(required_count=2))
    state = _state(plan=_plan(), summary=_summary(0), ranked=[], judgments=[])
    decision = controller.decide(state)
    assert decision.action is ControllerAction.REPLAN
    assert any("no product passes" in reason for reason in decision.reasons)


def test_controller_no_valid_candidates_without_replan_budget() -> None:
    controller = Controller(_config(required_count=2, max_replan_rounds=0))
    state = _state(
        plan=_plan(),
        summary=_summary(0),
        ranked=[],
        judgments=[],
        budget=BudgetState(remaining_replan_rounds=2),
        max_replan_rounds=0,
    )
    decision = controller.decide(state)
    assert decision.action is ControllerAction.STOP
    assert decision.stop_reason is StopReason.NO_VALID_CANDIDATES
    assert decision.quality_satisfied is False


def test_controller_api_failure_stops_with_api_failed_and_component() -> None:
    """D4: an exhausted API call is ``api_failed``, not retrieval degradation."""
    controller = Controller(_config(required_count=2))
    state = _state(plan=_plan(), summary=_summary(2), ranked=_ranked(["p1", "p2"]))
    failure = FailureInfo(
        failed_component=FailedComponent.JUDGMENT,
        error_type="timeout",
        retryable=True,
        attempts=2,
        message="judgment batch 1: timeout",
    )
    decision = controller.decide(state, failure=failure)
    assert decision.action is ControllerAction.STOP
    assert decision.stop_reason is StopReason.API_FAILED
    assert decision.quality_satisfied is False
    assert decision.budget_exhausted is False
    assert decision.failure == failure
    assert decision.failure is not None
    assert decision.failure.failed_component is FailedComponent.JUDGMENT
    assert any("component=judgment" in reason for reason in decision.reasons)
    assert any("retryable" in reason for reason in decision.reasons)


def test_controller_api_failure_without_valid_candidates_is_still_api_failed() -> None:
    """D4: zero candidates must not relabel an API failure as no_valid_candidates."""
    controller = Controller(_config(required_count=2))
    state = _state(plan=_plan(), summary=_summary(0), ranked=[])
    decision = controller.decide(
        state,
        failure=FailureInfo(
            failed_component=FailedComponent.PLANNER,
            error_type="auth",
            retryable=False,
            attempts=1,
            message="initial planner failed",
        ),
    )
    assert decision.action is ControllerAction.STOP
    assert decision.stop_reason is StopReason.API_FAILED
    assert decision.stop_reason is not StopReason.NO_VALID_CANDIDATES
    assert decision.quality_satisfied is False
    assert decision.failure is not None
    assert decision.failure.failed_component is FailedComponent.PLANNER


def test_controller_api_failed_decision_roundtrips_through_state() -> None:
    controller = Controller(_config(required_count=2))
    state = _state(plan=_plan(), summary=_summary(2), ranked=_ranked(["p1", "p2"]))
    failure = FailureInfo(
        failed_component=FailedComponent.GENERATOR,
        error_type="server",
        retryable=True,
        attempts=3,
        message="generator failed",
    )
    decision = controller.decide(state, failure=failure)
    assert decision.stop_reason is StopReason.API_FAILED
    assert decision.failure is not None
    restored = decision.model_validate_json(decision.model_dump_json())
    assert restored.failure is not None
    assert restored.failure.failed_component is FailedComponent.GENERATOR


def test_controller_returns_clarify_when_plan_requires_it() -> None:
    controller = Controller(_config())
    state = _state(plan=_plan(clarification_required=True), summary=_summary(2))
    decision = controller.decide(state)
    assert decision.action is ControllerAction.CLARIFY
    assert decision.stop_reason is None
    assert any("clarification" in reason for reason in decision.reasons)


# -- loop guard -------------------------------------------------------------


def test_duplicate_query_detection_normalizes_case_and_whitespace() -> None:
    plan = _plan(queries=["Daily  Moisturizer", "night cream"])
    seen = [query_hash("daily moisturizer")]
    repeated = duplicate_queries(plan, seen)
    assert repeated == [query_hash("Daily  Moisturizer")]
    assert len(plan_query_hashes(plan)) == 2
    assert query_hash("Daily  Moisturizer") == query_hash("daily moisturizer")
    assert duplicate_queries(_plan(queries=["brand new query"]), seen) == []


def test_normalize_query_is_nfkc_casefold_and_whitespace_strict() -> None:
    """D5: full-width字符、大小写、制表符/表意空格都归一到同一 hash。"""
    assert normalize_query("  ＤＡＩＬＹ  Moisturizer　UNDER 30  ") == "daily moisturizer under 30"
    assert normalize_query("Daily\tMoisturizer\nUnder  30") == "daily moisturizer under 30"
    assert normalize_query("ｄａｉｌｙ moisturizer under 30") == "daily moisturizer under 30"
    canonical = query_hash("daily moisturizer under 30")
    for variant in (
        "ＤＡＩＬＹ MOISTURIZER UNDER 30",
        "daily\tmoisturizer   under 30",
        "  daily moisturizer under 30\n",
        "Daily Moisturizer Under 30",
    ):
        assert query_hash(variant) == canonical, variant
    # Word order and stop words are NOT normalised away: different semantics, different
    # hash (decision D5 forbids reordering / stop-word stripping).
    assert query_hash("moisturizer daily under 30") != canonical
    assert query_hash("daily moisturizer") != canonical


def test_duplicate_query_hashing_sees_full_width_variants_as_duplicates() -> None:
    plan = _plan(queries=["ＤＡＩＬＹ　ＭＯＩＳＴＵＲＩＺＥＲ"])
    seen = [query_hash("daily moisturizer")]
    assert duplicate_queries(plan, seen) == [query_hash("ＤＡＩＬＹ　ＭＯＩＳＴＵＲＩＺＥＲ")]


def test_recommendation_state_decision_roundtrips() -> None:
    controller = Controller(_config(required_count=2))
    state = _state(
        plan=_plan(),
        summary=_summary(2),
        ranked=_ranked(["p1", "p2"]),
        judgments=[_judgment("p1"), _judgment("p2")],
    )
    decision = controller.decide(state)
    state.decision = decision
    state.action = decision.action
    state.stop_reason = decision.stop_reason
    state.quality_satisfied = decision.quality_satisfied
    restored = RecommendationState.model_validate(state.model_dump(mode="json"))
    assert restored.decision is not None
    assert restored.decision.stop_reason is StopReason.QUALITY_SATISFIED


# -- workflow ---------------------------------------------------------------


def _planner_result(plan: RecommendationPlan) -> PlannerResult:
    return PlannerResult(
        plan=plan,
        prompt_version="test.prompt.v1",
        model="deepseek-flash",
        attempts=1,
        latency_ms=5.0,
        request_id="req_controller",
        trace_id="tr_controller",
        input_tokens=10,
        output_tokens=20,
    )


class _FakePlanner:
    def __init__(
        self,
        initial: PlannerResult | Exception,
        replans: list[PlannerResult | Exception] | None = None,
    ) -> None:
        self.initial = initial
        self.replans = list(replans or [])
        self.initial_calls = 0
        self.replan_calls = 0

    def plan_initial(self, data: Any) -> PlannerResult:
        self.initial_calls += 1
        if isinstance(self.initial, Exception):
            raise self.initial
        return self.initial

    def plan_replan(self, data: Any) -> PlannerResult:
        self.replan_calls += 1
        if not self.replans:
            raise AssertionError("unexpected replan call")
        item = self.replans.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeJudge:
    def __init__(
        self,
        *,
        per_product: dict[str, dict[str, Any]] | None = None,
        fail_on_call: int | None = None,
    ) -> None:
        self.per_product = per_product or {}
        self.fail_on_call = fail_on_call
        self.calls = 0

    def batch_count(self, product_count: int) -> int:
        return 1 if product_count > 0 else 0

    def judge(
        self,
        *,
        request_id: str,
        user_request: str,
        plan: RecommendationPlan,
        candidates: list[Any],
    ) -> JudgmentBatch:
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise JudgmentBatchError(
                "judgment timeout",
                batch_index=1,
                product_ids=[candidate.product_id for candidate in candidates],
                attempts=2,
                last_error="ReadTimeout",
                request_id=request_id,
                trace_id="tr_controller",
                error_kind="timeout",
            )
        records = []
        for candidate in candidates:
            options = self.per_product.get(candidate.product_id, {})
            records.append(
                _judgment(
                    candidate.product_id,
                    match=options.get("match", MatchAssessment.PARTIAL_MATCH),
                    uncertainty=options.get("uncertainty", UncertaintyLevel.MEDIUM),
                )
            )
        return JudgmentBatch(request_id=request_id, records=records)


class _FakeRetrieval:
    def __init__(
        self,
        rounds: list[list[str] | Exception],
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self.rounds = list(rounds)
        self.fail_on_call = fail_on_call
        self.calls = 0

    def __call__(self, plan: RecommendationPlan) -> CandidatePool:
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RetrievalError("E5 index unavailable")
        if not self.rounds:
            raise AssertionError("unexpected retrieval call")
        eligible = self.rounds.pop(0)
        return _pool(eligible) if eligible else _empty_pool()


def _rerank(plan: RecommendationPlan, pool: CandidatePool) -> list[RankedProduct]:
    return _ranked(pool.product_ids)


def _workflow(
    planner: _FakePlanner,
    judge: _FakeJudge,
    retrieve: _FakeRetrieval,
    *,
    config: ControllerConfig | None = None,
    select_top_k: int | None = None,
    rerank: Callable[[RecommendationPlan, CandidatePool], list[RankedProduct]] | None = None,
) -> RecommendationWorkflow:
    controller = Controller(config or _config())
    dependencies = WorkflowDependencies(
        planner=planner,
        judge=judge,
        retrieve=retrieve,
        rerank=rerank or _rerank,
    )
    options = WorkflowOptions(
        controller=controller,
        capabilities=SystemCapabilities(retrieval_modules=[RecallPath.E5_FAISS]),
        select_top_k=select_top_k,
    )
    return RecommendationWorkflow(dependencies, options=options)


def _workflow_state(budget: BudgetState | None = None) -> RecommendationState:
    return RecommendationState(
        user_request="我想买一款30美元以内的清爽日常面霜",
        budget_state=budget or BudgetState(),
        request_id="req_wf",
        trace_id="tr_wf",
    )


def test_workflow_stops_with_quality_and_selects_products() -> None:
    planner = _FakePlanner(_planner_result(_plan()))
    judge = _FakeJudge()
    retrieve = _FakeRetrieval([["p1", "p2"]])
    workflow = _workflow(planner, judge, retrieve, config=_config(required_count=2))

    state = workflow.run(_workflow_state())

    assert state.action is ControllerAction.STOP
    assert state.stop_reason is StopReason.QUALITY_SATISFIED
    assert state.quality_satisfied is True
    assert state.budget_exhausted is False
    assert state.selected_products == ["p1", "p2"]
    assert state.current_round == 1
    assert state.decision is not None
    assert planner.initial_calls == 1
    assert planner.replan_calls == 0
    assert retrieve.calls == 1
    assert state.budget_state.remaining_retrieval_calls == 5
    assert state.budget_state.remaining_replan_rounds == 2
    assert state.budget_state.remaining_api_calls == 58
    assert state.seen_query_hashes == plan_query_hashes(_plan())


def test_workflow_replans_once_then_stops_with_quality() -> None:
    replan_plan = _plan(
        plan_id="plan_002",
        version=2,
        mode="replan",
        parent_plan_id="plan_001",
        queries=["fragrance-free lightweight moisturizer"],
        change_summary=["增加无香型检索方向"],
    )
    planner = _FakePlanner(_planner_result(_plan()), [_planner_result(replan_plan)])
    judge = _FakeJudge()
    retrieve = _FakeRetrieval([["p1"], ["p1", "p2"]])
    workflow = _workflow(planner, judge, retrieve, config=_config(required_count=2))

    state = workflow.run(_workflow_state())

    assert planner.replan_calls == 1
    assert retrieve.calls == 2
    assert state.current_plan is not None
    assert state.current_plan.plan_id == "plan_002"
    assert state.current_round == 2
    assert state.stop_reason is StopReason.QUALITY_SATISFIED
    assert state.selected_products == ["p1", "p2"]
    assert state.budget_state.remaining_replan_rounds == 1
    assert state.budget_state.remaining_retrieval_calls == 4
    assert state.budget_state.remaining_api_calls == 56
    assert len(state.seen_query_hashes) == 2


def test_workflow_duplicate_replan_query_stops_before_retrieval() -> None:
    duplicate_plan = _plan(
        plan_id="plan_002",
        version=2,
        mode="replan",
        parent_plan_id="plan_001",
        queries=["daily moisturizer under 30"],
        change_summary=["rephrased nothing"],
    )
    planner = _FakePlanner(_planner_result(_plan()), [_planner_result(duplicate_plan)])
    judge = _FakeJudge()
    retrieve = _FakeRetrieval([[]])
    workflow = _workflow(planner, judge, retrieve, config=_config(required_count=2))

    state = workflow.run(_workflow_state())

    assert retrieve.calls == 1
    assert state.stop_reason is StopReason.MAX_ROUNDS_REACHED
    assert state.quality_satisfied is False
    assert state.decision is not None
    assert any("repeated" in reason.lower() for reason in state.decision.reasons)


def test_workflow_initial_planner_failure_stops_api_failed_planner() -> None:
    """D7 default + D4: no fallback, so the exhausted Planner stops api_failed/planner."""
    planner_error = PlanGenerationError(
        "no valid initial plan",
        mode=PlanMode.INITIAL,
        attempts=2,
        prompt_version="planner.initial.v1",
        last_error="invalid json",
        request_id="req_wf",
        trace_id="tr_wf",
    )
    planner = _FakePlanner(planner_error)
    judge = _FakeJudge()
    retrieve = _FakeRetrieval([])
    workflow = _workflow(planner, judge, retrieve)

    state = workflow.run_python(_workflow_state())

    assert state.action is ControllerAction.STOP
    assert state.stop_reason is StopReason.API_FAILED
    assert state.quality_satisfied is False
    assert state.selected_products == []
    assert state.decision is not None
    assert state.decision.failure is not None
    assert state.decision.failure.failed_component is FailedComponent.PLANNER
    assert state.decision.failure.attempts == 2
    assert retrieve.calls == 0


def test_workflow_replanner_failure_keeps_previous_plan_and_stops_api_failed() -> None:
    planner_error = PlanGenerationError(
        "no valid replan",
        mode=PlanMode.REPLAN,
        attempts=2,
        prompt_version="planner.replan.v1",
        last_error="dropped hard constraint",
        request_id="req_wf",
        trace_id="tr_wf",
    )
    planner = _FakePlanner(_planner_result(_plan()), [planner_error])
    judge = _FakeJudge()
    retrieve = _FakeRetrieval([["p1"]])
    workflow = _workflow(planner, judge, retrieve, config=_config(required_count=2))

    state = workflow.run_python(_workflow_state())

    assert retrieve.calls == 1
    assert state.current_plan is not None and state.current_plan.plan_id == "plan_001"
    assert state.stop_reason is StopReason.API_FAILED
    assert state.quality_satisfied is False
    assert state.decision is not None
    assert state.decision.failure is not None
    assert state.decision.failure.failed_component is FailedComponent.PLANNER
    assert any("replanner failed" in reason for reason in state.decision.reasons)


def test_workflow_judgment_failure_stops_api_failed_judgment() -> None:
    planner = _FakePlanner(_planner_result(_plan()))
    judge = _FakeJudge(fail_on_call=1)
    retrieve = _FakeRetrieval([["p1", "p2"]])
    workflow = _workflow(planner, judge, retrieve, config=_config(required_count=2))

    state = workflow.run_python(_workflow_state())

    assert state.action is ControllerAction.STOP
    assert state.stop_reason is StopReason.API_FAILED
    assert state.quality_satisfied is False
    assert state.selected_products == []
    assert judge.calls == 1
    assert state.decision is not None
    assert state.decision.failure is not None
    assert state.decision.failure.failed_component is FailedComponent.JUDGMENT
    assert state.decision.failure.error_type == "timeout"
    assert state.decision.failure.retryable is True
    assert state.decision.failure.attempts == 2


def test_workflow_budget_exhausted_before_retrieval() -> None:
    planner = _FakePlanner(_planner_result(_plan()))
    judge = _FakeJudge()
    retrieve = _FakeRetrieval([["p1", "p2"]])
    workflow = _workflow(planner, judge, retrieve, config=_config(required_count=2))

    state = workflow.run(
        _workflow_state(
            BudgetState(
                remaining_replan_rounds=2,
                remaining_retrieval_calls=0,
                remaining_api_calls=60,
            )
        )
    )

    assert retrieve.calls == 0
    assert state.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert state.quality_satisfied is False
    assert state.budget_exhausted is True


def test_workflow_selection_excludes_no_match_and_high_uncertainty() -> None:
    planner = _FakePlanner(_planner_result(_plan()))
    judge = _FakeJudge(
        per_product={
            "p1": {"match": MatchAssessment.DIRECT_MATCH},
            "p2": {"match": MatchAssessment.NO_MATCH},
            "p3": {"uncertainty": UncertaintyLevel.HIGH},
        }
    )
    retrieve = _FakeRetrieval([["p1", "p2", "p3"]])
    workflow = _workflow(planner, judge, retrieve, config=_config(required_count=3))

    state = workflow.run(_workflow_state())

    assert state.stop_reason is StopReason.QUALITY_SATISFIED
    assert state.selected_products == ["p1"]


class _PartialJudge(_FakeJudge):
    """Returns judgments only for a subset of candidates (D8 alignment case)."""

    def __init__(self, judged: set[str]) -> None:
        super().__init__()
        self.judged = judged

    def judge(
        self,
        *,
        request_id: str,
        user_request: str,
        plan: RecommendationPlan,
        candidates: list[Any],
    ) -> JudgmentBatch:
        self.calls += 1
        records = [
            _judgment(candidate.product_id)
            for candidate in candidates
            if candidate.product_id in self.judged
        ]
        return JudgmentBatch(request_id=request_id, records=records)


def test_workflow_selection_requires_a_judgment_for_each_selected_product() -> None:
    """D8: a ranked product without a valid Judgment is never selected into the top-K."""
    planner = _FakePlanner(_planner_result(_plan()))
    judge = _PartialJudge({"p1", "p2"})
    retrieve = _FakeRetrieval([["p1", "p2", "p3"]])
    workflow = _workflow(
        planner,
        judge,
        retrieve,
        config=_config(required_count=2, min_judgments=2),
    )

    state = workflow.run(_workflow_state())

    assert state.stop_reason is StopReason.QUALITY_SATISFIED
    assert state.selected_products == ["p1", "p2"]
    assert "p3" not in state.selected_products


def test_workflow_retrieval_failure_stops_degraded() -> None:
    planner = _FakePlanner(_planner_result(_plan()))
    judge = _FakeJudge()
    retrieve = _FakeRetrieval([["p1"]], fail_on_call=1)
    workflow = _workflow(planner, judge, retrieve)

    state = workflow.run(_workflow_state())

    assert state.stop_reason is StopReason.RETRIEVAL_DEGRADED
    assert state.quality_satisfied is False


def test_workflow_empty_rerank_stops_degraded() -> None:
    planner = _FakePlanner(_planner_result(_plan()))
    judge = _FakeJudge()
    retrieve = _FakeRetrieval([["p1"]])
    workflow = _workflow(
        planner,
        judge,
        retrieve,
        rerank=lambda plan, pool: [],
    )

    state = workflow.run(_workflow_state())

    assert state.stop_reason is StopReason.RETRIEVAL_DEGRADED
    assert state.quality_satisfied is False
    assert state.selected_products == []
    assert judge.calls == 0


# -- D3 runtime dispatch (LangGraph formal runtime vs pure-Python fallback) --


def _quality_scenario() -> tuple[_FakePlanner, _FakeJudge, _FakeRetrieval]:
    return (
        _FakePlanner(_planner_result(_plan())),
        _FakeJudge(),
        _FakeRetrieval([["p1", "p2"]]),
    )


def test_run_langgraph_fails_loudly_when_runtime_missing() -> None:
    if langgraph_available():
        pytest.skip("langgraph is installed: the integration test covers the real runtime")
    planner, judge, retrieve = _quality_scenario()
    workflow = _workflow(planner, judge, retrieve, config=_config(required_count=2))

    with pytest.raises(LangGraphUnavailableError):
        workflow.run_langgraph(_workflow_state())


def test_run_falls_back_to_python_and_records_it_without_langgraph(
    caplog: pytest.LogCaptureFixture,
) -> None:
    if langgraph_available():
        pytest.skip("langgraph is installed: fallback is not the formal path here")
    caplog.set_level("WARNING", logger="recommender.controller.workflow")
    planner, judge, retrieve = _quality_scenario()
    workflow = _workflow(planner, judge, retrieve, config=_config(required_count=2))

    state = workflow.run(_workflow_state())
    assert state.stop_reason is StopReason.QUALITY_SATISFIED
    assert any("pure-Python fallback" in record.message for record in caplog.records)


def test_python_and_langgraph_runtimes_produce_identical_states() -> None:
    """D3: the formal runtime must not change any contract or business decision."""
    if not langgraph_available():
        pytest.skip("langgraph is not installed")
    planner, judge, retrieve = _quality_scenario()
    python_workflow = _workflow(planner, judge, retrieve, config=_config(required_count=2))
    planner2, judge2, retrieve2 = _quality_scenario()
    langgraph_workflow = _workflow(
        planner2, judge2, retrieve2, config=_config(required_count=2)
    )

    python_state = python_workflow.run_python(_workflow_state())
    langgraph_state = langgraph_workflow.run_langgraph(_workflow_state())

    assert python_state.model_dump(mode="json") == langgraph_state.model_dump(mode="json")
    assert langgraph_state.stop_reason is StopReason.QUALITY_SATISFIED
    assert langgraph_state.selected_products == ["p1", "p2"]
    assert langgraph_workflow._langgraph_graph is not None


def test_run_uses_the_langgraph_runtime_when_installed() -> None:
    if not langgraph_available():
        pytest.skip("langgraph is not installed")
    planner, judge, retrieve = _quality_scenario()
    workflow = _workflow(planner, judge, retrieve, config=_config(required_count=2))

    state = workflow.run(_workflow_state())

    assert state.stop_reason is StopReason.QUALITY_SATISFIED
    assert state.quality_satisfied is True
    # The compiled graph is cached on the workflow instance, proving the formal runtime
    # (not the pure-Python loop) executed this run.
    assert workflow._langgraph_graph is not None
