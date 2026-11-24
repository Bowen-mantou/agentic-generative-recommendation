"""Phase 5 Controller + workflow tests (deterministic Python, no API, no models).

Covers: quality aggregation, STOP/REPLAN/CLARIFY rules, forced budget stop vs quality
stop, max rounds, no-valid-candidate degradation, API-failure degradation, duplicate
query detection, workflow REPLAN loop, planner/judge/retrieval failures and the
``selected_products`` evidence policy.
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
    RecommendationWorkflow,
    WorkflowDependencies,
    WorkflowOptions,
    duplicate_queries,
    evaluate_quality,
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
    MatchAssessment,
    PlanMode,
    PreferenceFitLevel,
    RecallPath,
    StopReason,
    UncertaintyLevel,
)
from recommender.schemas.constraints import ConstraintSummary
from recommender.schemas.controller import BudgetState, RecommendationState
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
            calibrated=calibrated,
        ),
        loop_guard=LoopGuardConfig(
            max_replan_rounds=max_replan_rounds,
            max_rounds=max_rounds,
            duplicate_query_similarity=0.95,
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
    assert metrics.quality_satisfied is False

    evaluated = evaluate_quality(metrics, _config(required_count=3).quality)
    assert evaluated.quality_satisfied is True


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


def test_controller_api_failure_degrades_with_valid_candidates() -> None:
    controller = Controller(_config(required_count=2))
    state = _state(plan=_plan(), summary=_summary(2), ranked=_ranked(["p1", "p2"]))
    decision = controller.decide(state, api_failure="judgment batch 1: timeout")
    assert decision.action is ControllerAction.STOP
    assert decision.stop_reason is StopReason.RETRIEVAL_DEGRADED
    assert decision.quality_satisfied is False


def test_controller_api_failure_without_valid_candidates() -> None:
    controller = Controller(_config(required_count=2))
    state = _state(plan=_plan(), summary=_summary(0), ranked=[])
    decision = controller.decide(state, api_failure="initial planner failed")
    assert decision.stop_reason is StopReason.NO_VALID_CANDIDATES
    assert decision.quality_satisfied is False


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


def test_workflow_initial_planner_failure_stops_degraded() -> None:
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

    state = workflow.run(_workflow_state())

    assert state.action is ControllerAction.STOP
    assert state.stop_reason is StopReason.RETRIEVAL_DEGRADED
    assert state.quality_satisfied is False
    assert state.selected_products == []
    assert retrieve.calls == 0


def test_workflow_replanner_failure_keeps_previous_plan_and_stops() -> None:
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

    state = workflow.run(_workflow_state())

    assert retrieve.calls == 1
    assert state.current_plan is not None and state.current_plan.plan_id == "plan_001"
    assert state.stop_reason is StopReason.RETRIEVAL_DEGRADED
    assert state.quality_satisfied is False
    assert state.decision is not None
    assert any("replanner failed" in reason for reason in state.decision.reasons)


def test_workflow_judgment_failure_stops_without_fake_quality() -> None:
    planner = _FakePlanner(_planner_result(_plan()))
    judge = _FakeJudge(fail_on_call=1)
    retrieve = _FakeRetrieval([["p1", "p2"]])
    workflow = _workflow(planner, judge, retrieve, config=_config(required_count=2))

    state = workflow.run(_workflow_state())

    assert state.action is ControllerAction.STOP
    assert state.stop_reason is StopReason.RETRIEVAL_DEGRADED
    assert state.quality_satisfied is False
    assert state.selected_products == []
    assert judge.calls == 1


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
