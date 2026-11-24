"""D3 LangGraph runtime integration tests (real ``langgraph``, stub dependencies).

Decision D3 makes LangGraph the formal Phase 5 runtime. These tests execute the real
``langgraph`` graph (no mock of LangGraph itself) with stub Planner / Judge / retrieval
/ rerank dependencies, so no network, model or index is required.

Locally ``langgraph`` may be absent (only the ``workflow`` extra installs it); the whole
module is then skipped. On the S4000 server the ``workflow`` extra is installed in
``/tmp/agr/venv`` and these tests are the evidence that the graph really runs.

Covered: compiled-graph instantiation, quality STOP, REPLAN loop, judgment API failure
(D4 -> ``api_failed``/judgment), initial Planner failure (D4 -> ``api_failed``/planner),
CLARIFY exit, budget-exhausted exit, duplicate-query loop guard and runtime parity with
the pure-Python fallback.
"""

from __future__ import annotations

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
    langgraph_adapter,
    langgraph_available,
)
from recommender.judgment import JudgmentBatchError
from recommender.planner import PlanGenerationError, PlannerResult, SystemCapabilities
from recommender.schemas.common import (
    ConstraintStatus,
    ControllerAction,
    FailedComponent,
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
    JudgmentBatch,
    JudgmentLLMOutput,
    JudgmentRecord,
    PreferenceFit,
)
from recommender.schemas.plan import RecommendationPlan
from recommender.schemas.rerank import RankedProduct

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not langgraph_available(), reason="langgraph is not installed"),
]


# -- builders ---------------------------------------------------------------


def _config(
    *,
    required_count: int = 3,
    min_judgments: int = 3,
    max_rounds: int = 3,
) -> ControllerConfig:
    return ControllerConfig(
        budget=ControllerBudgetConfig(
            remaining_replan_rounds=2,
            remaining_retrieval_calls=6,
            remaining_api_calls=60,
        ),
        quality=QualityThresholds(
            required_count=required_count,
            min_valid_candidates=1,
            min_judgments=min_judgments,
        ),
        loop_guard=LoopGuardConfig(max_replan_rounds=2, max_rounds=max_rounds),
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
                ]
            },
            "preferences": {"current": ["lightweight"], "priority": ["price"]},
            "retrieval": {
                "strategy": "hybrid_balanced",
                "tiger": {"enabled": False, "candidate_budget": "standard"},
                "e5_faiss": {
                    "enabled": True,
                    "queries": queries or ["daily moisturizer under 30"],
                    "candidate_budget": "standard",
                },
            },
            "ranking": {"rerank_query": "daily moisturizer under $30"},
            "clarification": {
                "required": clarification_required,
                "question": "您更偏好哪种香型？" if clarification_required else None,
                "reason": "缺少香型偏好" if clarification_required else None,
            },
            "assumptions": [],
            "change_summary": change_summary or [],
        }
    )


def _summary(valid: int) -> ConstraintSummary:
    return ConstraintSummary(
        evaluated_products=valid,
        pass_count=valid,
        fail_count=0,
        unknown_count=0,
        not_configured_count=0,
        valid_product_count=valid,
        constraint_fields=["price"],
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


def _judgment(product_id: str) -> JudgmentRecord:
    output = JudgmentLLMOutput(
        product_id=product_id,
        preference_fit=PreferenceFit(level=PreferenceFitLevel.HIGH),
        attribute_evidence={},
        match_assessment=MatchAssessment.PARTIAL_MATCH,
        uncertainty=UncertaintyLevel.MEDIUM,
    )
    return JudgmentRecord(
        result=output,
        model="deepseek-flash",
        prompt_version="judgment.v1",
        latency_ms=10.0,
        input_tokens=100,
        output_tokens=50,
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
        request_id="req_lg",
        candidates=candidates,
        unverified_candidates=[],
        rejected_candidates=[],
        constraint_summary=_summary(len(eligible)),
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
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
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
        if self.fail:
            raise JudgmentBatchError(
                "judgment timeout",
                batch_index=1,
                product_ids=[candidate.product_id for candidate in candidates],
                attempts=2,
                last_error="ReadTimeout",
                request_id=request_id,
                trace_id="tr_lg",
                error_kind="timeout",
            )
        return JudgmentBatch(
            request_id=request_id,
            records=[_judgment(candidate.product_id) for candidate in candidates],
        )


class _FakeRetrieval:
    def __init__(self, rounds: list[list[str]]) -> None:
        self.rounds = list(rounds)
        self.calls = 0

    def __call__(self, plan: RecommendationPlan) -> CandidatePool:
        self.calls += 1
        if not self.rounds:
            raise AssertionError("unexpected retrieval call")
        return _pool(self.rounds.pop(0))


def _rerank(plan: RecommendationPlan, pool: CandidatePool) -> list[RankedProduct]:
    return _ranked(pool.product_ids)


def _planner_result(plan: RecommendationPlan) -> PlannerResult:
    return PlannerResult(
        plan=plan,
        prompt_version="test.prompt.v1",
        model="deepseek-flash",
        attempts=1,
        latency_ms=5.0,
        request_id="req_lg",
        trace_id="tr_lg",
        input_tokens=10,
        output_tokens=20,
    )


def _workflow(
    planner: _FakePlanner,
    judge: _FakeJudge,
    retrieve: _FakeRetrieval,
    *,
    config: ControllerConfig | None = None,
) -> RecommendationWorkflow:
    controller = Controller(config or _config())
    dependencies = WorkflowDependencies(
        planner=planner,
        judge=judge,
        retrieve=retrieve,
        rerank=_rerank,
    )
    options = WorkflowOptions(
        controller=controller,
        capabilities=SystemCapabilities(retrieval_modules=[RecallPath.E5_FAISS]),
    )
    return RecommendationWorkflow(dependencies, options=options)


def _state(budget: BudgetState | None = None) -> RecommendationState:
    return RecommendationState(
        user_request="我想买一款30美元以内的清爽日常面霜",
        budget_state=budget or BudgetState(),
        request_id="req_lg",
        trace_id="tr_lg",
    )


# -- real runtime -----------------------------------------------------------


def test_langgraph_is_installed_and_graph_is_a_compiled_state_graph() -> None:
    assert langgraph_available() is True
    workflow = _workflow(
        _FakePlanner(_planner_result(_plan())),
        _FakeJudge(),
        _FakeRetrieval([["p1"]]),
    )
    graph = langgraph_adapter.build_graph(workflow)

    compiled_class = langgraph_adapter.compiled_graph_class()
    assert compiled_class is not None
    assert isinstance(graph, compiled_class)


def test_langgraph_quality_stop_runs_through_the_graph() -> None:
    planner = _FakePlanner(_planner_result(_plan()))
    judge = _FakeJudge()
    retrieve = _FakeRetrieval([["p1", "p2", "p3"]])
    workflow = _workflow(planner, judge, retrieve, config=_config())

    state = workflow.run_langgraph(_state())

    assert state.action is ControllerAction.STOP
    assert state.stop_reason is StopReason.QUALITY_SATISFIED
    assert state.quality_satisfied is True
    assert state.selected_products == ["p1", "p2", "p3"]
    assert state.current_round == 1
    assert planner.initial_calls == 1
    assert retrieve.calls == 1
    assert judge.calls == 1
    assert state.budget_state.remaining_api_calls == 58
    assert state.budget_state.remaining_retrieval_calls == 5
    assert workflow._langgraph_graph is not None


def test_langgraph_replan_loop_runs_two_rounds() -> None:
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
    retrieve = _FakeRetrieval([["p1"], ["p1", "p2", "p3"]])
    workflow = _workflow(planner, judge, retrieve, config=_config())

    state = workflow.run_langgraph(_state())

    assert state.stop_reason is StopReason.QUALITY_SATISFIED
    assert planner.replan_calls == 1
    assert retrieve.calls == 2
    assert state.current_round == 2
    assert state.current_plan is not None and state.current_plan.plan_id == "plan_002"
    assert state.budget_state.remaining_replan_rounds == 1


def test_langgraph_judgment_failure_stops_api_failed_judgment() -> None:
    planner = _FakePlanner(_planner_result(_plan()))
    judge = _FakeJudge(fail=True)
    retrieve = _FakeRetrieval([["p1", "p2", "p3"]])
    workflow = _workflow(planner, judge, retrieve, config=_config())

    state = workflow.run_langgraph(_state())

    assert state.stop_reason is StopReason.API_FAILED
    assert state.quality_satisfied is False
    assert state.selected_products == []
    assert state.decision is not None
    assert state.decision.failure is not None
    assert state.decision.failure.failed_component is FailedComponent.JUDGMENT
    assert state.decision.failure.error_type == "timeout"
    assert state.decision.failure.retryable is True


def test_langgraph_initial_planner_failure_stops_api_failed_planner() -> None:
    planner_error = PlanGenerationError(
        "no valid initial plan",
        mode=PlanMode.INITIAL,
        attempts=2,
        prompt_version="planner.initial.v1",
        last_error="invalid json",
        request_id="req_lg",
        trace_id="tr_lg",
    )
    planner = _FakePlanner(planner_error)
    retrieve = _FakeRetrieval([])
    workflow = _workflow(planner, _FakeJudge(), retrieve, config=_config())

    state = workflow.run_langgraph(_state())

    assert state.stop_reason is StopReason.API_FAILED
    assert state.decision is not None
    assert state.decision.failure is not None
    assert state.decision.failure.failed_component is FailedComponent.PLANNER
    assert retrieve.calls == 0


def test_langgraph_clarification_exits_without_retrieval() -> None:
    planner = _FakePlanner(_planner_result(_plan(clarification_required=True)))
    retrieve = _FakeRetrieval([])
    workflow = _workflow(planner, _FakeJudge(), retrieve, config=_config())

    state = workflow.run_langgraph(_state())

    assert state.action is ControllerAction.CLARIFY
    assert retrieve.calls == 0


def test_langgraph_budget_exhausted_before_retrieval() -> None:
    planner = _FakePlanner(_planner_result(_plan()))
    retrieve = _FakeRetrieval([["p1"]])
    workflow = _workflow(planner, _FakeJudge(), retrieve, config=_config())

    state = workflow.run_langgraph(
        _state(
            BudgetState(
                remaining_replan_rounds=2,
                remaining_retrieval_calls=0,
                remaining_api_calls=60,
            )
        )
    )

    assert state.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert state.budget_exhausted is True
    assert retrieve.calls == 0


def test_langgraph_duplicate_query_loop_guard_stops_before_retrieval() -> None:
    duplicate_plan = _plan(
        plan_id="plan_002",
        version=2,
        mode="replan",
        parent_plan_id="plan_001",
        queries=["daily moisturizer under 30"],
        change_summary=["no real change"],
    )
    planner = _FakePlanner(_planner_result(_plan()), [_planner_result(duplicate_plan)])
    retrieve = _FakeRetrieval([["p1"]])
    workflow = _workflow(planner, _FakeJudge(), retrieve, config=_config())

    state = workflow.run_langgraph(_state())

    assert retrieve.calls == 1
    assert state.stop_reason is StopReason.MAX_ROUNDS_REACHED
    assert state.quality_satisfied is False


def test_langgraph_and_python_runtimes_agree() -> None:
    replan_plan = _plan(
        plan_id="plan_002",
        version=2,
        mode="replan",
        parent_plan_id="plan_001",
        queries=["fragrance-free lightweight moisturizer"],
        change_summary=["增加无香型检索方向"],
    )

    def build() -> RecommendationWorkflow:
        return _workflow(
            _FakePlanner(_planner_result(_plan()), [_planner_result(replan_plan)]),
            _FakeJudge(),
            _FakeRetrieval([["p1"], ["p1", "p2", "p3"]]),
            config=_config(),
        )

    python_state = build().run_python(_state())
    graph_state = build().run_langgraph(_state())

    assert python_state.model_dump(mode="json") == graph_state.model_dump(mode="json")
    assert graph_state.stop_reason is StopReason.QUALITY_SATISFIED
