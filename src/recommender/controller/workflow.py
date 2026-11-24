"""REPLAN/STOP workflow: LangGraph orchestration over pure-Python business logic.

Phase 5 hardening (decision D3, 2026-11-24) makes **LangGraph the formal runtime**:

* :meth:`RecommendationWorkflow.run` dispatches to a real ``langgraph`` graph when the
  optional ``workflow`` extra is installed, and keeps the pure-Python state machine
  (:meth:`RecommendationWorkflow.run_python`) as the explicit, logged fallback for
  offline environments.
* :meth:`RecommendationWorkflow.run_langgraph` is the hard entry point: it raises
  :class:`~recommender.controller.langgraph_adapter.LangGraphUnavailableError` instead of
  silently degrading when LangGraph is missing.
* Both runtimes execute the *same* step functions (``step_prepare``, ``step_plan``,
  ``step_clarification``, ``step_retrieve``, ``step_rerank``, ``step_judge``,
  ``step_controller``, ``step_replan``). LangGraph owns state passing and conditional
  edges; Python owns every business judgement (constraints, budgets, retries, schema).

The node/edge shape (module 2 §7, final-stage doc §4.5):

``prepare -> planner -> clarification_gate -> retrieve -> rerank -> judge -> controller``
with ``controller -> REPLAN -> replanner -> retrieve`` and ``STOP``/``CLARIFY`` exits.

Loop protection is layered:

* ``BudgetState`` (replan rounds, retrieval calls, API calls);
* ``ControllerConfig.loop_guard`` max rounds / max replan rounds;
* duplicate E5 query detection against ``RecommendationState.seen_query_hashes``
  (decision D5: strict NFKC-normalised SHA-256 equality, no similarity threshold);
* bounded repair inside Planner and Judge.

Failure semantics (decision D4): a Planner, Judgment or Generator call that exhausted
its bounded retries stops the workflow with ``StopReason.API_FAILED`` and a
``FailureInfo`` naming the failed component. Retrieval/Reranker failures keep
``RETRIEVAL_DEGRADED`` because they are not model-API failures. Nothing is ever
fabricated: no plan, no empty-candidate success and no quality pass.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from recommender.config import ControllerConfig
from recommender.controller.controller import (
    Controller,
    duplicate_queries,
    plan_query_hashes,
)
from recommender.judgment.judge import Judge, JudgmentBatchError
from recommender.observability.context import get_request_context
from recommender.observability.logging import get_logger
from recommender.observability.trace import TraceWriter
from recommender.planner.planner import PlanGenerationError, Planner
from recommender.planner.prompts import (
    ControllerFeedback,
    ExecutionFeedback,
    PlannerInput,
    ReplanInput,
    SystemCapabilities,
)
from recommender.schemas.common import (
    ControllerAction,
    EvidenceStatus,
    FailedComponent,
    MatchAssessment,
    StopReason,
    UncertaintyLevel,
)
from recommender.schemas.controller import (
    ControllerDecision,
    FailureInfo,
    RecommendationState,
)
from recommender.schemas.fusion import CandidatePool
from recommender.schemas.judgment import JudgmentCandidate
from recommender.schemas.plan import RecommendationPlan
from recommender.schemas.rerank import RankedProduct
from recommender.utils.errors import ContractError, RecommenderError

_LOGGER = get_logger(__name__)

RetrieveFn = Callable[[RecommendationPlan], CandidatePool]
RerankFn = Callable[[RecommendationPlan, CandidatePool], Sequence[RankedProduct]]
CandidatesFn = Callable[
    [CandidatePool, Sequence[RankedProduct], RecommendationPlan],
    list[JudgmentCandidate],
]

# Error kinds that a bounded retry could plausibly fix. Used only to annotate
# ``FailureInfo.retryable``; the retries themselves already happened upstream.
_RETRYABLE_ERROR_KINDS = frozenset({"timeout", "rate_limit", "server", "network", "truncated"})


def build_judgment_candidates(
    pool: CandidatePool,
    ranked: Sequence[RankedProduct],
    plan: RecommendationPlan,
) -> list[JudgmentCandidate]:
    """Default catalog view for Judgment, built from ``FusedCandidate.product_metadata``.

    ``product_metadata`` currently carries title/price/category/brand; description and
    structured attributes are only present when the pool was built with a richer catalog
    payload. Missing fields stay missing, which the Judgment prompt turns into
    ``unknown`` evidence instead of an inferred "supported".
    """
    del plan  # reserved for future capability-aware views
    by_id = {candidate.product_id: candidate for candidate in pool.candidates}
    result: list[JudgmentCandidate] = []
    for product in ranked:
        fused = by_id.get(product.product_id)
        if fused is None:
            continue
        metadata = fused.product_metadata or {}
        attributes: dict[str, str] = {}
        for nested_key in ("attributes", "details"):
            nested = metadata.get(nested_key)
            if isinstance(nested, dict):
                for attribute_key, attribute_value in nested.items():
                    if attribute_value is None:
                        continue
                    text = str(attribute_value).strip()
                    if text:
                        attributes[str(attribute_key)] = text
        for key in ("brand", "main_category"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                attributes[key] = value.strip()
        raw_price = metadata.get("price")
        price: float | None = None
        if isinstance(raw_price, (int, float)) and not isinstance(raw_price, bool):
            price = float(raw_price)
        raw_description = metadata.get("description")
        raw_title = metadata.get("title")
        result.append(
            JudgmentCandidate(
                product_id=product.product_id,
                title=str(raw_title) if raw_title is not None else "",
                price=price,
                description=(
                    str(raw_description) if isinstance(raw_description, str) else None
                ),
                attributes=attributes,
                constraint_status=fused.constraint_status,
            )
        )
    return result


@dataclass(frozen=True)
class WorkflowDependencies:
    """Injected execution nodes (retrieval/rerank are Phase 2/3 services)."""

    planner: Planner
    judge: Judge
    retrieve: RetrieveFn
    rerank: RerankFn
    candidates_from_pool: CandidatesFn = build_judgment_candidates


@dataclass(frozen=True)
class WorkflowOptions:
    """Optional wiring knobs that are not part of the state contract.

    ``Controller`` owns the authoritative ``ControllerConfig`` so the workflow cannot
    drift from the thresholds that produced a decision.

    ``use_langgraph`` (decision D3) selects the formal LangGraph runtime when the
    optional dependency is installed. It exists so an operator can deliberately run the
    pure-Python fallback for an offline diagnosis; production must leave it ``True``.
    """

    controller: Controller = field(default_factory=Controller)
    capabilities: SystemCapabilities = field(default_factory=SystemCapabilities)
    select_top_k: int | None = None
    use_langgraph: bool = True


class RecommendationWorkflow:
    """Phase 5 agent loop; LangGraph runtime with a pure-Python business core."""

    def __init__(
        self,
        dependencies: WorkflowDependencies,
        *,
        options: WorkflowOptions | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        self._deps = dependencies
        self._options = options or WorkflowOptions()
        self._controller = self._options.controller
        self._config: ControllerConfig = self._controller.config
        self._capabilities = self._options.capabilities
        self._trace = trace
        # Compiled LangGraph is cached per workflow instance (built lazily).
        self._langgraph_graph: Any | None = None

    @property
    def config(self) -> ControllerConfig:
        """Authoritative Controller configuration (thresholds, loop guard, budget)."""
        return self._config

    # -- entry points --------------------------------------------------------
    def run(self, state: RecommendationState) -> RecommendationState:
        """Formal Phase 5 entry point (decision D3).

        Uses the real LangGraph runtime when installed; otherwise logs the explicit
        fallback and runs the identical pure-Python state machine.
        """
        adapter = _langgraph_adapter()
        if adapter is not None and self._options.use_langgraph:
            return adapter.run_with_langgraph(self, state)
        if self._options.use_langgraph:
            _LOGGER.warning(
                "langgraph is not installed; running the pure-Python fallback workflow. "
                "Install the 'workflow' extra to execute the formal LangGraph runtime.",
                extra={"fallback": "python_state_machine", "request_id": state.request_id},
            )
        return self.run_python(state)

    def run_langgraph(self, state: RecommendationState) -> RecommendationState:
        """Run the formal LangGraph runtime; fail loudly when it is unavailable."""
        adapter = _langgraph_adapter()
        if adapter is None:
            from recommender.controller.langgraph_adapter import LangGraphUnavailableError

            raise LangGraphUnavailableError(
                "langgraph is not installed; install the 'workflow' extra "
                "(pip install -e '.[workflow]') or use run_python() explicitly"
            )
        return adapter.run_with_langgraph(self, state)

    def run_python(self, state: RecommendationState) -> RecommendationState:
        """Pure-Python state machine (business core and offline fallback)."""
        self.step_prepare(state)
        terminal = self.step_plan(state)
        if terminal is not None:
            return self.finalize(state, terminal)
        terminal = self.step_clarification(state)
        if terminal is not None:
            return self.finalize(state, terminal)

        while True:
            for step in (self.step_retrieve, self.step_rerank, self.step_judge):
                terminal = step(state)
                if terminal is not None:
                    return self.finalize(state, terminal)
            decision = self.step_controller(state)
            if decision.action is not ControllerAction.REPLAN:
                return self.finalize(state, decision)
            state.action = ControllerAction.REPLAN
            state.decision = decision
            stop = self.step_replan(state, decision)
            if stop is not None:
                return self.finalize(state, stop)

    # -- nodes shared by both runtimes --------------------------------------
    def step_prepare(self, state: RecommendationState) -> None:
        """Prepare node: clamp the loop guard and emit the start event."""
        if state.max_replan_rounds > self._config.loop_guard.max_replan_rounds:
            state.max_replan_rounds = self._config.loop_guard.max_replan_rounds
        self._emit(
            "workflow.started",
            {
                "request_id": state.request_id,
                "current_round": state.current_round,
                "budget": state.budget_state.model_dump(mode="json"),
                "capabilities": [
                    path.value for path in self._capabilities.retrieval_modules
                ],
            },
        )

    def step_plan(self, state: RecommendationState) -> ControllerDecision | None:
        """Initial Planner node. Returns a terminal decision when planning failed."""
        if state.current_plan is None:
            terminal = self._generate_initial_plan(state)
            if terminal is not None:
                return terminal
        plan = state.current_plan
        if plan is None:  # pragma: no cover - guarded by the branch above
            raise ContractError("workflow has no plan after the initial planner node")
        state.current_round = max(1, state.current_round)
        self._remember_queries(state, plan)
        return None

    def step_clarification(self, state: RecommendationState) -> ControllerDecision | None:
        """Clarification gate: a plan that needs user input exits as CLARIFY."""
        plan = state.current_plan
        if plan is None:  # pragma: no cover - guarded by the caller
            raise ContractError("clarification gate requires a current plan")
        if not plan.clarification.required:
            return None
        return self._controller.decide(state)

    def step_retrieve(self, state: RecommendationState) -> ControllerDecision | None:
        """Retrieval node: consume the retrieval budget and build the candidate pool."""
        plan = state.current_plan
        if plan is None:  # pragma: no cover - planner always runs first
            raise ContractError("retrieval requires a current plan")
        path_count = max(1, len(plan.retrieval.enabled_paths()))
        if state.budget_state.remaining_retrieval_calls < path_count:
            return self._controller.budget_stop(
                state,
                dimension="retrieval",
                reason=f"plan needs {path_count} retrieval calls",
            )
        state.budget_state = state.budget_state.consume_retrieval_calls(path_count)
        try:
            pool = self._deps.retrieve(plan)
        except RecommenderError as exc:
            self._emit("workflow.retrieval_failed", {"error": exc.to_dict()})
            return ControllerDecision(
                action=ControllerAction.STOP,
                stop_reason=StopReason.RETRIEVAL_DEGRADED,
                quality_satisfied=False,
                budget_exhausted=False,
                reasons=["retrieval failed", f"error: {exc.message}"],
                current_round=state.current_round,
            )
        state.candidate_pool = pool
        state.constraint_summary = pool.constraint_summary
        state.fused_candidates = [
            *pool.candidates,
            *pool.unverified_candidates,
            *pool.rejected_candidates,
        ]
        self._emit(
            "workflow.retrieval_completed",
            {
                "evaluated": pool.constraint_summary.evaluated_products,
                "eligible": len(pool.candidates),
                "unverified": len(pool.unverified_candidates),
                "rejected": len(pool.rejected_candidates),
            },
        )
        return None

    def step_rerank(self, state: RecommendationState) -> ControllerDecision | None:
        """Rerank node: order eligible candidates, refusing to fabricate quality."""
        plan = state.current_plan
        pool = state.candidate_pool
        if plan is None or pool is None:  # pragma: no cover - retrieval always runs first
            raise ContractError("rerank requires a current plan and candidate pool")
        ranked: list[RankedProduct] = []
        if pool.candidates:
            try:
                ranked = list(self._deps.rerank(plan, pool))
            except RecommenderError as exc:
                self._emit("workflow.rerank_failed", {"error": exc.to_dict()})
                return ControllerDecision(
                    action=ControllerAction.STOP,
                    stop_reason=StopReason.RETRIEVAL_DEGRADED,
                    quality_satisfied=False,
                    budget_exhausted=False,
                    reasons=["reranker failed", f"error: {exc.message}"],
                    current_round=state.current_round,
                )
        state.ranked_products = ranked
        state.judgment_results = []
        if pool.candidates and not ranked:
            self._emit("workflow.rerank_empty", {"eligible": len(pool.candidates)})
            return ControllerDecision(
                action=ControllerAction.STOP,
                stop_reason=StopReason.RETRIEVAL_DEGRADED,
                quality_satisfied=False,
                budget_exhausted=False,
                reasons=[
                    "reranker returned no ranked products for a non-empty candidate pool",
                    "degraded stop; no quality result is fabricated",
                ],
                current_round=state.current_round,
            )
        self._emit(
            "workflow.rerank_completed",
            {
                "ranked": len(ranked),
                "top_product_ids": [product.product_id for product in ranked[:10]],
            },
        )
        return None

    def step_judge(self, state: RecommendationState) -> ControllerDecision | None:
        """Judgment node; an exhausted batch stops with ``api_failed``/judgment (D4)."""
        plan = state.current_plan
        pool = state.candidate_pool
        if plan is None or pool is None:  # pragma: no cover - retrieval always runs first
            raise ContractError("judgment requires a current plan and candidate pool")
        candidates = (
            self._deps.candidates_from_pool(pool, state.ranked_products, plan)
            if state.ranked_products
            else []
        )
        if not candidates:
            state.judgment_results = []
            self._emit("workflow.judgment_completed", {"judged": 0, "product_ids": []})
            return None
        api_calls = self._deps.judge.batch_count(len(candidates))
        if state.budget_state.remaining_api_calls < api_calls:
            return self._controller.budget_stop(
                state,
                dimension="API",
                reason=f"judgment needs {api_calls} batch calls",
            )
        state.budget_state = state.budget_state.consume_api_calls(api_calls)
        try:
            judgment_batch = self._deps.judge.judge(
                request_id=state.request_id,
                user_request=state.user_request,
                plan=plan,
                candidates=candidates,
            )
        except JudgmentBatchError as exc:
            self._emit("workflow.judgment_failed", exc.to_dict())
            state.quality_metrics = self._controller.summarize(state)
            failure = FailureInfo(
                failed_component=FailedComponent.JUDGMENT,
                error_type=exc.error_kind,
                retryable=exc.error_kind in _RETRYABLE_ERROR_KINDS,
                attempts=exc.attempts,
                message=f"judgment batch {exc.batch_index}: {exc.message}",
            )
            return self._controller.decide(state, failure=failure)
        state.judgment_results = judgment_batch.records
        self._emit(
            "workflow.judgment_completed",
            {
                "judged": len(state.judgment_results),
                "product_ids": [
                    record.result.product_id for record in state.judgment_results
                ],
            },
        )
        return None

    def step_controller(self, state: RecommendationState) -> ControllerDecision:
        """Controller node: aggregate quality metrics and take the decision."""
        state.quality_metrics = self._controller.summarize(state)
        state.valid_product_count = state.quality_metrics.valid_product_count
        decision = self._controller.decide(state)
        state.action = decision.action
        state.decision = decision
        self._emit("workflow.controller_decision", decision.model_dump(mode="json"))
        return decision

    def step_replan(
        self,
        state: RecommendationState,
        decision: ControllerDecision,
    ) -> ControllerDecision | None:
        """Replanner node; returns a STOP decision when the loop must end."""
        if state.budget_state.remaining_replan_rounds <= 0:
            return self._controller.budget_stop(
                state, dimension="replan", reason="controller requested another replan"
            )
        if state.budget_state.remaining_api_calls < 1:
            return self._controller.budget_stop(
                state, dimension="API", reason="replanner needs one chat completion"
            )
        state.budget_state = (
            state.budget_state.consume_replan_round().consume_api_calls(1)
        )
        replan_input = self._build_replan_input(state, decision)
        try:
            result = self._deps.planner.plan_replan(replan_input)
        except PlanGenerationError as exc:
            self._emit("workflow.replan_failed", exc.to_dict())
            failure = FailureInfo(
                failed_component=FailedComponent.PLANNER,
                error_type=exc.error_kind,
                retryable=exc.error_kind in _RETRYABLE_ERROR_KINDS,
                attempts=exc.attempts,
                message=exc.message,
            )
            return ControllerDecision(
                action=ControllerAction.STOP,
                stop_reason=StopReason.API_FAILED,
                quality_satisfied=False,
                budget_exhausted=False,
                reasons=[
                    "replanner failed; the previous valid plan is retained and no new "
                    "retrieval round is executed (explicit degradation)",
                    f"planner_error: {exc.message}",
                ],
                current_round=state.current_round,
                failure=failure,
            )

        new_hashes = plan_query_hashes(result.plan)
        repeated = duplicate_queries(result.plan, state.seen_query_hashes)
        if new_hashes and len(repeated) == len(new_hashes):
            self._emit(
                "workflow.duplicate_query_stop",
                {
                    "plan_id": result.plan.plan_id,
                    "queries": result.plan.retrieval.e5_faiss.queries,
                },
            )
            return ControllerDecision(
                action=ControllerAction.STOP,
                stop_reason=StopReason.MAX_ROUNDS_REACHED,
                quality_satisfied=False,
                budget_exhausted=False,
                reasons=[
                    "loop guard: the replan repeated only previously executed E5 queries",
                    f"repeated_queries: {len(repeated)}",
                ],
                current_round=state.current_round,
            )
        if repeated:
            _LOGGER.warning(
                "replan partially repeated executed queries",
                extra={
                    "plan_id": result.plan.plan_id,
                    "repeated": len(repeated),
                    "total": len(new_hashes),
                },
            )

        state.current_plan = result.plan
        state.current_round += 1
        state.seen_query_hashes = _dedupe([*state.seen_query_hashes, *new_hashes])
        state.action = ControllerAction.REPLAN
        self._emit(
            "workflow.replan_generated",
            {
                "plan_id": result.plan.plan_id,
                "parent_plan_id": result.plan.parent_plan_id,
                "plan_version": result.plan.plan_version,
                "attempts": result.attempts,
                "prompt_version": result.prompt_version,
                "e5_queries": result.plan.retrieval.e5_faiss.queries,
                "change_summary": result.plan.change_summary,
                "current_round": state.current_round,
                "degraded": result.degraded,
                "plan_source": result.plan_source,
            },
        )
        return None

    def finalize(
        self,
        state: RecommendationState,
        decision: ControllerDecision,
    ) -> RecommendationState:
        """Terminal node: write the decision into the state and emit the stop event."""
        state.action = decision.action
        state.stop_reason = decision.stop_reason
        state.quality_satisfied = decision.quality_satisfied
        state.budget_exhausted = decision.budget_exhausted
        state.decision = decision
        if decision.action is ControllerAction.STOP:
            state.selected_products = self._select_products(state)
        self._emit(
            "workflow.finished",
            {
                "action": decision.action.value,
                "stop_reason": (
                    decision.stop_reason.value if decision.stop_reason is not None else None
                ),
                "quality_satisfied": decision.quality_satisfied,
                "budget_exhausted": decision.budget_exhausted,
                "current_round": state.current_round,
                "budget": state.budget_state.model_dump(mode="json"),
                "selected_products": state.selected_products,
                "failed_component": (
                    decision.failure.failed_component.value if decision.failure else None
                ),
                "reasons": decision.reasons,
            },
        )
        _LOGGER.info(
            "workflow finished",
            extra={
                "request_id": state.request_id,
                "action": decision.action.value,
                "stop_reason": (
                    decision.stop_reason.value if decision.stop_reason is not None else None
                ),
                "quality_satisfied": decision.quality_satisfied,
                "budget_exhausted": decision.budget_exhausted,
                "current_round": state.current_round,
                "selected_products": state.selected_products,
                "failed_component": (
                    decision.failure.failed_component.value if decision.failure else None
                ),
            },
        )
        return state

    # -- internals -----------------------------------------------------------
    def _generate_initial_plan(
        self,
        state: RecommendationState,
    ) -> ControllerDecision | None:
        if state.budget_state.remaining_api_calls < 1:
            return self._controller.budget_stop(
                state,
                dimension="API",
                reason="initial planner needs one chat completion",
            )
        state.budget_state = state.budget_state.consume_api_calls(1)
        planner_input = PlannerInput(
            user_request=state.user_request,
            user_memory=state.user_memory,
            session_history=state.session_history,
            system_capabilities=self._capabilities,
        )
        try:
            result = self._deps.planner.plan_initial(planner_input)
        except PlanGenerationError as exc:
            self._emit("workflow.initial_plan_failed", exc.to_dict())
            failure = FailureInfo(
                failed_component=FailedComponent.PLANNER,
                error_type=exc.error_kind,
                retryable=exc.error_kind in _RETRYABLE_ERROR_KINDS,
                attempts=exc.attempts,
                message=exc.message,
            )
            return ControllerDecision(
                action=ControllerAction.STOP,
                stop_reason=StopReason.API_FAILED,
                quality_satisfied=False,
                budget_exhausted=False,
                reasons=[
                    "initial planner failed; no plan was fabricated",
                    f"planner_error: {exc.message}",
                ],
                current_round=state.current_round,
                failure=failure,
            )
        state.current_plan = result.plan
        state.current_round = max(1, state.current_round)
        self._emit(
            "workflow.initial_plan_generated",
            {
                "plan_id": result.plan.plan_id,
                "plan_version": result.plan.plan_version,
                "attempts": result.attempts,
                "prompt_version": result.prompt_version,
                "latency_ms": round(result.latency_ms, 1),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "e5_queries": result.plan.retrieval.e5_faiss.queries,
                "degraded": result.degraded,
                "plan_source": result.plan_source,
            },
        )
        return None

    def _build_replan_input(
        self,
        state: RecommendationState,
        decision: ControllerDecision,
    ) -> ReplanInput:
        previous_plan = state.current_plan
        if previous_plan is None:  # pragma: no cover - caller guarantees a plan
            raise ContractError("replan requires a previous plan")
        summary = state.constraint_summary
        evaluated = (
            summary.evaluated_products if summary is not None else len(state.ranked_products)
        )
        judgment_passed = sum(
            1
            for record in state.judgment_results
            if record.result.match_assessment is not MatchAssessment.NO_MATCH
        )
        missing_attributes = _missing_attributes(state)
        execution = ExecutionFeedback(
            retrieved_candidates=evaluated,
            after_deduplication=evaluated,
            constraint_satisfied=summary.valid_product_count if summary is not None else 0,
            judgment_passed=judgment_passed,
            missing_attributes=missing_attributes,
            diversity_issue=False,
            notes=decision.reasons[:4],
        )
        feedback = ControllerFeedback(
            reason=decision.reasons[0] if decision.reasons else "insufficient quality",
            suggested_direction=_suggest_direction(state, missing_attributes),
            reasons=decision.reasons,
        )
        return ReplanInput(
            user_request=state.user_request,
            previous_plan=previous_plan,
            execution_feedback=execution,
            controller_feedback=feedback,
            budget_state=state.budget_state,
            user_memory=state.user_memory,
            session_history=state.session_history,
            system_capabilities=self._capabilities,
        )

    def _select_products(self, state: RecommendationState) -> list[str]:
        top_k = self._options.select_top_k or self._config.quality.required_count
        by_id = {
            record.result.product_id: record.result for record in state.judgment_results
        }
        selected: list[str] = []
        for product in state.ranked_products:
            judgment = by_id.get(product.product_id)
            if judgment is None:
                continue
            if judgment.match_assessment is MatchAssessment.NO_MATCH:
                continue
            if judgment.uncertainty is UncertaintyLevel.HIGH:
                continue
            selected.append(product.product_id)
            if len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def _remember_queries(state: RecommendationState, plan: RecommendationPlan) -> None:
        state.seen_query_hashes = _dedupe(
            [*state.seen_query_hashes, *plan_query_hashes(plan)]
        )

    def _emit(self, event: str, payload: dict[str, object]) -> None:
        if self._trace is not None:
            self._trace.emit(event, payload, stage="workflow")


def _langgraph_adapter() -> Any | None:
    """Return the LangGraph adapter module, or ``None`` when LangGraph is missing."""
    from recommender.controller import langgraph_adapter

    if not langgraph_adapter.langgraph_available():
        return None
    return langgraph_adapter


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _missing_attributes(state: RecommendationState) -> list[str]:
    seen: dict[str, None] = {}
    for record in state.judgment_results:
        for preference in record.result.preference_fit.unknown_preferences:
            seen.setdefault(preference, None)
        for attribute, evidence in record.result.attribute_evidence.items():
            if evidence.status is EvidenceStatus.UNKNOWN:
                seen.setdefault(attribute, None)
    return list(seen)[:20]


def _suggest_direction(state: RecommendationState, missing_attributes: list[str]) -> str:
    summary = state.constraint_summary
    if summary is not None and summary.valid_product_count == 0:
        return (
            "broaden and rephrase retrieval queries around the user's core need; "
            "do not drop or relax any hard constraint"
        )
    if missing_attributes:
        return (
            "retrieve products with explicit catalog evidence for: "
            + ", ".join(missing_attributes[:5])
        )
    return "increase recall coverage with new queries while keeping the same hard constraints"


def workflow_request_ids(state: RecommendationState) -> tuple[str, str]:
    """Return the request/trace identity used by the workflow (for CLI wiring)."""
    context = get_request_context()
    if context is not None:
        return context.request_id, context.trace_id
    return state.request_id, state.trace_id
