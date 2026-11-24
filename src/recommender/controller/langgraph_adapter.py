"""LangGraph runtime adapter for the Phase 5 recommendation loop (decision D3).

The business logic never lives here: every node wraps one of the
:class:`~recommender.controller.workflow.RecommendationWorkflow` step functions
(``step_plan`` / ``step_retrieve`` / ``step_rerank`` / ``step_judge`` /
``step_controller`` / ``step_replan`` / ``finalize``). LangGraph only owns state passing
and the conditional transitions, exactly as the design in module 2 §7 and
final-stage doc §4.5 requires:

```text
START -> prepare -> planner -> clarification_gate -> retrieve -> rerank -> judge
              ^                                                        |
              |                                                        v
              +-- replanner <- (REPLAN) <- controller <- (terminal?) --+
                                                                       |
                                                                       v
                                                                   finalize -> END
```

``langgraph`` is an optional dependency (``pyproject.toml`` extra ``workflow``). Import
of this module never fails: :func:`langgraph_available` reports whether the runtime can
be built, and :class:`LangGraphUnavailableError` is raised instead of silently
pretending the formal runtime executed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypedDict

from recommender.schemas.common import ControllerAction
from recommender.schemas.controller import ControllerDecision, RecommendationState
from recommender.utils.errors import ContractError, RecommenderError

if TYPE_CHECKING:
    from recommender.controller.workflow import RecommendationWorkflow


class LangGraphUnavailableError(RecommenderError):
    """The formal LangGraph runtime was requested but ``langgraph`` is not installed."""

    code = "langgraph_unavailable"


class WorkflowGraphState(TypedDict):
    """State channels of the workflow graph.

    ``state`` is the validated cross-module :class:`RecommendationState`; ``decision``
    carries the terminal Controller decision between a node and the router. Both are
    opaque values to LangGraph, so the strict Pydantic contracts stay authoritative.
    """

    state: RecommendationState
    decision: ControllerDecision | None


@lru_cache(maxsize=1)
def _runtime() -> dict[str, Any] | None:
    """Import the LangGraph runtime once; ``None`` when the extra is not installed."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return None
    runtime: dict[str, Any] = {"StateGraph": StateGraph, "START": START, "END": END}
    try:
        from langgraph.graph.state import CompiledStateGraph

        runtime["CompiledStateGraph"] = CompiledStateGraph
    except ImportError:  # pragma: no cover - very old langgraph layouts
        runtime["CompiledStateGraph"] = None
    return runtime


def langgraph_available() -> bool:
    """True when the optional LangGraph runtime can actually be imported."""
    return _runtime() is not None


def compiled_graph_class() -> type[Any] | None:
    """The real ``CompiledStateGraph`` class, for integration-test assertions."""
    runtime = _runtime()
    if runtime is None:
        return None
    return runtime["CompiledStateGraph"]


def build_graph(workflow: RecommendationWorkflow) -> Any:
    """Build and compile the LangGraph graph over the workflow step functions."""
    runtime = _runtime()
    if runtime is None:
        raise LangGraphUnavailableError(
            "langgraph is not installed; install the 'workflow' extra to run the "
            "formal Phase 5 runtime"
        )

    def node(step: Any) -> Any:
        def run_step(payload: WorkflowGraphState) -> WorkflowGraphState:
            state = payload["state"]
            decision = step(state)
            return {"state": state, "decision": decision}

        return run_step

    def replan_node(payload: WorkflowGraphState) -> WorkflowGraphState:
        state = payload["state"]
        decision = payload["decision"]
        if decision is None:  # pragma: no cover - routed only from a REPLAN decision
            raise ContractError("replanner node requires a controller decision")
        return {"state": state, "decision": workflow.step_replan(state, decision)}

    def finalize_node(payload: WorkflowGraphState) -> WorkflowGraphState:
        state = payload["state"]
        decision = payload["decision"]
        if decision is None:  # pragma: no cover - routed only with a terminal decision
            raise ContractError("finalize node requires a terminal decision")
        workflow.finalize(state, decision)
        return {"state": state, "decision": decision}

    def continue_to(next_node: str) -> Any:
        def route(payload: WorkflowGraphState) -> str:
            return "finalize" if payload["decision"] is not None else next_node

        return route

    def after_controller(payload: WorkflowGraphState) -> str:
        decision = payload["decision"]
        if decision is None:  # pragma: no cover - the controller always decides
            raise ContractError("controller node must produce a decision")
        return "replanner" if decision.action is ControllerAction.REPLAN else "finalize"

    state_graph = runtime["StateGraph"](WorkflowGraphState)
    state_graph.add_node("prepare", node(workflow.step_prepare))
    state_graph.add_node("planner", node(workflow.step_plan))
    state_graph.add_node("clarification_gate", node(workflow.step_clarification))
    state_graph.add_node("retrieve", node(workflow.step_retrieve))
    state_graph.add_node("rerank", node(workflow.step_rerank))
    state_graph.add_node("judge", node(workflow.step_judge))
    state_graph.add_node("controller", node(workflow.step_controller))
    state_graph.add_node("replanner", replan_node)
    state_graph.add_node("finalize", finalize_node)

    state_graph.add_edge(runtime["START"], "prepare")
    state_graph.add_edge("prepare", "planner")
    state_graph.add_conditional_edges(
        "planner",
        continue_to("clarification_gate"),
        {"clarification_gate": "clarification_gate", "finalize": "finalize"},
    )
    state_graph.add_conditional_edges(
        "clarification_gate",
        continue_to("retrieve"),
        {"retrieve": "retrieve", "finalize": "finalize"},
    )
    state_graph.add_conditional_edges(
        "retrieve",
        continue_to("rerank"),
        {"rerank": "rerank", "finalize": "finalize"},
    )
    state_graph.add_conditional_edges(
        "rerank",
        continue_to("judge"),
        {"judge": "judge", "finalize": "finalize"},
    )
    state_graph.add_conditional_edges(
        "judge",
        continue_to("controller"),
        {"controller": "controller", "finalize": "finalize"},
    )
    state_graph.add_conditional_edges(
        "controller",
        after_controller,
        {"replanner": "replanner", "finalize": "finalize"},
    )
    state_graph.add_conditional_edges(
        "replanner",
        continue_to("retrieve"),
        {"retrieve": "retrieve", "finalize": "finalize"},
    )
    state_graph.add_edge("finalize", runtime["END"])
    return state_graph.compile()


def _compiled_graph(workflow: RecommendationWorkflow) -> Any:
    """Return the cached compiled graph for ``workflow`` (build once per instance)."""
    graph = workflow._langgraph_graph
    if graph is None:
        graph = build_graph(workflow)
        workflow._langgraph_graph = graph
    return graph


def _recursion_limit(workflow: RecommendationWorkflow) -> int:
    """Generous but finite super-step limit derived from the configured loop guard."""
    return max(50, 12 * (workflow.config.loop_guard.max_rounds + 2))


def run_with_langgraph(
    workflow: RecommendationWorkflow,
    state: RecommendationState,
) -> RecommendationState:
    """Execute the formal LangGraph runtime and return the final recommendation state."""
    graph = _compiled_graph(workflow)
    result = graph.invoke(
        {"state": state, "decision": None},
        config={"recursion_limit": _recursion_limit(workflow)},
    )
    final_state = result["state"]
    if not isinstance(final_state, RecommendationState):  # pragma: no cover - defensive
        raise ContractError("LangGraph runtime did not return a RecommendationState")
    return final_state
