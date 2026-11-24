"""RecommendationPlan contract: structure, mode invariants and replan preservation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from recommender.schemas.common import BudgetTier, PlanMode, RecallPath, RetrievalStrategy
from recommender.schemas.plan import RecommendationPlan, validate_replan_preserves_hard_constraints

pytestmark = pytest.mark.unit


def test_initial_plan_is_valid(plan_payload: dict[str, Any]) -> None:
    plan = RecommendationPlan(**plan_payload)
    assert plan.mode is PlanMode.INITIAL
    assert plan.plan_version == 1
    assert plan.parent_plan_id is None
    assert plan.retrieval.enabled_paths() == [RecallPath.TIGER, RecallPath.E5_FAISS]
    assert plan.retrieval.strategy is RetrievalStrategy.HYBRID_BALANCED
    assert plan.retrieval.e5_faiss.candidate_budget is BudgetTier.STANDARD
    assert plan.constraints.hard[0].operator.value == "lte"


def test_parallel_top_level_retrieval_fields_are_rejected(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["retrieval_strategy"] = "hybrid_balanced"
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_initial_plan_cannot_have_parent(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["parent_plan_id"] = "plan_000"
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_initial_plan_must_be_version_one(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["plan_version"] = 2
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_initial_plan_cannot_have_change_summary(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["change_summary"] = ["tightened category"]
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_replan_requires_parent(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["mode"] = "replan"
    payload["plan_version"] = 2
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_replan_self_parent_is_rejected(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload.update({"mode": "replan", "plan_version": 2, "parent_plan_id": payload["plan_id"]})
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_replan_requires_version_two(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload.update({"mode": "replan", "plan_version": 1, "parent_plan_id": "plan_000"})
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_e5_enabled_requires_queries(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["retrieval"]["e5_faiss"]["queries"] = []
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_both_paths_disabled_is_rejected(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["retrieval"]["tiger"]["enabled"] = False
    payload["retrieval"]["e5_faiss"]["enabled"] = False
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_queries_are_normalised_and_deduplicated(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["retrieval"]["e5_faiss"]["queries"] = [
        "  lightweight   daily moisturizer ",
        "Lightweight Daily Moisturizer",
        "",
    ]
    plan = RecommendationPlan(**payload)
    assert plan.retrieval.e5_faiss.queries == ["lightweight daily moisturizer"]


def test_hard_constraint_in_operator_requires_list(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["constraints"]["hard"][0].update({"operator": "in", "value": 30})
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_hard_constraint_scalar_operator_rejects_list(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["constraints"]["hard"][0]["value"] = [30, 40]
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_hard_constraint_rejects_mapping_value(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["constraints"]["hard"][0]["value"] = {"max": 30}
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_clarification_requires_question(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["clarification"] = {"required": True, "question": None, "reason": "unclear"}
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_clarification_without_requirement_carries_no_question(
    plan_payload: dict[str, Any],
) -> None:
    payload = deepcopy(plan_payload)
    payload["clarification"] = {"required": False, "question": "Which skin type?", "reason": None}
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def test_schema_version_is_fixed(plan_payload: dict[str, Any]) -> None:
    payload = deepcopy(plan_payload)
    payload["schema_version"] = "2.0"
    with pytest.raises(ValidationError):
        RecommendationPlan(**payload)


def _replan_from(plan_payload: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(plan_payload)
    payload.update(
        {
            "plan_id": "plan_002",
            "parent_plan_id": plan_payload["plan_id"],
            "plan_version": 2,
            "mode": "replan",
            "change_summary": ["broadened product type"],
        }
    )
    return payload


def test_replan_preserving_constraints_is_accepted(plan_payload: dict[str, Any]) -> None:
    parent = RecommendationPlan(**plan_payload)
    child = RecommendationPlan(**_replan_from(plan_payload))
    validate_replan_preserves_hard_constraints(parent, child)


def test_replan_dropping_hard_constraint_is_rejected(plan_payload: dict[str, Any]) -> None:
    parent = RecommendationPlan(**plan_payload)
    payload = _replan_from(plan_payload)
    payload["constraints"]["hard"] = []
    child = RecommendationPlan(**payload)
    with pytest.raises(ValueError, match="dropped or rewrote"):
        validate_replan_preserves_hard_constraints(parent, child)


def test_replan_rewriting_hard_constraint_is_rejected(plan_payload: dict[str, Any]) -> None:
    parent = RecommendationPlan(**plan_payload)
    payload = _replan_from(plan_payload)
    payload["constraints"]["hard"][0]["value"] = 40
    child = RecommendationPlan(**payload)
    with pytest.raises(ValueError, match="dropped or rewrote"):
        validate_replan_preserves_hard_constraints(parent, child)


def test_replan_must_reference_known_parent(plan_payload: dict[str, Any]) -> None:
    parent = RecommendationPlan(**plan_payload)
    payload = _replan_from(plan_payload)
    payload["parent_plan_id"] = "plan_999"
    child = RecommendationPlan(**payload)
    with pytest.raises(ValueError, match="does not match"):
        validate_replan_preserves_hard_constraints(parent, child)
