"""Judgment, generator, controller and memory contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from recommender.schemas.common import (
    ControllerAction,
    EventType,
    MatchAssessment,
    MemoryAction,
    MemorySourceType,
    MemoryTargetStore,
    MemoryType,
    PreferenceFitLevel,
    StopReason,
    UncertaintyLevel,
)
from recommender.schemas.controller import BudgetState, ControllerDecision, QualityMetrics
from recommender.schemas.generation import RecommendationItem, RecommendationOutput
from recommender.schemas.judgment import (
    AttributeEvidence,
    JudgmentBatch,
    JudgmentLLMOutput,
    JudgmentRecord,
    PreferenceFit,
)
from recommender.schemas.memory import (
    InteractionEvent,
    MemoryRecord,
    MemorySource,
    MemoryUpdateOperation,
)
from recommender.utils.errors import ContractError

pytestmark = pytest.mark.unit


def _judgment(product_id: str = "p1") -> JudgmentLLMOutput:
    return JudgmentLLMOutput(
        product_id=product_id,
        preference_fit=PreferenceFit(
            level=PreferenceFitLevel.HIGH,
            matched_preferences=["lightweight"],
            unknown_preferences=["fragrance_free"],
        ),
        attribute_evidence={
            "lightweight": AttributeEvidence(status="supported", source="description"),
            "fragrance_free": AttributeEvidence(status="unknown", source=None),
        },
        match_assessment=MatchAssessment.PARTIAL_MATCH,
        uncertainty=UncertaintyLevel.MEDIUM,
        warnings=["fragrance_free 未得到商品信息支持"],
    )


def test_judgment_output_is_valid() -> None:
    output = _judgment()
    assert output.match_assessment is MatchAssessment.PARTIAL_MATCH
    assert output.attribute_evidence["fragrance_free"].source is None


def test_supported_evidence_requires_source() -> None:
    with pytest.raises(ValidationError):
        AttributeEvidence(status="supported", source=None)


def test_unknown_evidence_cannot_cite_source() -> None:
    with pytest.raises(ValidationError):
        AttributeEvidence(status="unknown", source="description")


def test_preference_sets_must_be_disjoint() -> None:
    with pytest.raises(ValidationError):
        PreferenceFit(
            level=PreferenceFitLevel.HIGH, matched_preferences=["x"], unknown_preferences=["x"]
        )


def test_judgment_batch_rejects_duplicate_products() -> None:
    record = JudgmentRecord(result=_judgment(), model="deepseek-chat", prompt_version="judgment.v1")
    with pytest.raises(ValidationError):
        JudgmentBatch(request_id="req_1", records=[record, record])


def test_empty_recommendation_needs_reason() -> None:
    with pytest.raises(ValidationError):
        RecommendationOutput(request_id="req_1", recommendations=[])


def test_recommendation_output_ranks_and_ids() -> None:
    item = RecommendationItem(
        rank=1,
        product_id="p1",
        title="Daily Moisturizing Cream",
        price=25.99,
        reasons=["价格 25.99 低于预算 30", "描述明确提到清爽质地"],
        limitations=["未确认是否无香型"],
    )
    output = RecommendationOutput(
        request_id="req_1",
        recommendations=[item],
        stop_reason=StopReason.BUDGET_EXHAUSTED,
        quality_satisfied=False,
        budget_exhausted=True,
    )
    assert output.product_ids == ["p1"]

    with pytest.raises(ValidationError):
        RecommendationOutput(
            request_id="req_1",
            recommendations=[item.model_copy(update={"rank": 2})],
            stop_reason=StopReason.QUALITY_SATISFIED,
            quality_satisfied=True,
        )


def test_budget_stop_cannot_claim_quality() -> None:
    with pytest.raises(ValidationError):
        RecommendationOutput(
            request_id="req_1",
            recommendations=[RecommendationItem(rank=1, product_id="p1", reasons=["ok"])],
            stop_reason=StopReason.BUDGET_EXHAUSTED,
            budget_exhausted=False,
        )


def test_budget_consumption_is_bounded() -> None:
    budget = BudgetState(
        remaining_replan_rounds=1, remaining_retrieval_calls=2, remaining_api_calls=3
    )
    assert budget.can_replan()
    reduced = budget.consume_replan_round().consume_retrieval_calls(2)
    assert reduced.remaining_replan_rounds == 0
    assert reduced.is_exhausted
    with pytest.raises(ContractError):
        reduced.consume_replan_round()
    with pytest.raises(ContractError):
        budget.consume_retrieval_calls(3)


def test_controller_decision_semantics() -> None:
    decision = ControllerDecision(
        action=ControllerAction.STOP,
        stop_reason=StopReason.BUDGET_EXHAUSTED,
        budget_exhausted=True,
        quality_satisfied=False,
        reasons=["budget exhausted"],
    )
    assert decision.stop_reason is StopReason.BUDGET_EXHAUSTED

    with pytest.raises(ValidationError):
        ControllerDecision(action=ControllerAction.STOP, quality_satisfied=True)
    with pytest.raises(ValidationError):
        ControllerDecision(
            action=ControllerAction.REPLAN,
            budget_exhausted=True,
            stop_reason=None,
        )
    with pytest.raises(ValidationError):
        ControllerDecision(
            action=ControllerAction.STOP,
            stop_reason=StopReason.QUALITY_SATISFIED,
            quality_satisfied=False,
        )


def test_quality_metrics_are_typed() -> None:
    metrics = QualityMetrics(
        valid_product_count=2,
        required_count=5,
        hard_constraints_satisfied=True,
        mean_rerank_score=1.5,
        match_distribution={"direct_match": 1, "partial_match": 1},
        evidence_coverage=0.5,
        unknown_attribute_ratio=0.5,
        diversity=0.4,
        quality_satisfied=False,
    )
    assert metrics.quality_satisfied is False


def test_interaction_event_requires_synthetic_flag() -> None:
    with pytest.raises(ValidationError):
        InteractionEvent(
            event_id="evt_1",
            user_id="u1",
            product_id="p1",
            event_type=EventType.REVIEW,
            timestamp=1_700_000_000,
            source="amazon_reviews_2023",
        )
    event = InteractionEvent(
        event_id="evt_1",
        user_id="u1",
        product_id="p1",
        event_type=EventType.REVIEW,
        rating=5,
        timestamp=1_700_000_000,
        source="amazon_reviews_2023",
        source_id="review_1",
        is_synthetic=False,
    )
    assert event.rating == 5


def test_click_event_cannot_carry_rating() -> None:
    with pytest.raises(ValidationError):
        InteractionEvent(
            event_id="evt_1",
            user_id="u1",
            product_id="p1",
            event_type=EventType.CLICK,
            rating=5,
            timestamp=1,
            source="first_party",
            is_synthetic=True,
        )


def test_memory_record_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        MemoryRecord(
            memory_id="mem_1",
            user_id="u1",
            memory_type=MemoryType.LONG_TERM_PREFERENCE,
            content="偏好无香型",
            confidence=1.5,
            source=MemorySource(source_type=MemorySourceType.AMAZON_REVIEW, source_ids=["r1"]),
            created_at="2026-11-01T00:00:00Z",
            last_observed_at="2026-11-20T00:00:00Z",
            is_synthetic=False,
        )


def test_inference_cannot_create_long_term_preference() -> None:
    with pytest.raises(ValidationError):
        MemoryUpdateOperation(
            memory_type=MemoryType.LONG_TERM_PREFERENCE,
            action=MemoryAction.ADD,
            content="用户可能喜欢清爽型",
            source=MemorySourceType.SYSTEM_INFERENCE,
            target_store=MemoryTargetStore.MEM0,
            is_synthetic=False,
        )


def test_episodic_memory_must_go_to_sqlite() -> None:
    with pytest.raises(ValidationError):
        MemoryUpdateOperation(
            memory_type=MemoryType.EPISODIC,
            action=MemoryAction.ADD,
            content="用户查看 p1",
            source=MemorySourceType.EXPLICIT_USER_FEEDBACK,
            target_store=MemoryTargetStore.MEM0,
            is_synthetic=False,
        )
