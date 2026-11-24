"""Retrieval, constraint, fusion and rerank contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from recommender.schemas.common import (
    ConstraintOperator,
    ConstraintStatus,
    RecallPath,
    RetrievalStatus,
)
from recommender.schemas.constraints import (
    ConstraintCheck,
    ConstraintEvaluation,
    ConstraintSummary,
    aggregate_constraint_status,
)
from recommender.schemas.fusion import CandidatePool, FusedCandidate
from recommender.schemas.rerank import RankedProduct, RerankResult
from recommender.schemas.retrieval import QueryRecallResult, RecallHit, RecallResult

pytestmark = pytest.mark.unit


def _check(status: ConstraintStatus, reason: str | None = None) -> ConstraintCheck:
    defaults = {
        ConstraintStatus.PASS: None,
        ConstraintStatus.FAIL: "price 41.0 exceeds 30.0",
        ConstraintStatus.UNKNOWN: "price missing in catalog",
    }
    return ConstraintCheck(
        field="price",
        operator=ConstraintOperator.LTE,
        expected=30.0,
        observed=None if status is ConstraintStatus.UNKNOWN else 25.0,
        status=status,
        reason=reason if reason is not None else defaults[status],
    )


def _fused(
    product_id: str,
    score: float,
    status: ConstraintStatus = ConstraintStatus.PASS,
    checks: list[ConstraintCheck] | None = None,
) -> FusedCandidate:
    if checks is None:
        checks = [_check(status)]
    return FusedCandidate(
        product_id=product_id,
        fusion_score=score,
        retrieval_sources=[RecallPath.TIGER, RecallPath.E5_FAISS],
        source_ranks={"tiger": 1, "e5_faiss": 2},
        raw_scores={"tiger": 0.9, "e5_faiss": 0.7},
        constraint_status=status,
        constraint_checks=checks,
    )


def test_aggregate_status_precedence() -> None:
    assert aggregate_constraint_status([]) is None
    assert aggregate_constraint_status([ConstraintStatus.PASS]) is ConstraintStatus.PASS
    assert (
        aggregate_constraint_status([ConstraintStatus.PASS, ConstraintStatus.UNKNOWN])
        is ConstraintStatus.UNKNOWN
    )
    assert (
        aggregate_constraint_status([ConstraintStatus.UNKNOWN, ConstraintStatus.FAIL])
        is ConstraintStatus.FAIL
    )


def test_failed_check_requires_reason() -> None:
    with pytest.raises(ValidationError):
        ConstraintCheck(
            field="price",
            operator=ConstraintOperator.LTE,
            expected=30.0,
            observed=41.0,
            status=ConstraintStatus.FAIL,
        )


def test_unknown_check_requires_reason() -> None:
    with pytest.raises(ValidationError):
        ConstraintCheck(
            field="price",
            operator=ConstraintOperator.LTE,
            expected=30.0,
            status=ConstraintStatus.UNKNOWN,
        )


def test_evaluation_status_must_match_checks() -> None:
    with pytest.raises(ValidationError):
        ConstraintEvaluation(
            product_id="p1", status=ConstraintStatus.FAIL, checks=[_check(ConstraintStatus.PASS)]
        )


def test_evaluation_without_constraints_is_null() -> None:
    evaluation = ConstraintEvaluation(product_id="p1")
    assert evaluation.status is None
    assert evaluation.unknown_fields == []
    with pytest.raises(ValidationError):
        ConstraintEvaluation(product_id="p1", status=ConstraintStatus.PASS)


def test_fused_candidate_source_ranks_must_match_sources() -> None:
    with pytest.raises(ValidationError):
        FusedCandidate(
            product_id="p1",
            fusion_score=0.03,
            retrieval_sources=[RecallPath.TIGER],
            source_ranks={"e5_faiss": 1},
        )


def test_fused_candidate_pass_cannot_hide_failed_check() -> None:
    with pytest.raises(ValidationError):
        _fused("p1", 0.03, status=ConstraintStatus.PASS, checks=[_check(ConstraintStatus.FAIL)])


def test_fused_candidate_unknown_status_requires_unknown_check() -> None:
    with pytest.raises(ValidationError):
        _fused("p1", 0.03, status=ConstraintStatus.UNKNOWN, checks=[_check(ConstraintStatus.PASS)])


def _summary(
    passed: int, failed: int, unknown: int, *, not_configured: int = 0
) -> ConstraintSummary:
    return ConstraintSummary(
        evaluated_products=passed + failed + unknown + not_configured,
        pass_count=passed,
        fail_count=failed,
        unknown_count=unknown,
        not_configured_count=not_configured,
        valid_product_count=passed + not_configured,
        constraint_fields=[] if not_configured else ["price"],
    )


def test_candidate_pool_partitions_and_ordering() -> None:
    pool = CandidatePool(
        request_id="req_1",
        candidates=[_fused("p2", 0.03), _fused("p1", 0.02)],
        unverified_candidates=[_fused("p3", 0.01, status=ConstraintStatus.UNKNOWN)],
        rejected_candidates=[_fused("p4", 0.005, status=ConstraintStatus.FAIL)],
        constraint_summary=_summary(passed=2, failed=1, unknown=1),
    )
    assert pool.product_ids == ["p2", "p1"]


def test_candidate_pool_without_configured_constraints() -> None:
    candidate = FusedCandidate(
        product_id="p1",
        fusion_score=0.03,
        retrieval_sources=[RecallPath.E5_FAISS],
        source_ranks={"e5_faiss": 1},
        raw_scores={"e5_faiss": 0.8},
        constraint_status=None,
        constraint_checks=[],
    )
    pool = CandidatePool(
        request_id="req_nc",
        candidates=[candidate],
        constraint_summary=_summary(passed=0, failed=0, unknown=0, not_configured=1),
    )
    assert pool.product_ids == ["p1"]


def test_candidate_pool_not_configured_must_have_null_status() -> None:
    with pytest.raises(ValidationError):
        CandidatePool(
            request_id="req_nc",
            candidates=[_fused("p1", 0.03)],
            constraint_summary=_summary(passed=0, failed=0, unknown=0, not_configured=1),
        )


def test_candidate_pool_rejects_duplicate_products() -> None:
    with pytest.raises(ValidationError):
        CandidatePool(
            request_id="req_1",
            candidates=[_fused("p1", 0.03)],
            unverified_candidates=[_fused("p1", 0.02, status=ConstraintStatus.UNKNOWN)],
            constraint_summary=_summary(passed=1, failed=0, unknown=1),
        )


def test_candidate_pool_rejects_wrong_status_partition() -> None:
    with pytest.raises(ValidationError):
        CandidatePool(
            request_id="req_1",
            candidates=[_fused("p1", 0.03, status=ConstraintStatus.UNKNOWN)],
            constraint_summary=_summary(passed=1, failed=0, unknown=0),
        )


def test_candidate_pool_summary_must_match() -> None:
    with pytest.raises(ValidationError):
        CandidatePool(
            request_id="req_1",
            candidates=[_fused("p1", 0.03)],
            constraint_summary=_summary(passed=2, failed=0, unknown=0),
        )


def test_candidate_pool_must_be_sorted() -> None:
    with pytest.raises(ValidationError):
        CandidatePool(
            request_id="req_1",
            candidates=[_fused("p1", 0.01), _fused("p2", 0.03)],
            constraint_summary=_summary(passed=2, failed=0, unknown=0),
        )


def test_tiger_hit_requires_semantic_id() -> None:
    with pytest.raises(ValidationError):
        RecallHit(product_id="p1", path=RecallPath.TIGER, rank=1, raw_score=0.9)


def test_e5_hit_requires_query_and_forbids_semantic_id() -> None:
    with pytest.raises(ValidationError):
        RecallHit(product_id="p1", path=RecallPath.E5_FAISS, rank=1)
    with pytest.raises(ValidationError):
        RecallHit(
            product_id="p1",
            path=RecallPath.E5_FAISS,
            rank=1,
            query="moisturizer",
            semantic_id=[1, 2],
        )


def test_recall_result_ranks_must_be_contiguous() -> None:
    hits = [
        RecallHit(product_id="p1", path=RecallPath.E5_FAISS, rank=1, query="q"),
        RecallHit(product_id="p2", path=RecallPath.E5_FAISS, rank=3, query="q"),
    ]
    with pytest.raises(ValidationError):
        RecallResult(path=RecallPath.E5_FAISS, hits=hits, query_count=1)


def test_recall_result_rejects_duplicates() -> None:
    hits = [
        RecallHit(product_id="p1", path=RecallPath.E5_FAISS, rank=1, query="q"),
        RecallHit(product_id="p1", path=RecallPath.E5_FAISS, rank=2, query="q"),
    ]
    with pytest.raises(ValidationError):
        RecallResult(path=RecallPath.E5_FAISS, hits=hits, query_count=1)


def test_recall_result_non_ok_requires_error() -> None:
    with pytest.raises(ValidationError):
        RecallResult(path=RecallPath.TIGER, status=RetrievalStatus.UNAVAILABLE)
    result = RecallResult(
        path=RecallPath.TIGER,
        status=RetrievalStatus.UNAVAILABLE,
        error="no interaction history for user",
    )
    assert result.hits == []


def test_successful_e5_result_reports_query_count() -> None:
    with pytest.raises(ValidationError):
        RecallResult(path=RecallPath.E5_FAISS, query_count=0)
    result = RecallResult(
        path=RecallPath.E5_FAISS,
        hits=[RecallHit(product_id="p1", path=RecallPath.E5_FAISS, rank=1, query="q")],
        query_count=1,
    )
    assert result.rank_of("p1") == 1


def test_query_recall_result_requires_matching_query() -> None:
    hit = RecallHit(product_id="p1", path=RecallPath.E5_FAISS, rank=1, query="other query")
    with pytest.raises(ValidationError):
        QueryRecallResult(query="moisturizer", hits=[hit])


def test_rerank_result_ordering_is_enforced() -> None:
    ranked = [
        RankedProduct(
            product_id="p1", rerank_score=1.0, rerank_rank=1, retrieval_sources=[RecallPath.TIGER]
        ),
        RankedProduct(
            product_id="p2", rerank_score=2.0, rerank_rank=2, retrieval_sources=[RecallPath.TIGER]
        ),
    ]
    with pytest.raises(ValidationError):
        RerankResult(model_id="Qwen/Qwen3-Reranker-0.6B", query="q", ranked=ranked)


def test_rerank_result_records_score_semantics() -> None:
    ranked = [
        RankedProduct(
            product_id="p1",
            rerank_score=2.0,
            rerank_rank=1,
            retrieval_sources=[RecallPath.E5_FAISS],
        ),
        RankedProduct(
            product_id="p2",
            rerank_score=1.0,
            rerank_rank=2,
            retrieval_sources=[RecallPath.E5_FAISS],
        ),
    ]
    result = RerankResult(model_id="Qwen/Qwen3-Reranker-0.6B", query="q", ranked=ranked)
    assert result.score_interpretation == "relevance_score_not_purchase_probability"
    assert result.product_ids == ["p1", "p2"]
