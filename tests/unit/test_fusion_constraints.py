"""Cross-path fusion, the single constraint evaluator and candidate-pool assembly."""

from __future__ import annotations

import pytest

from recommender.constraints import ConstraintEvaluator, evaluate_constraint, resolve_attribute
from recommender.fusion import build_candidate_pool, fuse_paths
from recommender.schemas.common import (
    ConstraintOperator,
    ConstraintStatus,
    RecallPath,
    RetrievalStatus,
)
from recommender.schemas.plan import HardConstraint
from recommender.schemas.product import ProductRecord
from recommender.schemas.retrieval import RecallHit, RecallResult
from recommender.utils.errors import ContractError, RetrievalError

pytestmark = pytest.mark.unit


def _e5_result(ids: list[str]) -> RecallResult:
    hits = [
        RecallHit(
            product_id=pid,
            path=RecallPath.E5_FAISS,
            rank=index + 1,
            raw_score=0.9 - index * 0.01,
            query="q",
        )
        for index, pid in enumerate(ids)
    ]
    return RecallResult(path=RecallPath.E5_FAISS, hits=hits, query_count=1)


def _tiger_result(ids: list[str]) -> RecallResult:
    hits = [
        RecallHit(
            product_id=pid,
            path=RecallPath.TIGER,
            rank=index + 1,
            raw_score=0.5 - index * 0.01,
            semantic_id=[1, 2, index],
        )
        for index, pid in enumerate(ids)
    ]
    return RecallResult(path=RecallPath.TIGER, hits=hits)


def _product(**overrides: object) -> ProductRecord:
    payload: dict[str, object] = {
        "product_id": "p1",
        "parent_asin": "p1",
        "title": "Daily Moisturizing Cream",
        "main_category": "All Beauty",
        "brand": "Brand A",
        "price": 25.99,
        "currency": "USD",
        "details": {"Scent": "Unscented", "Item Form": "Cream"},
    }
    payload.update(overrides)
    return ProductRecord(**payload)


def _constraint(
    field: str, operator: ConstraintOperator, value: object, unit: str | None = None
) -> HardConstraint:
    return HardConstraint(field=field, operator=operator, value=value, unit=unit)


def test_fusion_orders_by_cross_path_rrf() -> None:
    outcome = fuse_paths(
        [_tiger_result(["A", "B", "C"]), _e5_result(["B", "E", "F", "A"])], rrf_k=60
    )
    # RRF scores: B>A>E>C=F; ties break by product_id, so C precedes F.
    assert [c.product_id for c in outcome.candidates] == ["B", "A", "E", "C", "F"]
    by_id = {c.product_id: c for c in outcome.candidates}
    assert by_id["A"].retrieval_sources == [RecallPath.TIGER, RecallPath.E5_FAISS]
    assert by_id["A"].source_ranks == {"tiger": 1, "e5_faiss": 4}
    assert by_id["A"].raw_scores == {"tiger": 0.5, "e5_faiss": 0.87}
    assert by_id["C"].retrieval_sources == [RecallPath.TIGER]


def test_fusion_rejects_duplicate_paths_and_empty_input() -> None:
    with pytest.raises(ContractError):
        fuse_paths([_e5_result(["A"]), _e5_result(["B"])])
    with pytest.raises(ContractError):
        fuse_paths([])


def test_fusion_degrades_when_one_path_fails() -> None:
    failed = RecallResult(
        path=RecallPath.TIGER, status=RetrievalStatus.UNAVAILABLE, error="no history"
    )
    outcome = fuse_paths([failed, _e5_result(["A", "B"])])
    assert outcome.degraded_paths == ["tiger"]
    assert [c.product_id for c in outcome.candidates] == ["A", "B"]


def test_fusion_raises_when_all_paths_fail() -> None:
    with pytest.raises(RetrievalError):
        fuse_paths(
            [
                RecallResult(path=RecallPath.TIGER, status=RetrievalStatus.UNAVAILABLE, error="x"),
                RecallResult(
                    path=RecallPath.E5_FAISS, status=RetrievalStatus.UNAVAILABLE, error="y"
                ),
            ]
        )


def test_fusion_respects_candidate_limit() -> None:
    outcome = fuse_paths([_e5_result(["A", "B", "C"])], max_candidates=2)
    assert len(outcome.candidates) == 2
    assert outcome.dropped_by_limit == 1


def test_constraint_price_pass_fail_unknown() -> None:
    product = _product(price=25.99)
    passing = evaluate_constraint(_constraint("price", ConstraintOperator.LTE, 30, "USD"), product)
    failing = evaluate_constraint(_constraint("price", ConstraintOperator.LTE, 20, "USD"), product)
    missing = evaluate_constraint(
        _constraint("price", ConstraintOperator.LTE, 30, "USD"), _product(price=None)
    )
    assert passing.status is ConstraintStatus.PASS
    assert failing.status is ConstraintStatus.FAIL and "does not satisfy" in (failing.reason or "")
    assert missing.status is ConstraintStatus.UNKNOWN


def test_constraint_category_and_details() -> None:
    product = _product()
    assert (
        evaluate_constraint(
            _constraint("category", ConstraintOperator.EQ, "All Beauty"), product
        ).status
        is ConstraintStatus.PASS
    )
    assert (
        evaluate_constraint(
            _constraint("main_category", ConstraintOperator.EQ, "Premium Beauty"), product
        ).status
        is ConstraintStatus.FAIL
    )
    assert (
        evaluate_constraint(
            _constraint("Scent", ConstraintOperator.EQ, "unscented"), product
        ).status
        is ConstraintStatus.PASS
    )
    assert (
        evaluate_constraint(_constraint("Scent", ConstraintOperator.EQ, "Citrus"), product).status
        is ConstraintStatus.FAIL
    )
    assert (
        evaluate_constraint(
            _constraint("Unknown Attribute", ConstraintOperator.EQ, "x"), product
        ).status
        is ConstraintStatus.UNKNOWN
    )


def test_constraint_in_contains_and_currency() -> None:
    product = _product()
    assert (
        evaluate_constraint(
            _constraint("main_category", ConstraintOperator.IN, ["All Beauty", "Premium Beauty"]),
            product,
        ).status
        is ConstraintStatus.PASS
    )
    assert (
        evaluate_constraint(
            _constraint("main_category", ConstraintOperator.NOT_IN, ["All Beauty"]), product
        ).status
        is ConstraintStatus.FAIL
    )
    assert (
        evaluate_constraint(
            _constraint("title", ConstraintOperator.CONTAINS, "moisturizing"), product
        ).status
        is ConstraintStatus.PASS
    )
    assert (
        evaluate_constraint(_constraint("price", ConstraintOperator.LTE, 30, "EUR"), product).status
        is ConstraintStatus.UNKNOWN
    )


def test_constraint_non_numeric_comparison_is_unknown() -> None:
    check = evaluate_constraint(_constraint("brand", ConstraintOperator.GTE, 3), _product())
    assert check.status is ConstraintStatus.UNKNOWN
    assert "non-numeric" in (check.reason or "")


def test_constraint_evaluation_without_constraints_is_null() -> None:
    evaluation = ConstraintEvaluator([]).evaluate("p1", _product())
    assert evaluation.status is None
    assert evaluation.checks == []


def test_resolve_attribute_never_infers() -> None:
    assert resolve_attribute(None, "price").found is False
    assert resolve_attribute(_product(), "price").value == 25.99
    assert resolve_attribute(_product(), "missing").found is False


def test_candidate_pool_partitions_and_summary() -> None:
    outcome = fuse_paths([_e5_result(["cheap", "expensive", "unknown_price"])])
    products = {
        "cheap": _product(product_id="cheap", parent_asin="cheap", price=10.0),
        "expensive": _product(product_id="expensive", parent_asin="expensive", price=100.0),
        "unknown_price": _product(
            product_id="unknown_price", parent_asin="unknown_price", price=None
        ),
    }
    pool = build_candidate_pool(
        request_id="req_1",
        outcome=outcome,
        constraints=[_constraint("price", ConstraintOperator.LTE, 30, "USD")],
        products=products,
    )
    assert [c.product_id for c in pool.candidates] == ["cheap"]
    assert [c.product_id for c in pool.rejected_candidates] == ["expensive"]
    assert [c.product_id for c in pool.unverified_candidates] == ["unknown_price"]
    summary = pool.constraint_summary
    assert summary.pass_count == 1 and summary.fail_count == 1 and summary.unknown_count == 1
    assert summary.valid_product_count == 1
    assert pool.candidates[0].product_metadata["title"] == "Daily Moisturizing Cream"


def test_candidate_pool_without_constraints_is_not_configured() -> None:
    outcome = fuse_paths([_e5_result(["a", "b"])])
    pool = build_candidate_pool(
        request_id="req_2",
        outcome=outcome,
        constraints=[],
        products={"a": _product(product_id="a", parent_asin="a")},
    )
    assert pool.constraint_summary.not_configured_count == 2
    assert pool.constraint_summary.valid_product_count == 2
    assert all(candidate.constraint_status is None for candidate in pool.candidates)
