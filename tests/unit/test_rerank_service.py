"""Rerank service: eligible filtering, ordering and protocol validation (stub scorer)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from recommender.reranking import RerankService, eligible_candidates
from recommender.schemas.common import ConstraintStatus, RecallPath
from recommender.schemas.fusion import FusedCandidate
from recommender.utils.errors import ContractError

pytestmark = pytest.mark.unit


class _StubScorer:
    model_id = "stub/reranker"
    batch_size = 2

    def __init__(self, mapping: dict[str, float]) -> None:
        self.mapping = mapping
        self.calls: list[list[str]] = []

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append(list(documents))
        return [self.mapping[document] for document in documents]


def _candidate(pid: str, score: float = 0.01) -> FusedCandidate:
    return FusedCandidate(
        product_id=pid,
        fusion_score=score,
        retrieval_sources=[RecallPath.E5_FAISS],
        source_ranks={"e5_faiss": 1},
        raw_scores={"e5_faiss": 0.8},
        constraint_status=ConstraintStatus.PASS,
    )


def test_rerank_orders_by_score_and_keeps_sources() -> None:
    candidates = [_candidate("p1", 0.03), _candidate("p2", 0.02), _candidate("p3", 0.01)]
    texts = {"p1": "cream A", "p2": "cream B", "p3": "cream C"}
    scorer = _StubScorer({"cream A": 0.1, "cream B": 0.9, "cream C": 0.5})
    result = RerankService(scorer, texts).rerank("moisturizer", candidates)
    assert [product.product_id for product in result.ranked] == ["p2", "p3", "p1"]
    assert [product.rerank_rank for product in result.ranked] == [1, 2, 3]
    top = result.ranked[0]
    assert top.fusion_rank == 2
    assert top.fusion_score == 0.02
    assert top.retrieval_sources == [RecallPath.E5_FAISS]
    assert result.model_id == "stub/reranker"


def test_rerank_top_k_truncates() -> None:
    candidates = [_candidate("p1"), _candidate("p2")]
    scorer = _StubScorer({"a": 0.1, "b": 0.9})
    result = RerankService(scorer, {"p1": "a", "p2": "b"}).rerank("q", candidates, top_k=1)
    assert [product.product_id for product in result.ranked] == ["p2"]


def test_rerank_skips_candidates_without_text() -> None:
    candidates = [_candidate("p1"), _candidate("p2")]
    scorer = _StubScorer({"a": 0.9})
    result = RerankService(scorer, {"p1": "a"}).rerank("q", candidates)
    assert [product.product_id for product in result.ranked] == ["p1"]


def test_rerank_requires_text_and_candidates() -> None:
    scorer = _StubScorer({})
    with pytest.raises(ContractError):
        RerankService(scorer, {}).rerank("q", [_candidate("p1")])
    with pytest.raises(ContractError):
        RerankService(scorer, {"p1": "a"}).rerank("q", [])


def test_rerank_rejects_wrong_score_count() -> None:
    class _BadScorer(_StubScorer):
        def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
            return [0.5]

    with pytest.raises(ContractError):
        RerankService(_BadScorer({}), {"p1": "a", "p2": "b"}).rerank(
            "q", [_candidate("p1"), _candidate("p2")]
        )


def test_eligible_candidates_exclude_unknown_and_failed() -> None:
    pass_candidate = _candidate("p1")
    unknown = _candidate("p2").model_copy(update={"constraint_status": ConstraintStatus.UNKNOWN})
    no_constraints = _candidate("p3").model_copy(update={"constraint_status": None})
    assert [
        c.product_id for c in eligible_candidates([pass_candidate, unknown, no_constraints])
    ] == [
        "p1",
        "p3",
    ]
