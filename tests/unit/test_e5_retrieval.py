"""Two-level RRF and the E5 multi-query retrieval service (stubbed encoder/index)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from recommender.fusion.rrf import reciprocal_rank_fusion
from recommender.retrieval.e5.service import E5Retriever
from recommender.schemas.common import RecallPath, RetrievalStatus
from recommender.utils.errors import ContractError

pytestmark = pytest.mark.unit


def test_rrf_scores_and_order() -> None:
    entries = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert [entry.product_id for entry in entries] == ["b", "a", "c"]
    by_id = {entry.product_id: entry for entry in entries}
    assert by_id["a"].score == pytest.approx(1 / 61)
    assert by_id["b"].score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["c"].score == pytest.approx(1 / 62)
    assert by_id["a"].ranks == (1, None)
    assert by_id["b"].ranks == (2, 1)


def test_rrf_weights() -> None:
    entries = reciprocal_rank_fusion([["a"], ["b"]], k=10, weights=[3.0, 1.0])
    assert [entry.product_id for entry in entries] == ["a", "b"]


def test_rrf_ties_break_deterministically() -> None:
    entries = reciprocal_rank_fusion([["b"], ["a"]], k=60)
    assert [entry.product_id for entry in entries] == ["a", "b"]


def test_rrf_rejects_duplicates_and_bad_parameters() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion([["a", "a"]])
    with pytest.raises(ValueError, match="k must be"):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError, match="weights length"):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])


def test_rrf_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []


class _StubEncoder:
    embedding_dim = 2

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        self.seen.append(list(texts))
        return np.ones((len(texts), 2), dtype=np.float32)


class _StubIndex:
    def __init__(self, size: int, responses: dict[int, tuple[list[int], list[float]]]) -> None:
        self._size = size
        self._responses = responses

    @property
    def size(self) -> int:
        return self._size

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query_count = vectors.shape[0]
        scores = np.full((query_count, top_k), -1.0, dtype=np.float32)
        ids = np.full((query_count, top_k), -1, dtype=np.int64)
        for query_index in range(query_count):
            row_ids, row_scores = self._responses[query_index]
            for position, (row, score) in enumerate(zip(row_ids, row_scores, strict=True)):
                ids[query_index, position] = row
                scores[query_index, position] = score
        return scores, ids


def _retriever() -> E5Retriever:
    products = ["p0", "p1", "p2", "p3"]
    responses = {
        0: ([0, 1], [0.90, 0.80]),
        1: ([1, 2, 0], [0.95, 0.70, 0.60]),
    }
    return E5Retriever(
        encoder=_StubEncoder(),
        index=_StubIndex(len(products), responses),
        product_ids=products,
        top_k_per_query=3,
        rrf_k=60,
        max_queries=5,
    )


def test_e5_multi_query_fusion() -> None:
    output = _retriever().search(["lightweight cream", "fragrance free cream"])
    assert [hit.product_id for hit in output.fused.hits] == ["p1", "p0", "p2"]
    assert output.fused.query_count == 2
    top = output.fused.hits[0]
    assert top.product_id == "p1"
    assert top.query == "fragrance free cream"
    assert top.raw_score == pytest.approx(0.95)
    assert [result.query for result in output.per_query] == [
        "lightweight cream",
        "fragrance free cream",
    ]
    assert output.per_query[0].hits[0].product_id == "p0"
    assert output.fused.status is RetrievalStatus.OK


def test_e5_queries_are_normalised_and_deduplicated() -> None:
    retriever = _retriever()
    output = retriever.search(["  lightweight   cream ", "Lightweight Cream"])
    assert [result.query for result in output.per_query] == ["lightweight cream"]
    assert output.fused.query_count == 1


def test_e5_rejects_empty_and_too_many_queries() -> None:
    retriever = _retriever()
    with pytest.raises(ContractError, match="at least one"):
        retriever.search(["   "])
    with pytest.raises(ContractError, match="max_queries"):
        retriever.search([f"q{i}" for i in range(6)])


def test_e5_mapping_must_match_index_size() -> None:
    with pytest.raises(ContractError, match="index size"):
        E5Retriever(
            encoder=_StubEncoder(),
            index=_StubIndex(3, {}),
            product_ids=["p0", "p1"],
        )


def test_e5_fused_hits_validate_against_per_query_results() -> None:
    output = _retriever().search(["q1", "q2"])
    assert output.fused.path is RecallPath.E5_FAISS
    assert all(hit.path is RecallPath.E5_FAISS for hit in output.fused.hits)
