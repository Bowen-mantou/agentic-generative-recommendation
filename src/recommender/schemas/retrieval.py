"""Unified recall protocol for TIGER and E5 + FAISS.

Both paths must emit :class:`RecallHit` objects keyed by ``product_id`` so that
Candidate Fusion can merge them. Semantic IDs remain a TIGER-internal detail and are
carried only as diagnostic data (AGENTS.md §2.1).
"""

from __future__ import annotations

from pydantic import Field, model_validator
from typing_extensions import Self

from recommender.schemas.common import (
    ContractModel,
    NonEmptyStr,
    RecallPath,
    RetrievalStatus,
)


class RecallHit(ContractModel):
    """One recalled product from one path."""

    product_id: NonEmptyStr
    path: RecallPath
    rank: int = Field(ge=1)
    raw_score: float | None = Field(default=None, allow_inf_nan=False)
    query: str | None = None
    semantic_id: list[int] | None = None
    is_synthetic: bool = False

    @model_validator(mode="after")
    def _path_specific_fields(self) -> Self:
        if self.path is RecallPath.TIGER:
            if not self.semantic_id:
                raise ValueError("TIGER hits require a non-empty semantic_id")
            if any(token < 0 for token in self.semantic_id):
                raise ValueError("semantic_id tokens must be non-negative")
            if self.query is not None:
                raise ValueError("TIGER hits must not carry an E5 query")
        else:
            if not self.query:
                raise ValueError("E5 hits require the query that produced them")
            if self.semantic_id is not None:
                raise ValueError("semantic_id is TIGER-internal and must not appear on E5 hits")
        return self


class QueryRecallResult(ContractModel):
    """Per-query E5 result before the internal RRF merge."""

    query: NonEmptyStr
    path: RecallPath = RecallPath.E5_FAISS
    status: RetrievalStatus = RetrievalStatus.OK
    hits: list[RecallHit] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None

    @model_validator(mode="after")
    def _hits_are_consistent(self) -> Self:
        for hit in self.hits:
            if hit.query != self.query:
                raise ValueError("hit.query must equal QueryRecallResult.query")
            if hit.path is not RecallPath.E5_FAISS:
                raise ValueError("QueryRecallResult only carries E5 hits")
        if self.status is RetrievalStatus.OK and self.error is not None:
            raise ValueError("status=ok must not carry an error")
        if self.status is not RetrievalStatus.OK and not self.error:
            raise ValueError("non-ok status must explain the failure")
        return self


class RecallResult(ContractModel):
    """Ordered output of one retrieval path.

    ``hits`` are ranked 1..n after de-duplication; ``status`` makes failures explicit so
    a broken path can never masquerade as "no candidates" (AGENTS.md §7.10).
    """

    path: RecallPath
    status: RetrievalStatus = RetrievalStatus.OK
    hits: list[RecallHit] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0.0)
    query_count: int = Field(default=0, ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def _ranking_is_well_formed(self) -> Self:
        expected = list(range(1, len(self.hits) + 1))
        if [hit.rank for hit in self.hits] != expected:
            raise ValueError("hits must be ranked contiguously from 1 in list order")
        product_ids = [hit.product_id for hit in self.hits]
        if len(set(product_ids)) != len(product_ids):
            raise ValueError("RecallResult must be de-duplicated by product_id")
        if any(hit.path is not self.path for hit in self.hits):
            raise ValueError("every hit.path must match RecallResult.path")
        if self.status is RetrievalStatus.OK and self.error is not None:
            raise ValueError("status=ok must not carry an error")
        if self.status is not RetrievalStatus.OK and not self.error:
            raise ValueError("non-ok status must explain the failure")
        e5_ok = self.status is RetrievalStatus.OK and self.path is RecallPath.E5_FAISS
        if e5_ok and self.query_count < 1:
            raise ValueError("successful E5 results must report at least one executed query")
        return self

    @property
    def product_ids(self) -> list[str]:
        return [hit.product_id for hit in self.hits]

    def rank_of(self, product_id: str) -> int | None:
        for hit in self.hits:
            if hit.product_id == product_id:
                return hit.rank
        return None


class E5PathOutput(ContractModel):
    """Complete E5 path output: per-query results plus the internally fused ranking."""

    fused: RecallResult
    per_query: list[QueryRecallResult] = Field(min_length=1)

    @model_validator(mode="after")
    def _output_is_consistent(self) -> Self:
        if self.fused.path is not RecallPath.E5_FAISS:
            raise ValueError("fused result must be an E5 path result")
        if any(result.path is not RecallPath.E5_FAISS for result in self.per_query):
            raise ValueError("per_query results must be E5 path results")
        if self.fused.query_count != len(self.per_query):
            raise ValueError("fused.query_count must equal the number of per-query results")
        queries = [result.query for result in self.per_query]
        if len({query.casefold() for query in queries}) != len(queries):
            raise ValueError("per-query results must use unique queries")
        allowed = set(queries)
        for hit in self.fused.hits:
            if hit.query not in allowed:
                raise ValueError("fused hits must reference one of the executed queries")
        per_query_ids = {hit.product_id for result in self.per_query for hit in result.hits}
        unknown = {hit.product_id for hit in self.fused.hits} - per_query_ids
        if unknown:
            raise ValueError(f"fused ranking contains ids absent from per-query results: {unknown}")
        return self
