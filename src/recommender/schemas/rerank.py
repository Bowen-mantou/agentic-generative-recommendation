"""Reranker contracts.

Qwen3-Reranker-0.6B produces a *relevance* score, not a purchase probability; scores
are only comparable within one query (Reranker design doc §5.4, AGENTS.md §3).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from recommender.schemas.common import ContractModel, NonEmptyStr, RecallPath

ScoreSemantics = Literal["relevance_score_not_purchase_probability"]
SCORE_SEMANTICS: ScoreSemantics = "relevance_score_not_purchase_probability"


class RankedProduct(ContractModel):
    product_id: NonEmptyStr
    rerank_score: float = Field(allow_inf_nan=False)
    rerank_rank: int = Field(ge=1)
    fusion_score: float | None = Field(default=None, allow_inf_nan=False)
    fusion_rank: int | None = Field(default=None, ge=1)
    retrieval_sources: list[RecallPath] = Field(min_length=1)

    @model_validator(mode="after")
    def _sources_unique(self) -> Self:
        if len(set(self.retrieval_sources)) != len(self.retrieval_sources):
            raise ValueError("retrieval_sources must be unique")
        return self


class RerankResult(ContractModel):
    model_id: NonEmptyStr
    query: NonEmptyStr
    instruction: str | None = None
    ranked: list[RankedProduct] = Field(default_factory=list)
    batch_size: int = Field(default=1, ge=1)
    latency_ms: float = Field(default=0.0, ge=0.0)
    score_interpretation: Literal["relevance_score_not_purchase_probability"] = SCORE_SEMANTICS

    @model_validator(mode="after")
    def _ranking_is_well_formed(self) -> Self:
        expected = list(range(1, len(self.ranked) + 1))
        if [product.rerank_rank for product in self.ranked] != expected:
            raise ValueError("ranked products must be ranked contiguously from 1")
        ids = [product.product_id for product in self.ranked]
        if len(set(ids)) != len(ids):
            raise ValueError("ranked products must be unique")
        scores = [product.rerank_score for product in self.ranked]
        if scores != sorted(scores, reverse=True):
            raise ValueError("ranked products must be ordered by rerank_score desc")
        return self

    @property
    def product_ids(self) -> list[str]:
        return [product.product_id for product in self.ranked]
