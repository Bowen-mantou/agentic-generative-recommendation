"""Rerank service: candidate pool -> Qwen3 relevance ranking.

Only eligible candidates (hard constraints ``pass`` or "not configured") are reranked;
``unknown`` products are never treated as satisfying constraints (Q3/Q5). Missing
document text is skipped explicitly instead of being scored with invented content.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from recommender.observability.logging import get_logger
from recommender.reranking.qwen import PairScorer
from recommender.schemas.common import ConstraintStatus
from recommender.schemas.fusion import FusedCandidate
from recommender.schemas.rerank import RankedProduct, RerankResult
from recommender.utils.errors import ContractError

_LOGGER = get_logger(__name__)


def eligible_candidates(candidates: Sequence[FusedCandidate]) -> list[FusedCandidate]:
    """Candidates the reranker is allowed to see (pass, or no constraint configured)."""
    return [
        candidate
        for candidate in candidates
        if candidate.constraint_status in (ConstraintStatus.PASS, None)
    ]


class RerankService:
    """Scores eligible candidates with a pair scorer and returns a validated ranking."""

    def __init__(self, scorer: PairScorer, texts: Mapping[str, str]) -> None:
        self.scorer = scorer
        self.texts = texts

    def rerank(
        self,
        query: str,
        candidates: Sequence[FusedCandidate],
        *,
        top_k: int | None = None,
    ) -> RerankResult:
        if not query.strip():
            raise ContractError("rerank query must not be empty")
        if not candidates:
            raise ContractError("rerank requires at least one eligible candidate")
        if top_k is not None and top_k <= 0:
            raise ContractError("top_k must be positive when provided")

        documents: list[str] = []
        kept: list[FusedCandidate] = []
        missing: list[str] = []
        for candidate in candidates:
            text = self.texts.get(candidate.product_id)
            if not text:
                missing.append(candidate.product_id)
                continue
            documents.append(text)
            kept.append(candidate)
        if not documents:
            raise ContractError("no candidate has catalog rerank text")
        if missing:
            _LOGGER.warning(
                "candidates without rerank text skipped",
                extra={"missing": missing[:20], "missing_count": len(missing)},
            )

        started = time.perf_counter()
        scores = self.scorer.score_pairs(query, documents)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if len(scores) != len(kept):
            raise ContractError(f"scorer returned {len(scores)} scores for {len(kept)} documents")
        if any(score != score for score in scores):  # NaN guard
            raise ContractError("scorer returned NaN relevance score")

        ordered = sorted(
            zip(kept, scores, strict=True),
            key=lambda item: (-item[1], item[0].product_id),
        )
        ranked: list[RankedProduct] = []
        for rank, (candidate, score) in enumerate(ordered, start=1):
            if top_k is not None and rank > top_k:
                break
            ranked.append(
                RankedProduct(
                    product_id=candidate.product_id,
                    rerank_score=float(score),
                    rerank_rank=rank,
                    fusion_score=candidate.fusion_score,
                    fusion_rank=next(
                        index
                        for index, item in enumerate(candidates, start=1)
                        if item.product_id == candidate.product_id
                    ),
                    retrieval_sources=candidate.retrieval_sources,
                )
            )
        result = RerankResult(
            model_id=self.scorer.model_id,
            query=query,
            instruction=None,
            ranked=ranked,
            batch_size=self.scorer.batch_size,
            latency_ms=elapsed_ms,
        )
        _LOGGER.info(
            "candidates reranked",
            extra={
                "query": query[:80],
                "scored": len(kept),
                "missing_text": len(missing),
                "latency_ms": round(elapsed_ms, 1),
                "model": self.scorer.model_id,
            },
        )
        return result
