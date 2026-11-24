"""Qwen3-Reranker-0.6B cross-encoder reranking (Phase 3, decision Q2)."""

from __future__ import annotations

from recommender.reranking.qwen import PairScorer, Qwen3Reranker
from recommender.reranking.service import RerankService, eligible_candidates

__all__ = ["PairScorer", "Qwen3Reranker", "RerankService", "eligible_candidates"]
