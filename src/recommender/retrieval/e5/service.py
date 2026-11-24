"""E5 retrieval: multi-query FAISS search with internal RRF.

Flow (module 3 §4.3): each Planner query is encoded and searched independently, then
the per-query rankings are merged with RRF into one unique E5 ranking. Raw similarity
scores are kept for traceability but never summed across queries.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from recommender.fusion.rrf import reciprocal_rank_fusion
from recommender.observability.logging import get_logger
from recommender.retrieval.e5.encoder import QueryEncoder
from recommender.retrieval.e5.index import VectorIndex
from recommender.schemas.artifacts import E5IndexManifest
from recommender.schemas.common import RecallPath, RetrievalStatus
from recommender.schemas.retrieval import E5PathOutput, QueryRecallResult, RecallHit, RecallResult
from recommender.utils.errors import ContractError

_LOGGER = get_logger(__name__)


@dataclass
class E5Retriever:
    """Stateless E5 + FAISS retriever over one index bundle."""

    encoder: QueryEncoder
    index: VectorIndex
    product_ids: Sequence[str]
    manifest: E5IndexManifest | None = None
    top_k_per_query: int = 50
    rrf_k: int = 60
    max_queries: int = 5
    _product_id_list: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._product_id_list = list(self.product_ids)
        if not self._product_id_list:
            raise ContractError("E5 retriever requires a non-empty product id mapping")
        if len(set(self._product_id_list)) != len(self._product_id_list):
            raise ContractError("product id mapping contains duplicates")
        if self.index.size != len(self._product_id_list):
            raise ContractError(
                f"index size {self.index.size} != id mapping size {len(self._product_id_list)}"
            )
        if self.top_k_per_query <= 0 or self.rrf_k < 1 or self.max_queries < 1:
            raise ContractError("invalid E5 retrieval configuration")

    @property
    def product_id_for_row(self) -> list[str]:
        return self._product_id_list

    def normalize_queries(self, queries: Sequence[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = " ".join(query.split())
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        if not cleaned:
            raise ContractError("E5 retrieval requires at least one non-empty query")
        if len(cleaned) > self.max_queries:
            raise ContractError(
                f"E5 retrieval received {len(cleaned)} queries but max_queries={self.max_queries}"
            )
        return cleaned

    def search(
        self,
        queries: Sequence[str],
        *,
        top_k_per_query: int | None = None,
    ) -> E5PathOutput:
        """Run every query, then merge with internal RRF into one E5 ranking."""
        cleaned = self.normalize_queries(queries)
        k = top_k_per_query if top_k_per_query is not None else self.top_k_per_query
        if k <= 0:
            raise ContractError("top_k_per_query must be positive")

        started = time.perf_counter()
        vectors = np.asarray(self.encoder.encode_queries(cleaned), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(cleaned):
            raise ContractError(f"encoder returned {vectors.shape} for {len(cleaned)} queries")
        scores, row_ids = self.index.search(vectors, k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        per_query: list[QueryRecallResult] = []
        for query_index, query in enumerate(cleaned):
            hits: list[RecallHit] = []
            seen: set[str] = set()
            for position in range(row_ids.shape[1]):
                row = int(row_ids[query_index, position])
                if row < 0 or row >= len(self._product_id_list):
                    continue
                product_id = self._product_id_list[row]
                if product_id in seen:
                    continue
                seen.add(product_id)
                hits.append(
                    RecallHit(
                        product_id=product_id,
                        path=RecallPath.E5_FAISS,
                        rank=len(hits) + 1,
                        raw_score=float(scores[query_index, position]),
                        query=query,
                    )
                )
            per_query.append(
                QueryRecallResult(
                    query=query,
                    hits=hits,
                    # FAISS batches the queries; the batch latency is recorded once on
                    # the fused result instead of being invented per query.
                    latency_ms=0.0,
                    status=RetrievalStatus.OK,
                )
            )

        rankings = [[hit.product_id for hit in result.hits] for result in per_query]
        fused_entries = reciprocal_rank_fusion(rankings, k=self.rrf_k)
        fused_hits: list[RecallHit] = []
        for fused_rank, entry in enumerate(fused_entries, start=1):
            present = [
                (rank, query_index)
                for query_index, rank in enumerate(entry.ranks)
                if rank is not None
            ]
            _, best_query_index = min(present)
            representative = next(
                hit
                for hit in per_query[best_query_index].hits
                if hit.product_id == entry.product_id
            )
            fused_hits.append(
                RecallHit(
                    product_id=entry.product_id,
                    path=RecallPath.E5_FAISS,
                    rank=fused_rank,
                    raw_score=representative.raw_score,
                    query=representative.query,
                )
            )

        fused = RecallResult(
            path=RecallPath.E5_FAISS,
            status=RetrievalStatus.OK,
            hits=fused_hits,
            latency_ms=elapsed_ms,
            query_count=len(cleaned),
        )
        _LOGGER.info(
            "e5 retrieval finished",
            extra={
                "queries": len(cleaned),
                "fused_hits": len(fused_hits),
                "latency_ms": round(elapsed_ms, 2),
                "rrf_k": self.rrf_k,
            },
        )
        return E5PathOutput(fused=fused, per_query=per_query)
