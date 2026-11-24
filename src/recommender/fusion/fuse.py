"""Candidate fusion: cross-path RRF over the unified recall protocol.

Inputs are :class:`RecallResult` objects (TIGER and/or E5 + FAISS). Fusion maps every hit
to ``product_id`` (already the case in the protocol), de-duplicates, keeps per-source
ranks and raw scores, then applies cross-path RRF. Raw scores from different models are
never summed (AGENTS.md §2.5, module 3 §5.4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from recommender.fusion.rrf import reciprocal_rank_fusion
from recommender.observability.logging import get_logger
from recommender.schemas.common import RecallPath, RetrievalStatus
from recommender.schemas.fusion import FusedCandidate
from recommender.schemas.retrieval import RecallResult
from recommender.utils.errors import ContractError, RetrievalError

_LOGGER = get_logger(__name__)

CANONICAL_PATH_ORDER: tuple[RecallPath, ...] = (RecallPath.TIGER, RecallPath.E5_FAISS)


@dataclass(frozen=True)
class FusionOutcome:
    """Result of cross-path fusion plus diagnostics for the Controller/trace."""

    candidates: list[FusedCandidate]
    degraded_paths: list[str] = field(default_factory=list)
    dropped_by_limit: int = 0
    used_paths: list[str] = field(default_factory=list)


def _hit_lookup(result: RecallResult) -> dict[str, tuple[int, float | None]]:
    return {hit.product_id: (hit.rank, hit.raw_score) for hit in result.hits}


def fuse_paths(
    results: Sequence[RecallResult],
    *,
    rrf_k: int = 60,
    max_candidates: int = 300,
    weights: Mapping[RecallPath, float] | None = None,
) -> FusionOutcome:
    """Fuse ranked recall results with cross-path RRF.

    A failed path (``status != ok``) is excluded and reported in ``degraded_paths``;
    when every path failed the call raises :class:`RetrievalError` instead of returning
    an empty candidate list (AGENTS.md §5: API failures must not look like "no matches").
    """
    if not results:
        raise ContractError("fusion requires at least one recall result")
    if max_candidates <= 0:
        raise ContractError("max_candidates must be positive")
    seen_paths: set[RecallPath] = set()
    for result in results:
        if result.path in seen_paths:
            raise ContractError(f"duplicate recall result for path {result.path.value}")
        seen_paths.add(result.path)

    usable = [
        result
        for result in results
        if result.status is RetrievalStatus.OK
        or (result.status is RetrievalStatus.DEGRADED and result.hits)
    ]
    degraded = [result.path.value for result in results if result not in usable]
    if not usable:
        reasons = "; ".join(
            f"{result.path.value}: {result.error or 'unavailable'}" for result in results
        )
        raise RetrievalError(f"all retrieval paths are unavailable ({reasons})")

    ordered = sorted(usable, key=lambda result: CANONICAL_PATH_ORDER.index(result.path))
    rankings = [[hit.product_id for hit in result.hits] for result in ordered]
    weight_list = [float(weights.get(result.path, 1.0)) for result in ordered] if weights else None
    entries = reciprocal_rank_fusion(rankings, k=rrf_k, weights=weight_list)

    lookups = {result.path: _hit_lookup(result) for result in ordered}
    path_positions = {result.path: index for index, result in enumerate(ordered)}
    candidates: list[FusedCandidate] = []
    for entry in entries[:max_candidates]:
        sources = [
            path
            for path in CANONICAL_PATH_ORDER
            if path in path_positions and entry.ranks[path_positions[path]] is not None
        ]
        rank_map: dict[str, int] = {}
        for path in sources:
            rank = entry.ranks[path_positions[path]]
            if rank is not None:
                rank_map[path.value] = rank
        raw_scores = {
            path.value: score
            for path in sources
            if (score := lookups[path][entry.product_id][1]) is not None
        }
        candidates.append(
            FusedCandidate(
                product_id=entry.product_id,
                fusion_score=entry.score,
                retrieval_sources=sources,
                source_ranks=rank_map,
                raw_scores=raw_scores,
            )
        )

    dropped = max(0, len(entries) - len(candidates))
    _LOGGER.info(
        "candidates fused",
        extra={
            "paths": [result.path.value for result in ordered],
            "degraded_paths": degraded,
            "fused": len(candidates),
            "dropped_by_limit": dropped,
            "rrf_k": rrf_k,
        },
    )
    return FusionOutcome(
        candidates=candidates,
        degraded_paths=degraded,
        dropped_by_limit=dropped,
        used_paths=[result.path.value for result in ordered],
    )
