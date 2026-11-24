"""Two-level Reciprocal Rank Fusion.

Level 1 (inside E5): merge the ranked lists of several queries.
Level 2 (cross-path): merge the unique E5 ranking with the TIGER ranking.

Only ranks are combined; raw similarity/generation scores from different models are
never summed (AGENTS.md §2.5, module 3 §5.4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RrfEntry:
    product_id: str
    score: float
    ranks: tuple[int | None, ...]
    contributions: tuple[float, ...]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RrfEntry]:
    """Fuse ranked product-id lists with RRF.

    Args:
        rankings: each inner sequence is a de-duplicated ranking, best first.
        k: smoothing constant (must be >= 1). Smaller k sharpens rank differences.
        weights: optional per-ranking weights (default 1.0 each).

    Returns:
        Entries sorted by descending score, ties broken by ``product_id`` so the output
        is deterministic regardless of dictionary iteration order.
    """
    if k < 1:
        raise ValueError("RRF k must be >= 1")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights length must match the number of rankings")
    contributions_by_product: dict[str, list[float]] = {}
    ranks_by_product: dict[str, list[int | None]] = {}
    for ranking_index, ranking in enumerate(rankings):
        seen: set[str] = set()
        weight = 1.0 if weights is None else float(weights[ranking_index])
        for rank, product_id in enumerate(ranking, start=1):
            if not product_id:
                raise ValueError("ranking contains an empty product_id")
            if product_id in seen:
                raise ValueError(f"ranking {ranking_index} contains duplicate id {product_id}")
            seen.add(product_id)
            scores = contributions_by_product.setdefault(product_id, [0.0] * len(rankings))
            ranks = ranks_by_product.setdefault(product_id, [None] * len(rankings))
            contribution = weight / (k + rank)
            scores[ranking_index] += contribution
            ranks[ranking_index] = rank
    entries = [
        RrfEntry(
            product_id=product_id,
            score=sum(scores),
            ranks=tuple(ranks_by_product[product_id]),
            contributions=tuple(scores),
        )
        for product_id, scores in contributions_by_product.items()
    ]
    entries.sort(key=lambda entry: (-entry.score, entry.product_id))
    return entries
