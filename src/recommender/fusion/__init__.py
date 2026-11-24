"""Candidate Fusion: cross-path RRF, constraint annotation and candidate pools (Phase 2)."""

from __future__ import annotations

from recommender.fusion.fuse import FusionOutcome, fuse_paths
from recommender.fusion.pool import build_candidate_pool
from recommender.fusion.rrf import RrfEntry, reciprocal_rank_fusion

__all__ = [
    "FusionOutcome",
    "RrfEntry",
    "build_candidate_pool",
    "fuse_paths",
    "reciprocal_rank_fusion",
]
