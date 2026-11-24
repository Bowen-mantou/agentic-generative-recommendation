"""E5 + FAISS semantic retrieval (Phase 1)."""

from __future__ import annotations

from recommender.retrieval.e5.artifacts import (
    build_bundle,
    load_retriever,
    read_manifest,
    verify_bundle,
)
from recommender.retrieval.e5.encoder import E5Encoder
from recommender.retrieval.e5.service import E5Retriever

__all__ = [
    "E5Encoder",
    "E5Retriever",
    "build_bundle",
    "load_retriever",
    "read_manifest",
    "verify_bundle",
]
