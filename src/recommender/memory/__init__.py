"""Mem0 adapter and memory-write rules (Phase 6).

The public surface is :func:`create_memory_adapter` plus the
:class:`~recommender.memory.adapter.MemoryAdapter` protocol. Importing this package does
not import Mem0, load an embedding model or contact Qdrant.
"""

from __future__ import annotations

from recommender.memory.adapter import (
    MemoryAdapter,
    MemoryHit,
    MemoryUnavailableError,
    UnavailableMemoryAdapter,
    create_memory_adapter,
    missing_memory_modules,
)
from recommender.memory.mem0_adapter import E5MemoryEmbedder, Mem0MemoryAdapter

__all__ = [
    "E5MemoryEmbedder",
    "Mem0MemoryAdapter",
    "MemoryAdapter",
    "MemoryHit",
    "MemoryUnavailableError",
    "UnavailableMemoryAdapter",
    "create_memory_adapter",
    "missing_memory_modules",
]
