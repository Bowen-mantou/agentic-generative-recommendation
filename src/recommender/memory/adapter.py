"""User-memory adapter contract and explicit unavailable fallback (Phase 6, Q7/R11).

Three layers are kept separate on purpose (AGENTS.md §2.2):

* :class:`MemoryAdapter` is the only interface the E2E service sees. It returns
  semantic-memory hits, never TIGER behavior sequences.
* :class:`Mem0MemoryAdapter` (imported lazily from :mod:`recommender.memory.mem0_adapter`)
  talks to Mem0 OSS + the independent Qdrant service.
* :class:`UnavailableMemoryAdapter` is returned when the memory backend is disabled or
  its dependencies are missing. ``read`` returns an empty list, but ``add`` **raises**:
  a configuration problem must never look like a successful memory write.

The factory never imports Mem0 eagerly, so the data contracts and unit tests keep
working without the optional ``memory`` extra installed. Importing this module performs
no network call and loads no model.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from recommender.config import MemoryConfig
from recommender.utils.errors import RecommenderError

# Modules required by :class:`~recommender.memory.mem0_adapter.Mem0MemoryAdapter`.
# They are checked with ``importlib.util.find_spec`` so that importing this contract
# module stays cheap and side-effect free.
MEMORY_REQUIRED_MODULES: tuple[str, ...] = ("mem0", "qdrant_client")


@dataclass(frozen=True)
class MemoryHit:
    """One semantic-memory hit returned by :meth:`MemoryAdapter.read`.

    ``score`` is the backend similarity score (higher is closer for the cosine Qdrant
    collection); it is **not** a calibrated relevance probability. ``metadata`` keeps
    the caller-provided metadata (including ``is_synthetic``) plus the entity fields the
    backend promotes to the payload top level.
    """

    memory_id: str
    text: str
    score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryUnavailableError(RecommenderError):
    """The memory backend could not execute the requested operation.

    Raises are used for writes (and for failed reads of a *configured* backend) so the
    caller can record an explicit degraded state. Returning an empty list instead would
    silently turn a broken memory service into "no known preferences", which is a
    different and misleading state.
    """

    code = "memory_unavailable"

    def __init__(
        self,
        message: str,
        *,
        backend: str = "mem0",
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.backend = backend
        self.reason = reason or message

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update({"backend": self.backend, "reason": self.reason})
        return payload


@runtime_checkable
class MemoryAdapter(Protocol):
    """Minimal semantic-memory interface required by the Phase 6 E2E path."""

    def read(self, user_id: str, query: str, limit: int = 5) -> list[MemoryHit]:
        """Return at most ``limit`` memories of ``user_id`` relevant to ``query``."""
        ...

    def add(
        self,
        *,
        user_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        is_synthetic: bool = False,
    ) -> MemoryHit:
        """Persist one semantic memory for ``user_id`` and return the stored hit."""
        ...

    def health(self) -> dict[str, Any]:
        """Report availability plus backend/version details (never a fake success)."""
        ...


@dataclass
class UnavailableMemoryAdapter:
    """Explicit degraded backend used when memory is disabled or cannot be loaded.

    ``read`` returns ``[]`` (the E2E may proceed with no long-term memory), while
    ``add`` raises :class:`MemoryUnavailableError` so no caller can believe a write
    happened. ``health()`` always reports ``available=False`` and the concrete reason.
    """

    reason: str
    backend: str = "unavailable"

    def read(self, user_id: str, query: str, limit: int = 5) -> list[MemoryHit]:
        return []

    def add(
        self,
        *,
        user_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        is_synthetic: bool = False,
    ) -> MemoryHit:
        raise MemoryUnavailableError(
            f"memory backend unavailable: {self.reason}",
            backend=self.backend,
            reason=self.reason,
        )

    def health(self) -> dict[str, Any]:
        return {
            "available": False,
            "backend": self.backend,
            "provider": "mem0",
            "reason": self.reason,
        }


def missing_memory_modules() -> list[str]:
    """Return the required modules that are not importable in this environment."""
    missing: list[str] = []
    for module in MEMORY_REQUIRED_MODULES:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):  # pragma: no cover - broken parent package
            found = False
        if not found:
            missing.append(module)
    return missing


def create_memory_adapter(config: MemoryConfig) -> MemoryAdapter:
    """Build the configured memory adapter, degrading explicitly when unavailable.

    * ``config.enabled=false`` → :class:`UnavailableMemoryAdapter` (Phase 0/1 default).
    * unsupported vector store / missing ``mem0ai``/``qdrant-client`` →
      :class:`UnavailableMemoryAdapter` with an actionable reason.
    * otherwise → :class:`~recommender.memory.mem0_adapter.Mem0MemoryAdapter`
      (initialization is lazy; the first ``health()``/``read()`` establishes whether the
      Qdrant collection is actually reachable).
    """
    if not config.enabled:
        return UnavailableMemoryAdapter(
            reason=(
                "memory is disabled (memory.enabled=false); no semantic memory is read or written"
            ),
            backend="disabled",
        )

    vector_store = config.mem0.vector_store
    if vector_store.provider != "qdrant":
        return UnavailableMemoryAdapter(
            reason=(
                "unsupported memory vector store provider "
                f"{vector_store.provider!r}; Phase 6 implements 'qdrant' only (Q7)"
            ),
            backend="unavailable",
        )

    missing = missing_memory_modules()
    if missing:
        return UnavailableMemoryAdapter(
            reason=(
                "memory dependencies are not installed: "
                + ", ".join(missing)
                + "; install the project 'memory' extra (mem0ai==2.2.1, qdrant-client)"
            ),
            backend="unavailable",
        )

    try:
        from recommender.memory.mem0_adapter import Mem0MemoryAdapter
    except ImportError as exc:  # pragma: no cover - packaging/dependency failure
        return UnavailableMemoryAdapter(
            reason=f"Mem0 adapter could not be imported: {exc}",
            backend="unavailable",
        )
    return Mem0MemoryAdapter(config=config)
