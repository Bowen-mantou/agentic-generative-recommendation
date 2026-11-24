"""Phase 6 memory adapter tests (stub Mem0 object; no live Qdrant or model).

The real Mem0/Qdrant round-trip is exercised separately on the server; these unit tests
pin the adapter contract: user_id isolation, metadata / is_synthetic handling, explicit
unavailable degradation and read/add error paths.
"""

from __future__ import annotations

from typing import Any

import pytest

from recommender.config import MemoryConfig
from recommender.memory import (
    E5MemoryEmbedder,
    Mem0MemoryAdapter,
    MemoryHit,
    MemoryUnavailableError,
    UnavailableMemoryAdapter,
    create_memory_adapter,
)

pytestmark = pytest.mark.unit


def _config(*, enabled: bool = True, provider: str = "qdrant") -> MemoryConfig:
    return MemoryConfig.model_validate(
        {
            "enabled": enabled,
            "mem0": {
                "vector_store": {
                    "provider": provider,
                    "host": "127.0.0.1",
                    "port": 6333,
                    "collection": "agr_user_memory",
                },
                "embedder": {
                    "provider": "huggingface",
                    "model": "intfloat/multilingual-e5-base",
                },
            },
        }
    )


class StubMem0Memory:
    """Small stand-in for ``mem0.Memory`` recording every call."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self.search_results = results or []
        self.search_calls: list[dict[str, Any]] = []
        self.added: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.deleted_all: list[str] = []
        self._next_id = 0

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
    ) -> dict[str, Any]:
        self.search_calls.append({"query": query, "filters": filters, "top_k": top_k})
        return {"results": self.search_results}

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        infer: bool,
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self._next_id += 1
        self.added.append(
            {
                "messages": messages,
                "user_id": user_id,
                "infer": infer,
                "metadata": metadata,
            }
        )
        return [
            {
                "id": f"mem-{self._next_id}",
                "memory": messages[0]["content"],
                "event": "ADD",
                "role": "user",
            }
        ]

    def delete(self, memory_id: str) -> None:
        self.deleted.append(memory_id)

    def delete_all(self, *, user_id: str) -> None:
        self.deleted_all.append(user_id)


def _adapter(
    *,
    memory: Any | None = None,
    results: list[dict[str, Any]] | None = None,
) -> tuple[Mem0MemoryAdapter, StubMem0Memory]:
    stub = memory if memory is not None else StubMem0Memory(results)
    adapter = Mem0MemoryAdapter(
        config=_config(), memory=stub, history_db_path="artifacts/mem0/test_history.db"
    )
    return adapter, stub


# -- unavailable / explicit degradation --------------------------------------
def test_unavailable_adapter_reads_empty_but_never_fakes_writes() -> None:
    adapter = UnavailableMemoryAdapter(reason="unit-test reason")
    assert adapter.read("u1", "moisturizer") == []
    with pytest.raises(MemoryUnavailableError) as excinfo:
        adapter.add(user_id="u1", text="prefers fragrance free")
    assert "unit-test reason" in str(excinfo.value)
    health = adapter.health()
    assert health["available"] is False
    assert health["backend"] == "unavailable"
    assert health["reason"] == "unit-test reason"


def test_create_memory_adapter_disabled_returns_explicit_unavailable() -> None:
    adapter = create_memory_adapter(_config(enabled=False))
    assert isinstance(adapter, UnavailableMemoryAdapter)
    assert adapter.health()["available"] is False
    assert adapter.health()["backend"] == "disabled"
    with pytest.raises(MemoryUnavailableError):
        adapter.add(user_id="u1", text="x")


def test_create_memory_adapter_reports_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "recommender.memory.adapter.missing_memory_modules",
        lambda: ["mem0", "qdrant_client"],
    )
    adapter = create_memory_adapter(_config())
    assert isinstance(adapter, UnavailableMemoryAdapter)
    reason = adapter.health()["reason"]
    assert "mem0" in reason and "qdrant_client" in reason


def test_create_memory_adapter_rejects_non_qdrant_store() -> None:
    adapter = create_memory_adapter(_config(provider="pgvector"))
    assert isinstance(adapter, UnavailableMemoryAdapter)
    assert "qdrant" in adapter.health()["reason"]


def test_create_memory_adapter_returns_mem0_adapter_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("recommender.memory.adapter.missing_memory_modules", lambda: [])
    adapter = create_memory_adapter(_config())
    assert isinstance(adapter, Mem0MemoryAdapter)


def test_health_reports_init_failure_and_read_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = Mem0MemoryAdapter(config=_config(), history_db_path="artifacts/mem0/test.db")

    def broken_build(self: Mem0MemoryAdapter) -> Any:
        raise RuntimeError("qdrant refused connection")

    monkeypatch.setattr(Mem0MemoryAdapter, "_build_memory", broken_build)
    health = adapter.health()
    assert health["available"] is False
    assert "qdrant refused connection" in health["reason"]
    with pytest.raises(MemoryUnavailableError):
        adapter.read("u1", "query")


# -- read ---------------------------------------------------------------------
def test_read_filters_user_id_and_maps_hits() -> None:
    results = [
        {
            "id": "m1",
            "memory": "prefers fragrance-free moisturizer",
            "score": 0.93,
            "user_id": "u1",
            "created_at": "2026-11-24T10:00:00+00:00",
            "metadata": {"is_synthetic": False, "memory_type": "preference"},
        },
        {
            "id": "m2",
            "memory": "likes 4K monitors",
            "score": 0.71,
            "user_id": "u2",  # foreign payload: must be dropped even if the backend leaks it
            "metadata": {"is_synthetic": False},
        },
        {
            "id": "m3",
            "memory": "budget under $30",
            "score": None,
            "user_id": "u1",
            "metadata": {"is_synthetic": True},
        },
        {"id": "m4", "memory": "   ", "user_id": "u1"},
    ]
    adapter, stub = _adapter(results=results)
    hits = adapter.read("u1", "fragrance free", limit=3)
    assert [hit.memory_id for hit in hits] == ["m1", "m3"]
    assert isinstance(hits[0], MemoryHit)
    assert hits[0].score == pytest.approx(0.93)
    assert hits[0].metadata["is_synthetic"] is False
    assert hits[0].metadata["created_at"] == "2026-11-24T10:00:00+00:00"
    assert hits[1].score is None
    assert hits[1].metadata["is_synthetic"] is True
    assert stub.search_calls == [
        {"query": "fragrance free", "filters": {"user_id": "u1"}, "top_k": 3}
    ]


def test_read_validates_arguments() -> None:
    adapter, _ = _adapter()
    with pytest.raises(ValueError):
        adapter.read("", "q")
    with pytest.raises(ValueError):
        adapter.read("u1", "  ")
    with pytest.raises(ValueError):
        adapter.read("u1", "q", limit=0)


def test_read_raises_when_backend_fails() -> None:
    class Broken(StubMem0Memory):
        def search(self, *args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("qdrant down")

    adapter = Mem0MemoryAdapter(
        config=_config(), memory=Broken(), history_db_path="artifacts/mem0/test.db"
    )
    with pytest.raises(MemoryUnavailableError) as excinfo:
        adapter.read("u1", "query")
    assert "qdrant down" in str(excinfo.value)


# -- add ----------------------------------------------------------------------
def test_add_uses_infer_false_and_marks_synthetic() -> None:
    adapter, stub = _adapter()
    hit = adapter.add(
        user_id=" u1 ",
        text="  prefers fragrance-free products  ",
        metadata={"source": "explicit_feedback", "user_id": "attacker", "hash": "x"},
        is_synthetic=True,
    )
    assert stub.added == [
        {
            "messages": [{"role": "user", "content": "prefers fragrance-free products"}],
            "user_id": "u1",
            "infer": False,
            "metadata": {"source": "explicit_feedback", "is_synthetic": True},
        }
    ]
    assert hit.memory_id == "mem-1"
    assert hit.text == "prefers fragrance-free products"
    assert hit.score is None
    assert hit.metadata["is_synthetic"] is True
    assert "user_id" not in hit.metadata and "hash" not in hit.metadata


def test_add_raises_when_backend_returns_no_record() -> None:
    class Empty(StubMem0Memory):
        def add(self, *args: Any, **kwargs: Any) -> Any:
            return []

    adapter = Mem0MemoryAdapter(
        config=_config(), memory=Empty(), history_db_path="artifacts/mem0/test.db"
    )
    with pytest.raises(MemoryUnavailableError) as excinfo:
        adapter.add(user_id="u1", text="fact")
    assert "refusing to report a write" in str(excinfo.value)


def test_add_raises_when_backend_fails() -> None:
    class Broken(StubMem0Memory):
        def add(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("embedding failed")

    adapter = Mem0MemoryAdapter(
        config=_config(), memory=Broken(), history_db_path="artifacts/mem0/test.db"
    )
    with pytest.raises(MemoryUnavailableError) as excinfo:
        adapter.add(user_id="u1", text="fact")
    assert "embedding failed" in str(excinfo.value)


def test_add_validates_arguments() -> None:
    adapter, _ = _adapter()
    with pytest.raises(ValueError):
        adapter.add(user_id="", text="x")
    with pytest.raises(ValueError):
        adapter.add(user_id="u1", text="   ")


# -- maintenance / health -----------------------------------------------------
def test_delete_and_delete_all_delegate_to_mem0() -> None:
    adapter, stub = _adapter()
    adapter.delete("mem-9")
    adapter.delete_all("u1")
    assert stub.deleted == ["mem-9"]
    assert stub.deleted_all == ["u1"]


def test_health_reports_backend_versions_and_locked_embedding() -> None:
    adapter, _ = _adapter()
    health = adapter.health()
    assert health["available"] is True
    assert health["backend"] == "mem0"
    assert health["collection"] == "agr_user_memory"
    assert health["vector_store"]["host"] == "127.0.0.1"
    assert health["vector_store"]["port"] == 6333
    assert health["vector_store"]["embedding_dim"] == 768
    assert health["embedder"]["model"] == "intfloat/multilingual-e5-base"
    assert health["embedder"]["embedding_dim"] == 768
    assert health["embedder"]["memory_action_prefixes"]["search"] == "query: "
    assert health["embedder"]["memory_action_prefixes"]["add"] == "passage: "
    assert "filters" in health["user_id_isolation"]
    assert set(health["versions"]) == {"mem0ai", "qdrant-client"}


# -- E5 memory embedder -------------------------------------------------------
class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def embedding_dim(self) -> int:
        return 768

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("query", list(texts)))
        return [[0.1] * 768 for _ in texts]

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("passage", list(texts)))
        return [[0.2] * 768 for _ in texts]


def test_memory_embedder_uses_query_prefix_for_search_and_passage_for_writes() -> None:
    encoder = RecordingEncoder()
    embedder = E5MemoryEmbedder(encoder=encoder)  # type: ignore[arg-type]
    assert embedder.embed("fragrance free", "search") == pytest.approx([0.1] * 768)
    assert embedder.embed("user likes fragrance free", "add") == pytest.approx([0.2] * 768)
    assert embedder.embed("updated fact", "update") == pytest.approx([0.2] * 768)
    assert encoder.calls == [
        ("query", ["fragrance free"]),
        ("passage", ["user likes fragrance free"]),
        ("passage", ["updated fact"]),
    ]
    assert embedder.embedding_dim == 768


def test_memory_embedder_rejects_invalid_input_and_action() -> None:
    embedder = E5MemoryEmbedder(encoder=RecordingEncoder())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        embedder.embed("   ", "add")
    with pytest.raises(ValueError):
        embedder.embed("fact", "delete")
    with pytest.raises(ValueError):
        embedder.embed_batch([])
    with pytest.raises(ValueError):
        embedder.embed_batch(["ok", ""])
