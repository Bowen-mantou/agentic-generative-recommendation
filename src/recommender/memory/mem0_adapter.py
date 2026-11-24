"""Mem0 OSS + Qdrant implementation of the user-memory adapter (Phase 6, Q7).

Design constraints from the D9 deployment report (``docs/phase6/D9_MEMORY_DEPLOYMENT.md``)
and AGENTS.md §2.2/§3:

* one Qdrant collection, every read filtered by ``user_id`` (R11);
* the project ``E5Encoder`` is wrapped so that ``add``/``update`` use the
  ``passage: `` prefix and ``search`` uses the ``query: `` prefix (E5 convention);
* the memory embedding model is ``intfloat/multilingual-e5-base`` (768 dimensions,
  commit ``d1287505…`` verified in D9) — a *different vector space* from the product
  ``intfloat/e5-base-v2`` index, so it must never share a collection with it (R8);
* Mem0's own SQLite history database is unrelated to the project InteractionEvent
  SQLite store and lives under ``artifacts/mem0/`` (never committed);
* no model or network call happens at import time. Mem0 is imported lazily inside
  :meth:`Mem0MemoryAdapter._build_memory` so the core package and unit tests keep
  working without the optional ``memory`` extra.

``is_synthetic`` is stored in the payload metadata on every write: simulated feedback
must stay identifiable and must never be mixed into real user memory (AGENTS.md §2.2).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from recommender.config import PROJECT_ROOT, MemoryConfig, Settings, load_settings
from recommender.memory.adapter import MemoryHit, MemoryUnavailableError
from recommender.observability.logging import get_logger
from recommender.retrieval.e5.encoder import E5Encoder
from recommender.utils.errors import ConfigError

_LOGGER = get_logger(__name__)

# Locked by D9 §6/§8: the memory collection uses the multilingual E5 model, 768 dims.
MEMORY_EMBEDDING_DIM = 768
MEMORY_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
# D9 §6.1 verified snapshot commit for intfloat/multilingual-e5-base.
MEMORY_EMBEDDING_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
DEFAULT_HISTORY_DB_PATH = "artifacts/mem0/history.db"
# Placeholder base URL never contacted: it makes mem0's HuggingFace embedder construct
# an OpenAI client instead of eagerly loading a second copy of the model. The real
# prefix-aware E5 embedder is injected immediately after ``Memory.from_config``
# (D9 §7.3, option 1: construct then replace ``memory.embedding_model``).
_PLACEHOLDER_EMBEDDER_BASE_URL = "http://127.0.0.1:9/placeholder"
_RESERVED_METADATA_KEYS = frozenset(
    {
        "user_id",
        "agent_id",
        "run_id",
        "data",
        "hash",
        "created_at",
        "updated_at",
        "text_lemmatized",
    }
)
_MEMORY_ACTIONS = frozenset({"add", "search", "update"})


class E5MemoryEmbedder:
    """Prefix-aware E5 embedder for Mem0 (``query: `` vs ``passage: ``).

    Mem0 calls ``embed(text, memory_action)`` for semantic operations and
    ``embed_batch(texts, memory_action)`` for batched writes. The D9 mapping is used:
    ``add``/``update`` → ``passage: ``, ``search`` → ``query: ``.
    """

    def __init__(
        self,
        *,
        model_id: str = MEMORY_EMBEDDING_MODEL,
        revision: str = MEMORY_EMBEDDING_REVISION,
        device: str = "cpu",
        expected_dim: int = MEMORY_EMBEDDING_DIM,
        encoder: E5Encoder | None = None,
    ) -> None:
        if expected_dim <= 0:
            raise ValueError("expected_dim must be positive")
        self.model_id = model_id
        self.revision = revision
        self.expected_dim = expected_dim
        self._encoder = encoder or E5Encoder(
            model_id=model_id,
            revision=revision,
            device=device,
            max_length=512,
            query_prefix="query: ",
            passage_prefix="passage: ",
            normalize=True,
            expected_dim=expected_dim,
        )

    @property
    def embedding_dim(self) -> int:
        return self._encoder.embedding_dim

    def _action(self, memory_action: str | None) -> str:
        action = (memory_action or "add").strip().lower()
        if action not in _MEMORY_ACTIONS:
            raise ValueError(f"unsupported memory_action {memory_action!r}")
        return action

    def _encode(self, texts: Sequence[str], action: str) -> list[list[float]]:
        if not texts:
            raise ValueError("cannot embed an empty batch")
        if action == "search":
            vectors = self._encoder.encode_queries(texts)
        else:
            vectors = self._encoder.encode_passages(texts)
        return [[float(value) for value in row] for row in vectors]

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("memory text must be a non-empty string")
        return self._encode([text], self._action(memory_action))[0]

    def embed_batch(
        self,
        texts: Sequence[str],
        memory_action: str | None = "add",
    ) -> list[list[float]]:
        cleaned = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in cleaned):
            raise ValueError("memory texts must be non-empty strings")
        return self._encode(cleaned, self._action(memory_action))


def _package_version(distribution: str) -> str | None:
    """Best-effort installed version without importing the package."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - not installed
        return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _first_record(raw: Any) -> Mapping[str, Any] | None:
    """Normalise Mem0's ``add`` return (a list in 2.2.1, a dict in older releases)."""
    if isinstance(raw, Mapping):
        results = raw.get("results")
        if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
            for item in results:
                if isinstance(item, Mapping):
                    return item
        return None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping):
                return item
    return None


class Mem0MemoryAdapter:
    """Mem0-backed semantic memory with strict ``user_id`` isolation.

    ``memory`` / ``embedder`` are injectable so unit tests can exercise the adapter
    logic without a live Qdrant service or the Mem0 package. Production construction
    goes through :func:`recommender.memory.adapter.create_memory_adapter`.
    """

    def __init__(
        self,
        *,
        config: MemoryConfig,
        memory: Any | None = None,
        embedder: E5MemoryEmbedder | None = None,
        history_db_path: str | Path | None = None,
        settings: Settings | None = None,
        embedding_dim: int = MEMORY_EMBEDDING_DIM,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self._config = config
        self._memory = memory
        self._embedder = embedder
        self._settings = settings
        self._embedding_dim = embedding_dim
        configured_path = (
            history_db_path if history_db_path is not None else DEFAULT_HISTORY_DB_PATH
        )
        path = Path(configured_path)
        self._history_db_path = path if path.is_absolute() else (PROJECT_ROOT / path)
        self._init_error: str | None = None

    # -- configuration -------------------------------------------------------
    @property
    def collection_name(self) -> str:
        return self._config.mem0.vector_store.collection

    @property
    def embedding_model_id(self) -> str:
        return self._config.mem0.embedder.model

    def _vector_store_description(self) -> dict[str, Any]:
        store = self._config.mem0.vector_store
        return {
            "provider": store.provider,
            "host": store.host,
            "port": store.port,
            "collection": store.collection,
            "embedding_dim": self._embedding_dim,
            "on_disk": False,
        }

    # -- construction --------------------------------------------------------
    def _load_settings(self) -> Settings:
        if self._settings is not None:
            return self._settings
        self._settings = load_settings()
        return self._settings

    def _default_embedder(self) -> E5MemoryEmbedder:
        return E5MemoryEmbedder(
            model_id=self.embedding_model_id,
            revision=MEMORY_EMBEDDING_REVISION,
            expected_dim=self._embedding_dim,
        )

    def _build_memory(self) -> Any:
        """Import Mem0 and construct ``Memory`` with the project E5 embedder injected."""
        # Telemetry must be disabled before ``mem0`` is first imported (D9 §7.6).
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        settings = self._load_settings()
        try:
            mem0_module = importlib.import_module("mem0")
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MemoryUnavailableError(
                f"mem0ai is not installed: {exc}",
                backend="mem0",
                reason=f"ImportError: {exc}",
            ) from exc
        memory_cls = getattr(mem0_module, "Memory", None)
        if memory_cls is None:  # pragma: no cover - broken installation
            raise MemoryUnavailableError(
                "installed mem0 package exposes no Memory class",
                backend="mem0",
                reason="mem0.Memory missing",
            )

        embedder = self._embedder or self._default_embedder()
        self._embedder = embedder
        api_key = settings.require_deepseek_api_key()
        store = self._config.mem0.vector_store
        # The SQLite history DB is Mem0's changelog, not the project InteractionEvent
        # store (AGENTS.md §2.2); it lives under git-ignored artifacts/.
        self._history_db_path.parent.mkdir(parents=True, exist_ok=True)
        mem0_config: dict[str, Any] = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": store.collection,
                    "embedding_model_dims": self._embedding_dim,
                    "host": store.host,
                    "port": store.port,
                    "on_disk": False,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": self.embedding_model_id,
                    "embedding_dims": self._embedding_dim,
                    # Never contacted; avoids loading a second HF model during construction.
                    "huggingface_base_url": _PLACEHOLDER_EMBEDDER_BASE_URL,
                    "api_key": "placeholder-not-used",
                },
            },
            "llm": {
                # Only used by Mem0 when infer=True, which this adapter never requests.
                # The key is still configured so Memory() fails loudly instead of
                # silently degrading if infer is ever enabled by mistake.
                "provider": "deepseek",
                "config": {
                    "model": settings.config.deepseek.generator_model,
                    "api_key": api_key,
                    "deepseek_base_url": settings.config.deepseek.base_url,
                    "temperature": 0.0,
                    "max_tokens": 1024,
                },
            },
            "history_db_path": str(self._history_db_path),
        }
        try:
            memory = memory_cls.from_config(mem0_config)
        except ConfigError:
            raise
        except Exception as exc:
            raise MemoryUnavailableError(
                f"mem0/Qdrant initialisation failed: {type(exc).__name__}: {exc}",
                backend="mem0",
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc
        # D9 §7.3 option 1: replacing embedding_model after construction takes effect for
        # every add/search/update because Mem0 calls ``self.embedding_model``.
        memory.embedding_model = embedder
        if getattr(memory, "embedding_model", None) is not embedder:  # pragma: no cover
            raise MemoryUnavailableError(
                "failed to inject the prefix-aware E5 embedder into Mem0",
                backend="mem0",
                reason="embedding_model injection did not take effect",
            )
        _LOGGER.info(
            "mem0 memory adapter initialised",
            extra={
                "collection": store.collection,
                "host": store.host,
                "port": store.port,
                "embedding_model": self.embedding_model_id,
                "embedding_dim": self._embedding_dim,
            },
        )
        return memory

    def _ensure_memory(self) -> Any:
        if self._memory is not None:
            return self._memory
        if self._init_error is not None:
            raise MemoryUnavailableError(
                f"memory backend unavailable: {self._init_error}",
                backend="mem0",
                reason=self._init_error,
            )
        try:
            self._memory = self._build_memory()
        except MemoryUnavailableError as exc:
            self._init_error = exc.reason
            raise
        except Exception as exc:
            self._init_error = f"{type(exc).__name__}: {exc}"
            raise MemoryUnavailableError(
                f"memory backend unavailable: {self._init_error}",
                backend="mem0",
                reason=self._init_error,
            ) from exc
        return self._memory

    # -- validation ----------------------------------------------------------
    @staticmethod
    def _validate_user_id(user_id: str) -> str:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        return user_id.strip()

    @staticmethod
    def _validate_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("memory text must be a non-empty string")
        return text.strip()

    @staticmethod
    def _prepare_metadata(
        metadata: Mapping[str, Any] | None,
        *,
        is_synthetic: bool,
    ) -> dict[str, Any]:
        cleaned = {
            str(key): value
            for key, value in dict(metadata or {}).items()
            if str(key) not in _RESERVED_METADATA_KEYS
        }
        cleaned["is_synthetic"] = bool(is_synthetic)
        return cleaned

    # -- public API ----------------------------------------------------------
    def read(self, user_id: str, query: str, limit: int = 5) -> list[MemoryHit]:
        """Search this user's memories only; never returns another user's payload."""
        user = self._validate_user_id(user_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        memory = self._ensure_memory()
        try:
            raw = memory.search(query.strip(), filters={"user_id": user}, top_k=limit)
        except Exception as exc:
            raise MemoryUnavailableError(
                f"mem0 search failed: {type(exc).__name__}: {exc}",
                backend="mem0",
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc
        results: Any = raw.get("results", []) if isinstance(raw, Mapping) else raw
        hits: list[MemoryHit] = []
        if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
            return hits
        for item in results:
            if not isinstance(item, Mapping):
                continue
            # Defence in depth on top of the Qdrant user_id filter (R11): if the backend
            # ever returns a foreign payload, drop it instead of handing it to the LLM.
            item_user = item.get("user_id")
            if item_user is not None and str(item_user) != user:
                _LOGGER.warning(
                    "dropping memory hit with mismatched user_id",
                    extra={"requested_user_id": user},
                )
                continue
            text = item.get("memory") or item.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            metadata: dict[str, Any] = dict(item.get("metadata") or {})
            for key in ("user_id", "agent_id", "run_id", "created_at", "updated_at"):
                value = item.get(key)
                if value is not None:
                    metadata.setdefault(key, value)
            memory_id = item.get("id")
            hits.append(
                MemoryHit(
                    memory_id=str(memory_id) if memory_id is not None else "",
                    text=text.strip(),
                    score=_float_or_none(item.get("score")),
                    metadata=metadata,
                )
            )
        return hits

    def add(
        self,
        *,
        user_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        is_synthetic: bool = False,
    ) -> MemoryHit:
        """Store one raw memory (``infer=False``); never claims a write that failed."""
        user = self._validate_user_id(user_id)
        content = self._validate_text(text)
        prepared = self._prepare_metadata(metadata, is_synthetic=is_synthetic)
        memory = self._ensure_memory()
        try:
            raw = memory.add(
                [{"role": "user", "content": content}],
                user_id=user,
                infer=False,
                metadata=prepared,
            )
        except Exception as exc:
            raise MemoryUnavailableError(
                f"mem0 add failed: {type(exc).__name__}: {exc}",
                backend="mem0",
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc
        record = _first_record(raw)
        memory_id = record.get("id") if record is not None else None
        if not memory_id:
            raise MemoryUnavailableError(
                "mem0 add returned no memory record; refusing to report a write that "
                "did not persist",
                backend="mem0",
                reason="add returned no id",
            )
        stored_text = record.get("memory") if record is not None else None
        hit_metadata = dict(prepared)
        role = record.get("role") if record is not None else None
        if isinstance(role, str) and role:
            hit_metadata["role"] = role
        return MemoryHit(
            memory_id=str(memory_id),
            text=str(stored_text) if isinstance(stored_text, str) and stored_text else content,
            score=None,
            metadata=hit_metadata,
        )

    # -- maintenance (used by integration smoke tests and cleanup jobs) ------
    def delete(self, memory_id: str) -> None:
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        memory = self._ensure_memory()
        try:
            memory.delete(memory_id.strip())
        except Exception as exc:
            raise MemoryUnavailableError(
                f"mem0 delete failed: {type(exc).__name__}: {exc}",
                backend="mem0",
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc

    def delete_all(self, user_id: str) -> None:
        user = self._validate_user_id(user_id)
        memory = self._ensure_memory()
        try:
            memory.delete_all(user_id=user)
        except Exception as exc:
            raise MemoryUnavailableError(
                f"mem0 delete_all failed: {type(exc).__name__}: {exc}",
                backend="mem0",
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc

    def health(self) -> dict[str, Any]:
        """Initialise (once) and report availability plus locked versions."""
        details: dict[str, Any] = {
            "provider": "mem0",
            "backend": "mem0",
            "collection": self.collection_name,
            "vector_store": self._vector_store_description(),
            "embedder": {
                "model": self.embedding_model_id,
                "revision": MEMORY_EMBEDDING_REVISION,
                "embedding_dim": self._embedding_dim,
                "query_prefix": "query: ",
                "passage_prefix": "passage: ",
                "memory_action_prefixes": {
                    "add": "passage: ",
                    "update": "passage: ",
                    "search": "query: ",
                },
            },
            "user_id_isolation": "filters={'user_id': ...} on every search; payload guard on read",
            "history_db_path": str(self._history_db_path),
            "versions": {
                "mem0ai": _package_version("mem0ai"),
                "qdrant-client": _package_version("qdrant-client"),
            },
        }
        if self._memory is not None:
            return {**details, "available": True, "initialized": True}
        if self._init_error is not None:
            return {
                **details,
                "available": False,
                "initialized": False,
                "reason": self._init_error,
            }
        try:
            self._ensure_memory()
        except MemoryUnavailableError as exc:
            return {
                **details,
                "available": False,
                "initialized": False,
                "reason": exc.reason,
            }
        return {**details, "available": True, "initialized": True}
