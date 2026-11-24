"""FAISS vector index helpers.

``faiss`` is imported lazily so the data-contract layer stays installable without the
retrieval extra. Vectors are L2-normalised (cosine via inner product) by the builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from recommender.utils.errors import ConfigError


@runtime_checkable
class VectorIndex(Protocol):
    """Minimal search interface implemented by :class:`FaissVectorIndex`."""

    @property
    def size(self) -> int: ...

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]: ...


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ConfigError(
            "faiss is not installed; install the 'retrieval' extra (faiss-cpu)"
        ) from exc
    return faiss


def build_index(vectors: np.ndarray, index_type: str = "IndexFlatIP") -> Any:
    """Build a FAISS index over L2-normalised float32 vectors."""
    array = np.ascontiguousarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"vectors must be a non-empty 2D array, got shape {array.shape}")
    faiss = _import_faiss()
    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(array.shape[1])
    elif index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(array.shape[1])
    else:  # pragma: no cover - config-validated
        raise ConfigError(f"unsupported index type for building: {index_type}")
    index.add(array)
    return index


def save_index(index: Any, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    faiss = _import_faiss()
    faiss.write_index(index, str(target))
    return target


def load_index(path: Path | str) -> Any:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"FAISS index not found: {target}")
    faiss = _import_faiss()
    return faiss.read_index(str(target))


class FaissVectorIndex:
    """Thin wrapper enforcing float32 / shape / top-k validation."""

    def __init__(self, index: Any) -> None:
        self._index = index

    @classmethod
    def load(cls, path: Path | str) -> FaissVectorIndex:
        return cls(load_index(path))

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    @property
    def dimension(self) -> int:
        return int(self._index.d)

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        array = np.ascontiguousarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"query vectors must be 2D, got shape {array.shape}")
        if self.size == 0:
            raise ValueError("cannot search an empty index")
        scores, ids = self._index.search(array, min(top_k, self.size))
        return np.asarray(scores, dtype=np.float32), np.asarray(ids, dtype=np.int64)
