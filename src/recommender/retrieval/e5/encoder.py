"""E5 encoder wrapper with lazy model loading.

Importing this module never downloads or instantiates a model (AGENTS.md §4). The
encoder adds the model-specific ``query: `` / ``passage: `` prefixes, truncates to the
configured max length and L2-normalises when required.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from recommender.observability.logging import get_logger
from recommender.utils.errors import ConfigError

_LOGGER = get_logger(__name__)


@runtime_checkable
class QueryEncoder(Protocol):
    """Minimal encoder interface used by :class:`E5Retriever` (easy to stub in tests)."""

    @property
    def embedding_dim(self) -> int: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class E5Encoder:
    """Sentence-Transformers E5 encoder."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str = "main",
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 512,
        query_prefix: str = "",
        passage_prefix: str = "",
        normalize: bool = True,
        expected_dim: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.requested_device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.normalize = normalize
        self.expected_dim = expected_dim
        self._model: Any | None = None
        self._embedding_dim: int | None = expected_dim

    # -- device ---------------------------------------------------------------
    def _resolve_device(self) -> str:
        if self.requested_device != "auto":
            if self.requested_device.startswith("musa"):
                raise ConfigError(
                    "MUSA device is not verified for sentence-transformers; run E5 on CPU "
                    "(or CUDA on a supported host) and record the device in the manifest."
                )
            return self.requested_device
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # -- model ----------------------------------------------------------------
    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ConfigError(
                "sentence-transformers is not installed; install the 'retrieval' extra"
            ) from exc
        device = self._resolve_device()
        _LOGGER.info(
            "loading E5 encoder",
            extra={"model_id": self.model_id, "revision": self.revision, "device": device},
        )
        model = SentenceTransformer(self.model_id, revision=self.revision, device=device)
        model.max_seq_length = self.max_length
        # sentence-transformers >= 5 renamed the accessor; support both.
        dimension_getter = (
            getattr(model, "get_embedding_dimension", None)
            or model.get_sentence_embedding_dimension
        )
        dim = int(dimension_getter())
        if self.expected_dim is not None and dim != self.expected_dim:
            raise ConfigError(
                f"encoder dimension {dim} does not match configured embedding_dim "
                f"{self.expected_dim} for {self.model_id}"
            )
        self._embedding_dim = dim
        self._model = model
        self._device = device
        return model

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._load()
        assert self._embedding_dim is not None
        return self._embedding_dim

    @property
    def device(self) -> str:
        self._load()
        return getattr(self, "_device", self.requested_device)

    # -- encoding -------------------------------------------------------------
    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if not texts:
            raise ValueError("cannot encode an empty batch")
        model = self._load()
        prepared = [f"{prefix}{text}" if prefix else text for text in texts]
        vectors = model.encode(
            prepared,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"encoder returned unexpected shape {array.shape}")
        if self._embedding_dim is not None and array.shape[1] != self._embedding_dim:
            raise ValueError(
                f"encoder returned dim {array.shape[1]}, expected {self._embedding_dim}"
            )
        self._embedding_dim = int(array.shape[1])
        return array

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix)
