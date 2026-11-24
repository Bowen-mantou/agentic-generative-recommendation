"""Qwen3-Reranker-0.6B with the verified Transformers-native recipe (decision Q2).

Validated on 2026-11-24 (docs/audit/reranker-smoke.md):

* prefix/suffix tokens match the model card and ``tokenizer.chat_template``;
* ``yes`` = 9693, ``no`` = 2152 (single tokens, independently confirmed by the shipped
  ``1_LogitScore/config.json``);
* score = sigmoid(logit_yes - logit_no); scores saturate and are **not** probabilities.

No model is loaded at import time (AGENTS.md §4).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast, runtime_checkable

from recommender.observability.logging import get_logger
from recommender.utils.errors import ConfigError

_LOGGER = get_logger(__name__)

PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

MODEL_CARD_MAX_LENGTH = 8192


@runtime_checkable
class PairScorer(Protocol):
    """Minimal scoring interface so the rerank pipeline is testable without a model."""

    model_id: str

    @property
    def batch_size(self) -> int: ...

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]: ...


class Qwen3Reranker:
    """CPU/GPU Transformers reranker producing P(yes) relevance scores."""

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-Reranker-0.6B",
        revision: str = "main",
        device: str = "cpu",
        max_length: int = 1024,
        batch_size: int = 4,
        instruction: str = (
            "Rank products according to the user's shopping requirements and preferences."
        ),
    ) -> None:
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if device.startswith("musa"):
            raise ConfigError(
                "MUSA is not part of the first-stage baseline (decision Q11); use cpu"
            )
        if not 0 < max_length <= MODEL_CARD_MAX_LENGTH:
            raise ValueError(f"max_length must be in (0, {MODEL_CARD_MAX_LENGTH}]")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.max_length = max_length
        self._batch_size = batch_size
        self.instruction = " ".join(instruction.split())
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._yes_id: int | None = None
        self._no_id: int | None = None
        self._prefix_tokens: list[int] = []
        self._suffix_tokens: list[int] = []
        self._torch: Any | None = None

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ConfigError(
                "transformers/torch are required for the reranker; install the 'retrieval' extra"
            ) from exc
        _LOGGER.info(
            "loading Qwen3 reranker",
            extra={
                "model_id": self.model_id,
                "revision": self.revision,
                "device": self.device,
                "max_length": self.max_length,
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, padding_side="left"
        )
        loaded = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model = cast(Any, loaded).to(self.device).eval()
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")
        if not isinstance(yes_id, int) or not isinstance(no_id, int) or yes_id == no_id:
            raise ConfigError("could not resolve single-token yes/no ids for the reranker")
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._yes_id = yes_id
        self._no_id = no_id
        self._prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
        self._suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
        _LOGGER.info("reranker loaded", extra={"yes_id": yes_id, "no_id": no_id})

    def _score_batch(self, query: str, documents: Sequence[str]) -> list[float]:
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        bodies = [
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"
            for document in documents
        ]
        budget = self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        encoded = self._tokenizer(
            bodies,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max(budget, 1),
        )
        for index, ids in enumerate(encoded["input_ids"]):
            encoded["input_ids"][index] = self._prefix_tokens + ids + self._suffix_tokens
        padded = self._tokenizer.pad(
            encoded,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        padded = {key: value.to(self.device) for key, value in padded.items()}
        with self._torch.inference_mode():
            logits = self._model(**padded).logits[:, -1, :]
            stacked = self._torch.stack([logits[:, self._no_id], logits[:, self._yes_id]], dim=1)
            probabilities = self._torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()
        return [float(value) for value in probabilities.tolist()]

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return P(yes) for each query-document pair (relative ordering only)."""
        if not query.strip():
            raise ValueError("reranker query must not be empty")
        if not documents:
            return []
        self._load()
        scores: list[float] = []
        for start in range(0, len(documents), self._batch_size):
            batch = documents[start : start + self._batch_size]
            scores.extend(self._score_batch(query, batch))
        if len(scores) != len(documents):
            raise RuntimeError(
                f"reranker returned {len(scores)} scores for {len(documents)} documents"
            )
        return scores
