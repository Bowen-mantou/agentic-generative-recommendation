"""Build time-aware TIGER training samples from InteractionEvent + Semantic IDs.

Split policy (no leakage): global timestamp quantiles on real events define train /
validation / test boundaries. For a sample, history only contains events strictly before
the target; targets after a boundary belong to the later split. Sequences whose history
length is below ``min_history`` are dropped.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from recommender.interaction_history.store import InteractionStore
from recommender.observability.logging import get_logger
from recommender.retrieval.tiger.semantic_ids import SemanticIdTable
from recommender.schemas.common import EvalSplit
from recommender.schemas.memory import InteractionEvent
from recommender.utils.io import ensure_dir, write_json

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TigerSample:
    user_id: str
    history_ids: tuple[tuple[int, ...], ...]
    target_id: tuple[int, ...]
    target_product_id: str
    timestamp: int
    split: EvalSplit

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "history_ids": [list(item) for item in self.history_ids],
            "target_id": list(self.target_id),
            "target_product_id": self.target_product_id,
            "timestamp": self.timestamp,
            "split": self.split.value,
        }


@dataclass
class SequenceBuildStats:
    users_considered: int = 0
    samples_total: int = 0
    skipped_missing_history_id: int = 0
    skipped_missing_target_id: int = 0
    skipped_short_history: int = 0
    train: int = 0
    validation: int = 0
    test: int = 0
    cutoffs: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "users_considered": self.users_considered,
            "samples_total": self.samples_total,
            "skipped_missing_history_id": self.skipped_missing_history_id,
            "skipped_missing_target_id": self.skipped_missing_target_id,
            "skipped_short_history": self.skipped_short_history,
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "cutoffs": self.cutoffs,
        }


def _split_for(timestamp: int, train_cutoff: int, val_cutoff: int) -> EvalSplit:
    if timestamp <= train_cutoff:
        return EvalSplit.TRAIN
    if timestamp <= val_cutoff:
        return EvalSplit.VALIDATION
    return EvalSplit.TEST


def build_sequences(
    store: InteractionStore,
    table: SemanticIdTable,
    *,
    min_events: int = 3,
    min_history: int = 2,
    max_history_items: int = 10,
    train_quantile: float = 0.8,
    val_quantile: float = 0.9,
    cutoff_ts: int | None = None,
) -> tuple[list[TigerSample], SequenceBuildStats]:
    """Iterate real user sequences and emit next-item samples with time splits."""
    stats = SequenceBuildStats()
    timestamps: list[int] = []
    users: list[tuple[str, list[InteractionEvent]]] = list(
        store.iter_user_sequences(min_events=min_events, cutoff_ts=cutoff_ts)
    )
    stats.users_considered = len(users)
    for _, events in users:
        timestamps.extend(event.timestamp for event in events)
    if not timestamps:
        return [], stats
    train_cutoff = int(np.quantile(np.asarray(timestamps), train_quantile))
    val_cutoff = int(np.quantile(np.asarray(timestamps), val_quantile))
    stats.cutoffs = {"train_ts": train_cutoff, "val_ts": val_cutoff}

    samples: list[TigerSample] = []
    for user_id, events in users:
        history: list[tuple[int, ...]] = []
        for event in events:
            semantic_id = table.semantic_id_of(event.product_id)
            if len(history) >= min_history and semantic_id is not None:
                split = _split_for(event.timestamp, train_cutoff, val_cutoff)
                samples.append(
                    TigerSample(
                        user_id=user_id,
                        history_ids=tuple(history[-max_history_items:]),
                        target_id=semantic_id,
                        target_product_id=event.product_id,
                        timestamp=event.timestamp,
                        split=split,
                    )
                )
            elif len(history) < min_history:
                stats.skipped_short_history += 1
            else:
                stats.skipped_missing_target_id += 1
            if semantic_id is None:
                stats.skipped_missing_history_id += 1
                continue
            history.append(semantic_id)

    for sample in samples:
        if sample.split is EvalSplit.TRAIN:
            stats.train += 1
        elif sample.split is EvalSplit.VALIDATION:
            stats.validation += 1
        else:
            stats.test += 1
    stats.samples_total = len(samples)
    _LOGGER.info("tiger sequences built", extra=stats.as_dict())
    return samples, stats


def save_samples(
    samples: Sequence[TigerSample],
    directory: Path | str,
    stats: SequenceBuildStats,
    *,
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = ensure_dir(directory)
    by_split: dict[str, list[TigerSample]] = {"train": [], "validation": [], "test": []}
    for sample in samples:
        by_split[sample.split.value].append(sample)
    for split, rows in by_split.items():
        path = target / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in rows:
                handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    manifest = {
        "stats": stats.as_dict(),
        "max_history_items": max((len(s.history_ids) for s in samples), default=0),
        **(extra_manifest or {}),
    }
    write_json(target / "sequence_manifest.json", manifest)
    return manifest


def load_samples(directory: Path | str, split: str) -> list[dict[str, Any]]:
    path = Path(directory) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"sample file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def iter_all_events(
    store: InteractionStore, *, cutoff_ts: int | None = None
) -> Iterable[InteractionEvent]:
    for _, events in store.iter_user_sequences(min_events=1, cutoff_ts=cutoff_ts):
        yield from events
