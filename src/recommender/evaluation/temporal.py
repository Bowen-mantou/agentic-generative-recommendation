"""Temporal evaluation-sample recovery and audit (protocol §3, §13.A).

The existing TIGER test split stores semantic-ID history; this module reconstructs the
canonical product-level history from the SQLite event log and verifies that it maps back
to the exact semantic IDs. Only verified samples are written to the evaluation dataset.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recommender.catalog.build import load_products
from recommender.interaction_history.store import InteractionStore
from recommender.observability.logging import get_logger
from recommender.retrieval.tiger.semantic_ids import SemanticIdTable
from recommender.schemas.common import EvalSplit
from recommender.schemas.evaluation_dataset import TargetLabel, TemporalSample
from recommender.schemas.product import ProductRecord
from recommender.utils.io import ensure_dir, file_sha256, utc_now_iso, write_json

_LOGGER = get_logger(__name__)


def _sample_id(user_id: str, target_timestamp: int, target_product_id: str) -> str:
    digest = hashlib.sha256(
        f"{user_id}|{target_timestamp}|{target_product_id}".encode()
    ).hexdigest()[:16]
    return f"sample_{digest}"


@dataclass
class CatalogIndex:
    """Product lookup used by profile extraction and audit mapping."""

    products: dict[str, ProductRecord]
    path: Path
    version: str = "unknown"

    @classmethod
    def load(cls, path: Path | str) -> CatalogIndex:
        source = Path(path)
        if source.is_dir():
            source = source / "products.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"catalog products file not found: {source}")
        records = load_products(source)
        version = "unknown"
        manifest_path = source.parent / "manifest.json"
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            version = str(payload.get("version") or payload.get("catalog_version") or "unknown")
        return cls({record.product_id: record for record in records}, source, version)

    def get(self, product_id: str) -> ProductRecord | None:
        return self.products.get(product_id)


@dataclass
class RecoveryStats:
    raw_samples: int = 0
    recovered: int = 0
    repeated_target: int = 0
    novel_target: int = 0
    failures: Counter[str] = field(default_factory=Counter)
    history_lengths: list[int] = field(default_factory=list)
    timestamp_min: int | None = None
    timestamp_max: int | None = None

    def as_dict(self) -> dict[str, Any]:
        lengths = self.history_lengths
        return {
            "raw_samples": self.raw_samples,
            "recovered": self.recovered,
            "recovery_rate": round(self.recovered / self.raw_samples, 4)
            if self.raw_samples
            else 0.0,
            "repeated_target": self.repeated_target,
            "novel_target": self.novel_target,
            "failures": dict(self.failures),
            "history_length": {
                "min": min(lengths) if lengths else None,
                "p50": statistics.median(lengths) if lengths else None,
                "p90": (
                    sorted(lengths)[min(len(lengths) - 1, int(0.9 * len(lengths)))]
                    if lengths
                    else None
                ),
                "max": max(lengths) if lengths else None,
                "mean": round(statistics.fmean(lengths), 2) if lengths else None,
            },
            "target_timestamp_range": [self.timestamp_min, self.timestamp_max],
        }


def load_tiger_samples(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"TIGER sample file not found: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def recover_samples(
    raw_samples: Sequence[dict[str, Any]],
    store: InteractionStore,
    table: SemanticIdTable,
    catalog: CatalogIndex,
    *,
    dataset_version: str,
) -> tuple[list[TemporalSample], list[TargetLabel], RecoveryStats]:
    """Reconstruct product-level histories and labels for the existing TIGER test split."""
    stats = RecoveryStats(raw_samples=len(raw_samples))
    samples: list[TemporalSample] = []
    labels: list[TargetLabel] = []
    seen_ids: set[str] = set()

    for raw in raw_samples:
        try:
            user_id = str(raw["user_id"])
            target_product_id = str(raw["target_product_id"])
            target_timestamp = int(raw["timestamp"])
            history_ids = [tuple(int(value) for value in item) for item in raw["history_ids"]]
        except (KeyError, TypeError, ValueError):
            stats.failures["malformed_sample"] += 1
            continue

        if not history_ids:
            stats.failures["empty_history"] += 1
            continue
        if catalog.get(target_product_id) is None:
            stats.failures["target_not_in_catalog"] += 1
            continue

        events = store.events_for_user(user_id, cutoff_ts=target_timestamp + 1)
        target_events = [
            event
            for event in events
            if event.timestamp == target_timestamp and event.product_id == target_product_id
        ]
        if not target_events:
            stats.failures["target_event_not_found"] += 1
            continue
        target_event = sorted(target_events, key=lambda event: event.event_id)[0]
        previous = [event for event in events if event.timestamp < target_timestamp]
        history_events = previous[-len(history_ids) :]
        if len(history_events) < len(history_ids):
            stats.failures["history_too_short"] += 1
            continue

        reconstructed: list[tuple[int, ...]] = []
        for event in history_events:
            semantic_id = table.semantic_id_of(event.product_id)
            if semantic_id is None:
                reconstructed = []
                break
            reconstructed.append(semantic_id)
        if not reconstructed:
            stats.failures["history_product_missing_semantic_id"] += 1
            continue
        if [list(item) for item in reconstructed] != [list(item) for item in history_ids]:
            stats.failures["history_semantic_id_mismatch"] += 1
            continue
        if any(catalog.get(event.product_id) is None for event in history_events):
            stats.failures["history_not_in_catalog"] += 1
            continue

        cutoff = target_timestamp - 1
        if any(event.timestamp > cutoff for event in history_events):
            stats.failures["history_after_cutoff"] += 1
            continue
        if target_event.timestamp <= cutoff:
            stats.failures["target_not_after_cutoff"] += 1
            continue

        sample_id = _sample_id(user_id, target_timestamp, target_product_id)
        if sample_id in seen_ids:
            stats.failures["duplicate_sample_id"] += 1
            continue
        seen_ids.add(sample_id)

        split_value = str(raw.get("split", "test"))
        try:
            split = EvalSplit(split_value)
        except ValueError:
            split = EvalSplit.TEST

        samples.append(
            TemporalSample(
                sample_id=sample_id,
                user_id=user_id,
                split=split,
                cutoff_timestamp=cutoff,
                history_product_ids=[event.product_id for event in history_events],
                history_event_ids=[event.event_id for event in history_events],
                is_synthetic=False,
                dataset_version=dataset_version,
            )
        )
        target_product = catalog.get(target_product_id)
        labels.append(
            TargetLabel(
                sample_id=sample_id,
                target_product_id=target_product_id,
                target_event_id=target_event.event_id,
                target_timestamp=target_event.timestamp,
                target_title=target_product.title if target_product is not None else "",
            )
        )
        stats.recovered += 1
        stats.history_lengths.append(len(history_events))
        stats.timestamp_min = (
            target_event.timestamp
            if stats.timestamp_min is None
            else min(stats.timestamp_min, target_event.timestamp)
        )
        stats.timestamp_max = (
            target_event.timestamp
            if stats.timestamp_max is None
            else max(stats.timestamp_max, target_event.timestamp)
        )
        if target_product_id in [event.product_id for event in history_events]:
            stats.repeated_target += 1
        else:
            stats.novel_target += 1

    _LOGGER.info("temporal recovery finished", extra=stats.as_dict())
    return samples, labels, stats


def select_pilot_samples(
    samples: Sequence[TemporalSample], limit: int, *, seed: int = 42
) -> list[TemporalSample]:
    """History-length-stratified deterministic selection for the pilot."""
    if limit >= len(samples):
        return list(samples)
    ordered = sorted(
        samples, key=lambda sample: (len(sample.history_product_ids), sample.sample_id)
    )
    # Even spacing over the sorted list spans short and long histories without relying
    # on target outcomes.
    indexes: list[int] = []
    if limit == 1:
        indexes = [len(ordered) // 2]
    else:
        for position in range(limit):
            index = round(position * (len(ordered) - 1) / (limit - 1))
            if index not in indexes:
                indexes.append(index)
    while len(indexes) < limit:
        for index in range(len(ordered)):
            if index not in indexes:
                indexes.append(index)
                break
    selected = [ordered[index] for index in sorted(indexes)]
    _LOGGER.info(
        "pilot sample selection",
        extra={"requested": limit, "selected": len(selected), "seed": seed},
    )
    return selected


def save_temporal_audit(
    out_dir: Path | str,
    samples: Sequence[TemporalSample],
    labels: Sequence[TargetLabel],
    stats: RecoveryStats,
    *,
    source_samples: Path | str,
    source_db: Path | str,
) -> dict[str, Any]:
    directory = ensure_dir(out_dir)
    samples_dir = ensure_dir(directory / "samples")
    labels_dir = ensure_dir(directory / "labels")
    reports_dir = ensure_dir(directory / "reports")

    samples_path = samples_dir / "test_samples.jsonl"
    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(sample.model_dump_json() + "\n")
    labels_path = labels_dir / "target_labels.jsonl"
    with labels_path.open("w", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            handle.write(label.model_dump_json() + "\n")

    report = {
        "created_at": utc_now_iso(),
        "source_samples": str(source_samples),
        "source_samples_sha256": file_sha256(Path(source_samples)),
        "source_db": str(source_db),
        "dataset_version": samples[0].dataset_version if samples else "unknown",
        "counts": stats.as_dict(),
        "sample_id_rule": "sha256(user_id|target_timestamp|target_product_id)[:16]",
        "time_rule": "history_event.timestamp < target_timestamp; cutoff = target_timestamp - 1",
        "target_in_history_policy": "kept; novel_item subset reported separately",
        "valid": stats.failures == Counter() or stats.recovered > 0,
    }
    write_json(reports_dir / "temporal_split_audit.json", report)
    return report


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    return [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def iter_samples(path: Path | str) -> Iterable[TemporalSample]:
    for row in load_jsonl(path):
        yield TemporalSample.model_validate(row)
