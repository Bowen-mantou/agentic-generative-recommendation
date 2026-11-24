"""Build ``InteractionEvent`` rows from Amazon Reviews 2023 review files.

Amazon reviews are review/rating *proxy* events: they are never described as clicks or
purchases, and every row is marked ``is_synthetic=false`` (AGENTS.md §2.2, Q5). Reviews
for products outside the catalog are filtered out so TIGER cannot generate IDs that are
not resolvable.

Timestamps in the raw dataset are Unix milliseconds; they are normalised to seconds.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recommender.interaction_history.store import InteractionStore
from recommender.observability.logging import get_logger
from recommender.schemas.common import EventType
from recommender.schemas.memory import InteractionEvent
from recommender.utils.io import ensure_dir, stable_hash

_LOGGER = get_logger(__name__)

AMAZON_SOURCE = "amazon_reviews_2023"
MAX_TIMESTAMP_SECONDS = 4_102_444_800  # 2100-01-01
_INSERT_BATCH = 10_000


@dataclass
class ReviewBuildStats:
    records_read: int = 0
    events_written: int = 0
    duplicate_events: int = 0
    skipped_malformed_json: int = 0
    skipped_missing_user: int = 0
    skipped_missing_product: int = 0
    skipped_bad_timestamp: int = 0
    skipped_bad_rating: int = 0
    filtered_not_in_catalog: int = 0
    event_types: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "records_read": self.records_read,
            "events_written": self.events_written,
            "duplicate_events": self.duplicate_events,
            "skipped_malformed_json": self.skipped_malformed_json,
            "skipped_missing_user": self.skipped_missing_user,
            "skipped_missing_product": self.skipped_missing_product,
            "skipped_bad_timestamp": self.skipped_bad_timestamp,
            "skipped_bad_rating": self.skipped_bad_rating,
            "filtered_not_in_catalog": self.filtered_not_in_catalog,
            "event_types": self.event_types,
        }


def normalize_timestamp(raw: Any) -> int | None:
    """Normalise Amazon millisecond timestamps to seconds (returns ``None`` if invalid)."""
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value > 10**11:  # milliseconds (e.g. 1_700_000_000_000)
        value //= 1000
    if value > MAX_TIMESTAMP_SECONDS:
        return None
    return value


def normalize_rating(raw: Any) -> int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != int(value):
        return None
    rating = int(value)
    return rating if 1 <= rating <= 5 else None


def review_event_type(record: dict[str, Any]) -> EventType:
    """A review with text/title is a ``review``; rating-only rows are ``rating``."""
    text = record.get("text") or record.get("review_text") or record.get("title")
    if isinstance(text, str) and text.strip():
        return EventType.REVIEW
    return EventType.RATING


def parse_review_record(record: dict[str, Any]) -> InteractionEvent | None:
    """Map one raw review record to an event; ``None`` when required fields are invalid."""
    user_id = _as_id(record.get("user_id"))
    product_id = _as_id(record.get("parent_asin"))
    if user_id is None or product_id is None:
        return None
    timestamp = normalize_timestamp(record.get("timestamp"))
    if timestamp is None:
        return None
    rating = normalize_rating(record.get("rating"))
    asin = _as_id(record.get("asin")) or ""
    # Amazon review dumps have no stable review_id; derive a deterministic one.
    event_id = "evt_" + stable_hash(f"{user_id}|{product_id}|{timestamp}|{asin}", length=24)
    event_type = review_event_type(record)
    if event_type is EventType.RATING and rating is None:
        return None
    return InteractionEvent(
        event_id=event_id,
        user_id=user_id,
        product_id=product_id,
        event_type=event_type,
        rating=rating,
        timestamp=timestamp,
        source=AMAZON_SOURCE,
        source_id=None,
        is_synthetic=False,
    )


def _classify(record: dict[str, Any]) -> tuple[InteractionEvent | None, str | None]:
    """Return ``(event, skip_reason)`` without raising for ordinary bad rows."""
    if _as_id(record.get("user_id")) is None:
        return None, "skipped_missing_user"
    if _as_id(record.get("parent_asin")) is None:
        return None, "skipped_missing_product"
    if normalize_timestamp(record.get("timestamp")) is None:
        return None, "skipped_bad_timestamp"
    if (
        review_event_type(record) is EventType.RATING
        and normalize_rating(record.get("rating")) is None
    ):
        return None, "skipped_bad_rating"
    event = parse_review_record(record)
    if event is None:  # defensive: parse should agree with the checks above
        return None, "skipped_bad_rating"
    return event, None


def iter_review_records(path: Path | str) -> Iterator[tuple[int, dict[str, Any] | None]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                yield line_number, None
                continue
            yield line_number, record if isinstance(record, dict) else None


def load_catalog_product_ids(products_path: Path | str) -> set[str]:
    ids: set[str] = set()
    with Path(products_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            ids.add(str(json.loads(stripped)["product_id"]))
    return ids


def build_interaction_history(
    reviews_path: Path | str,
    db_path: Path | str,
    *,
    catalog_products_path: Path | str | None = None,
    limit: int | None = None,
    log_every: int = 100_000,
) -> ReviewBuildStats:
    """Stream a review file into the SQLite event store (idempotent)."""
    source = Path(reviews_path)
    if not source.is_file():
        raise FileNotFoundError(f"review file not found: {source}")
    allowed_products = (
        load_catalog_product_ids(catalog_products_path)
        if catalog_products_path is not None
        else None
    )
    ensure_dir(Path(db_path).parent)
    stats = ReviewBuildStats()
    batch: list[InteractionEvent] = []
    with InteractionStore(db_path) as store:
        for line_number, record in iter_review_records(source):
            if limit is not None and stats.records_read >= limit:
                break
            stats.records_read += 1
            if record is None:
                stats.skipped_malformed_json += 1
                _LOGGER.warning("malformed review line skipped", extra={"line": line_number})
                continue
            event, reason = _classify(record)
            if event is None:
                attribute = reason or "skipped_bad_rating"
                setattr(stats, attribute, getattr(stats, attribute) + 1)
                continue
            if allowed_products is not None and event.product_id not in allowed_products:
                stats.filtered_not_in_catalog += 1
                continue
            batch.append(event)
            if len(batch) >= _INSERT_BATCH:
                inserted = store.insert_events(batch)
                stats.events_written += inserted.inserted
                stats.duplicate_events += inserted.duplicates
                batch = []
            if stats.records_read % log_every == 0:
                _LOGGER.info(
                    "review ingestion progress",
                    extra={
                        "read": stats.records_read,
                        "written": stats.events_written,
                        "filtered": stats.filtered_not_in_catalog,
                    },
                )
        if batch:
            inserted = store.insert_events(batch)
            stats.events_written += inserted.inserted
            stats.duplicate_events += inserted.duplicates

        for event_type, count in store.stats()["event_types"].items():
            stats.event_types[str(event_type)] = int(count)
    _LOGGER.info("interaction history built", extra=stats.as_dict())
    return stats


def sequence_length_summary(lengths: Iterable[int]) -> dict[str, Any]:
    """Summarise user sequence lengths (percentiles) for TIGER feasibility (R3)."""
    materialized = sorted(lengths)
    if not materialized:
        return {"users": 0, "events": 0}
    total = len(materialized)

    def percentile(fraction: float) -> int:
        index = min(total - 1, max(0, round(fraction * (total - 1))))
        return materialized[index]

    return {
        "users": total,
        "events": sum(materialized),
        "mean": round(sum(materialized) / total, 2),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": materialized[-1],
        "users_ge_3": sum(1 for length in materialized if length >= 3),
        "users_ge_5": sum(1 for length in materialized if length >= 5),
        "users_ge_10": sum(1 for length in materialized if length >= 10),
    }


def _as_id(value: Any) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None
