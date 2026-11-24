"""Interaction History: review parsing, SQLite store and sequence statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recommender.interaction_history import (
    InteractionStore,
    build_interaction_history,
    normalize_rating,
    normalize_timestamp,
    parse_review_record,
    sequence_length_summary,
)
from recommender.schemas.common import EventType

pytestmark = pytest.mark.unit


def test_normalize_timestamp_handles_milliseconds_and_seconds() -> None:
    assert normalize_timestamp(1_700_000_000_000) == 1_700_000_000
    assert normalize_timestamp(1_700_000_000) == 1_700_000_000
    assert normalize_timestamp("1700000000000") == 1_700_000_000
    assert normalize_timestamp(None) is None
    assert normalize_timestamp(True) is None
    assert normalize_timestamp(0) is None
    assert normalize_timestamp(-5) is None
    assert normalize_timestamp(10**15) is None


def test_normalize_rating() -> None:
    assert normalize_rating(5) == 5
    assert normalize_rating(3.0) == 3
    assert normalize_rating(0) is None
    assert normalize_rating(6) is None
    assert normalize_rating(3.5) is None
    assert normalize_rating(None) is None


def test_review_event_type_depends_on_text() -> None:
    assert (
        parse_review_record(
            {
                "user_id": "u1",
                "parent_asin": "p1",
                "timestamp": 1_700_000_000,
                "rating": 5,
                "text": "great",
            }
        ).event_type
        is EventType.REVIEW
    )
    assert (
        parse_review_record(
            {
                "user_id": "u1",
                "parent_asin": "p1",
                "timestamp": 1_700_000_000,
                "rating": 5,
                "text": "",
            }
        ).event_type
        is EventType.RATING
    )


def test_parse_review_record_is_deterministic_and_real() -> None:
    record = {
        "user_id": "u1",
        "parent_asin": "p1",
        "asin": "p1",
        "timestamp": 1_700_000_000_000,
        "rating": 4,
        "text": "nice",
    }
    first = parse_review_record(record)
    second = parse_review_record(record)
    assert first is not None and second is not None
    assert first.event_id == second.event_id
    assert first.timestamp == 1_700_000_000
    assert first.is_synthetic is False
    assert first.source == "amazon_reviews_2023"
    assert first.rating == 4


def test_parse_review_record_rejects_missing_ids() -> None:
    assert parse_review_record({"user_id": "u1", "timestamp": 1}) is None
    assert parse_review_record({"parent_asin": "p1", "timestamp": 1}) is None


def _write_reviews(path: Path) -> None:
    rows = [
        # valid review (in catalog)
        {
            "user_id": "u1",
            "parent_asin": "p1",
            "asin": "p1",
            "timestamp": 1_700_000_000_000,
            "rating": 5,
            "text": "great",
        },
        # valid rating-only (in catalog)
        {
            "user_id": "u1",
            "parent_asin": "p2",
            "asin": "p2",
            "timestamp": 1_700_000_100_000,
            "rating": 4,
            "text": "",
        },
        # second user
        {
            "user_id": "u2",
            "parent_asin": "p1",
            "asin": "p1",
            "timestamp": 1_700_000_200_000,
            "rating": 3,
            "text": "ok",
        },
        # product outside catalog -> filtered
        {
            "user_id": "u3",
            "parent_asin": "pX",
            "asin": "pX",
            "timestamp": 1_700_000_300_000,
            "rating": 5,
            "text": "x",
        },
        # missing user -> skipped
        {"parent_asin": "p1", "timestamp": 1_700_000_400_000, "rating": 5},
        # bad timestamp -> skipped
        {"user_id": "u4", "parent_asin": "p1", "timestamp": "bad", "rating": 5},
        # rating-only with invalid rating -> skipped
        {
            "user_id": "u5",
            "parent_asin": "p1",
            "timestamp": 1_700_000_500_000,
            "rating": 9,
            "text": "",
        },
        # duplicate of the first record -> duplicate
        {
            "user_id": "u1",
            "parent_asin": "p1",
            "asin": "p1",
            "timestamp": 1_700_000_000_000,
            "rating": 5,
            "text": "great",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_interaction_history_end_to_end(tmp_path: Path) -> None:
    reviews = tmp_path / "reviews.jsonl"
    _write_reviews(reviews)
    catalog = tmp_path / "products.jsonl"
    catalog.write_text(
        "\n".join(json.dumps({"product_id": pid, "parent_asin": pid}) for pid in ("p1", "p2"))
        + "\n",
        encoding="utf-8",
    )
    db = tmp_path / "events.db"
    stats = build_interaction_history(reviews, db, catalog_products_path=catalog)
    assert stats.records_read == 8
    assert stats.events_written == 3
    assert stats.duplicate_events == 1
    assert stats.filtered_not_in_catalog == 1
    assert stats.skipped_missing_user == 1
    assert stats.skipped_bad_timestamp == 1
    assert stats.skipped_bad_rating == 1

    with InteractionStore(db) as store:
        assert store.count() == 3
        assert store.count(user_id="u1") == 2
        events = store.events_for_user("u1")
        assert [event.product_id for event in events] == ["p1", "p2"]
        sequences = dict(store.iter_user_sequences(min_events=1))
        assert set(sequences) == {"u1", "u2"}
        assert len(sequences["u1"]) == 2
        assert store.stats()["event_types"] == {"review": 2, "rating": 1}


def test_insert_is_idempotent(tmp_path: Path) -> None:
    reviews = tmp_path / "reviews.jsonl"
    _write_reviews(reviews)
    db = tmp_path / "events.db"
    build_interaction_history(reviews, db)
    second = build_interaction_history(reviews, db)
    assert second.events_written == 0
    assert second.duplicate_events == 5  # 4 new + 1 in-file duplicate, all already stored


def test_sequence_length_summary() -> None:
    summary = sequence_length_summary([1, 2, 2, 3, 10])
    assert summary["users"] == 5
    assert summary["events"] == 18
    assert summary["p50"] == 2
    assert summary["max"] == 10
    assert summary["users_ge_3"] == 2
    assert sequence_length_summary([]) == {"users": 0, "events": 0}


def test_build_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_interaction_history(tmp_path / "missing.jsonl", tmp_path / "events.db")


def test_store_cutoff_filters_future_events(tmp_path: Path) -> None:
    reviews = tmp_path / "reviews.jsonl"
    _write_reviews(reviews)
    db = tmp_path / "events.db"
    build_interaction_history(reviews, db)
    with InteractionStore(db) as store:
        assert store.count(cutoff_ts=1_700_000_050) == 1
        sequences = list(store.iter_user_sequences(min_events=2, cutoff_ts=1_700_000_050))
        assert sequences == []
        assert store.insert_events([]).inserted == 0
