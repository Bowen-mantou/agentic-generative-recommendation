"""TIGER sequence construction and the small autoregressive generator (tiny data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recommender.interaction_history import InteractionStore
from recommender.retrieval.tiger.semantic_ids import SemanticIdTable
from recommender.schemas.common import EventType
from recommender.schemas.memory import InteractionEvent

pytestmark = pytest.mark.unit


def _event(event_id: str, user: str, product: str, ts: int) -> InteractionEvent:
    return InteractionEvent(
        event_id=event_id,
        user_id=user,
        product_id=product,
        event_type=EventType.REVIEW,
        rating=5,
        timestamp=ts,
        source="amazon_reviews_2023",
        is_synthetic=False,
    )


def test_build_sequences_respects_time_and_history(tmp_path: Path) -> None:
    from recommender.retrieval.tiger.sequences import build_sequences

    db = tmp_path / "events.db"
    with InteractionStore(db) as store:
        events = []
        for user_index, user in enumerate(("u1", "u2", "u3")):
            for position in range(4):
                product = f"p{user_index}-{position}"
                events.append(
                    _event(
                        f"e{user_index}-{position}",
                        user,
                        product,
                        1_000_000 + user_index * 1000 + position * 100,
                    )
                )
        store.insert_events(events)
        table = SemanticIdTable.build(
            list(dict.fromkeys(event.product_id for event in events)),
            np.asarray([(index % 4, index % 3) for index in range(len(events))], dtype=np.int32),
        )
        samples, stats = build_sequences(store, table, min_events=4, min_history=2)

    assert stats.users_considered == 3
    assert stats.samples_total == len(samples)
    assert samples
    for sample in samples:
        assert len(sample.history_ids) >= 2
        assert sample.target_id is not None
    # global time split must be monotone: every train target earlier than every test target
    train_ts = [s.timestamp for s in samples if s.split.value == "train"]
    test_ts = [s.timestamp for s in samples if s.split.value == "test"]
    if train_ts and test_ts:
        assert max(train_ts) <= min(test_ts)


def _tiny_generator_config():
    from recommender.retrieval.tiger.generator import TigerGeneratorConfig

    return TigerGeneratorConfig(
        name="tiny",
        num_levels=2,
        codebook_size=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=2,
        feedforward_dim=32,
        dropout=0.0,
        max_seq_len=16,
        epochs=2,
        batch_size=4,
        beam_size=4,
        log_every_epochs=1,
    )


def test_generator_train_generate_and_evaluate(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from recommender.retrieval.tiger.generator import (
        evaluate_samples,
        generate_ids,
        rank_products_for_ids,
        train_generator,
    )

    rows = [
        {"history_ids": [[0, 1]], "target_id": [2, 3], "target_product_id": "p2"},
        {"history_ids": [[1, 2]], "target_id": [4, 5], "target_product_id": "p4"},
        {"history_ids": [[2, 3]], "target_id": [6, 7], "target_product_id": "p6"},
    ] * 4
    config = _tiny_generator_config()
    model, metrics = train_generator(rows[:8], rows[8:], config, tmp_path)
    assert len(metrics.train_loss) == 2
    assert all(value >= 0 for value in metrics.train_loss)

    decoded = generate_ids(model, config, [[0, 1]])
    assert decoded
    assert all(
        isinstance(semantic_id, tuple) and len(semantic_id) == 2 for semantic_id, _ in decoded
    )

    table = SemanticIdTable.build(
        ["p1", "p2", "p3", "p4"],
        np.asarray([[2, 3], [4, 5], [2, 3], [6, 7]], dtype=np.int32),
    )
    ranked = rank_products_for_ids(
        [((2, 3), -0.1), ((4, 5), -0.2)], table, {"p2": 10, "p1": 1}, top_k=3
    )
    assert ranked
    assert ranked[0][0] in {"p1", "p2"}
    report = evaluate_samples(model, config, rows[8:], table, {"p2": 10, "p4": 5}, k_values=(1, 2))
    assert report["samples"] == 4
    assert 0.0 <= report["mrr"] <= 1.0
    assert report["k_values"] == [1, 2]


def test_tiger_generator_config_file() -> None:
    from recommender.retrieval.tiger.generator import load_generator_config

    config = load_generator_config("configs/tiger/generator_pilot.yaml")
    assert config.num_levels == 3
    assert config.codebook_size == 256
    assert config.beam_size == 10
