"""Unit tests for temporal datasets, profiles, query validation and LLM contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from recommender.config import EvaluationProfileConfig, EvaluationQueryConfig, load_config
from recommender.evaluation.profile import ProfileExtractor, profile_to_input
from recommender.evaluation.query_auditor import QueryAuditor
from recommender.evaluation.query_generator import QueryGenerator
from recommender.evaluation.temporal import (
    CatalogIndex,
    recover_samples,
    select_pilot_samples,
)
from recommender.evaluation.validation import validate_query
from recommender.interaction_history import InteractionStore
from recommender.retrieval.tiger.semantic_ids import SemanticIdTable
from recommender.schemas.common import EventType
from recommender.schemas.evaluation_dataset import (
    ProfileFact,
    SyntheticQuery,
    TargetLabel,
    TemporalSample,
    UserProfile,
)
from recommender.schemas.memory import InteractionEvent
from recommender.schemas.product import ProductRecord

pytestmark = pytest.mark.unit


def _event(event_id: str, user: str, product: str, ts: int, rating: int | None) -> InteractionEvent:
    return InteractionEvent(
        event_id=event_id,
        user_id=user,
        product_id=product,
        event_type=EventType.REVIEW,
        rating=rating,
        timestamp=ts,
        source="amazon_reviews_2023",
        is_synthetic=False,
    )


def _catalog(tmp_path: Path) -> CatalogIndex:
    records = [
        ProductRecord(
            product_id="p1",
            parent_asin="p1",
            title="Fragrance Free Daily Face Moisturizer",
            brand="BrandA",
            main_category="All Beauty",
            details={
                "Scent": "Fragrance-free",
                "Skin Type": "Sensitive",
                "Package Dimensions": "x",
            },
            price=None,
        ),
        ProductRecord(
            product_id="p2",
            parent_asin="p2",
            title="Gentle Foaming Cleanser",
            brand="BrandB",
            main_category="All Beauty",
            details={"Scent": "Unscented", "Skin Type": "Sensitive"},
            price=12.0,
        ),
        ProductRecord(
            product_id="p3",
            parent_asin="p3",
            title="Strong Perfume Spray",
            brand="BrandC",
            main_category="All Beauty",
            details={"Scent": "Strong floral"},
            price=30.0,
        ),
    ]
    path = tmp_path / "catalog"
    path.mkdir()
    (path / "products.jsonl").write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n", encoding="utf-8"
    )
    return CatalogIndex.load(path)


def _table(tmp_path: Path, product_ids: list[str]) -> SemanticIdTable:
    import numpy as np

    codes = np.asarray([[index, index + 1] for index in range(len(product_ids))], dtype=np.int32)
    table = SemanticIdTable.build(product_ids, codes)
    table.save(tmp_path / "sid")
    return SemanticIdTable.load(tmp_path / "sid")


def test_recover_temporal_samples_and_allow_repeated_target(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    table = _table(tmp_path, ["p1", "p2", "p3"])
    store = InteractionStore(tmp_path / "events.db")
    store.insert_events(
        [
            _event("e1", "u1", "p1", 10, 5),
            _event("e2", "u1", "p2", 20, 4),
            _event("e3", "u1", "p3", 100, 5),
        ]
    )
    raw = [
        {
            "user_id": "u1",
            "timestamp": 100,
            "target_product_id": "p3",
            "history_ids": [list(table.semantic_id_of("p1")), list(table.semantic_id_of("p2"))],
            "split": "test",
        }
    ]
    samples, labels, stats = recover_samples(raw, store, table, catalog, dataset_version="v-test")
    store.close()
    assert stats.recovered == 1
    assert samples[0].history_product_ids == ["p1", "p2"]
    assert samples[0].cutoff_timestamp == 99
    assert labels[0].target_product_id == "p3"
    assert stats.novel_target == 1


def test_recover_rejects_history_semantic_mismatch(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    table = _table(tmp_path, ["p1", "p2", "p3"])
    store = InteractionStore(tmp_path / "events.db")
    store.insert_events([_event("e1", "u1", "p1", 10, 5), _event("e2", "u1", "p2", 20, 4)])
    raw = [
        {
            "user_id": "u1",
            "timestamp": 20,
            "target_product_id": "p2",
            "history_ids": [[99, 99]],
            "split": "test",
        }
    ]
    samples, labels, stats = recover_samples(raw, store, table, catalog, dataset_version="v-test")
    store.close()
    assert samples == [] and labels == []
    assert stats.failures["history_semantic_id_mismatch"] == 1


def test_select_pilot_is_deterministic_and_spans_histories() -> None:
    samples = [
        TemporalSample(
            sample_id=f"sample_{index}",
            user_id=f"u{index}",
            split="test",
            cutoff_timestamp=100 + index,
            history_product_ids=[f"p{j}" for j in range(index + 2)],
            history_event_ids=[f"e{j}" for j in range(index + 2)],
            dataset_version="v-test",
        )
        for index in range(20)
    ]
    first = select_pilot_samples(samples, 5, seed=42)
    second = select_pilot_samples(samples, 5, seed=42)
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert len(first) == 5
    assert len({len(sample.history_product_ids) for sample in first}) > 1


def test_profile_facts_are_deterministic_and_sourced(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    store = InteractionStore(tmp_path / "events.db")
    store.insert_events(
        [
            _event("e1", "u1", "p1", 10, 5),
            _event("e2", "u1", "p2", 20, 4),
            _event("e3", "u1", "p3", 30, 1),
        ]
    )
    sample = TemporalSample(
        sample_id="sample_u1",
        user_id="u1",
        split="test",
        cutoff_timestamp=30,
        history_product_ids=["p1", "p2", "p3"],
        history_event_ids=["e1", "e2", "e3"],
        dataset_version="v-test",
    )
    profile = ProfileExtractor(catalog.products, EvaluationProfileConfig()).build(sample, store)
    store.close()
    assert profile.profile_id.startswith("profile_")
    assert profile.category_interests
    assert all(fact.source_event_ids for fact in profile.category_interests)
    assert any("fragrance" in fact.value for fact in profile.attribute_interests)
    assert profile.negative_signals
    assert profile.unknown_preferences
    payload = profile_to_input(profile)
    assert "target" not in json.dumps(payload).lower()


def _profile() -> UserProfile:
    fact = ProfileFact(
        fact_id="fact_1",
        kind="category",
        value="moisturizer",
        support_count=3,
        positive_count=3,
        source_event_ids=["e1"],
        confidence="high",
    )
    return UserProfile(
        profile_id="profile_1",
        sample_id="sample_1",
        user_id="u1",
        cutoff_timestamp=99,
        profile_version="v1",
        history_length=3,
        category_interests=[fact],
    )


def test_validation_blocks_target_id_and_title_leakage() -> None:
    profile = _profile()
    label = TargetLabel(
        sample_id="sample_1",
        target_product_id="B0003",
        target_event_id="e9",
        target_timestamp=100,
        target_title="Fragrance Free Daily Face Moisturizer",
    )
    leaked = SyntheticQuery(
        query_id="query_1",
        sample_id="sample_1",
        query="I want B0003 Fragrance Free Daily Face Moisturizer",
        used_fact_ids=["fact_1"],
        status="generated",
        generator_model="deepseek-flash",
        prompt_version="query_generator_v1",
    )
    result = validate_query(leaked, profile, label)
    assert not result.passed
    assert "target_product_id_in_query" in result.errors
    assert "target_title_in_query" in result.errors


def test_validation_rejects_unknown_fact_ids() -> None:
    profile = _profile()
    label = TargetLabel(
        sample_id="sample_1",
        target_product_id="p9",
        target_event_id="e9",
        target_timestamp=100,
        target_title="Unrelated product",
    )
    query = SyntheticQuery(
        query_id="query_1",
        sample_id="sample_1",
        query="gentle daily moisturizer",
        used_fact_ids=["fact_unknown"],
        status="generated",
        generator_model="deepseek-flash",
        prompt_version="query_generator_v1",
    )
    result = validate_query(query, profile, label)
    assert not result.passed
    assert any(error.startswith("unknown_fact_ids") for error in result.errors)


class _StubClient:
    model = "deepseek-flash"

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.calls: list[dict] = []

    def chat_json(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        payload = self.payloads.pop(0)
        completion = SimpleNamespace(
            request_id="req_stub",
            api_request_id=None,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            finish_reason="stop",
        )
        return SimpleNamespace(data=payload, completion=completion)


def test_generator_prompt_has_no_target_and_uses_facts() -> None:
    stub = _StubClient(
        [
            {
                "query": "gentle moisturizer for sensitive skin",
                "used_fact_ids": ["fact_1"],
                "status": "generated",
            }
        ]
    )
    generator = QueryGenerator(stub, EvaluationQueryConfig())
    query = generator.generate(_profile())
    assert query.status == "generated"
    assert query.used_fact_ids == ["fact_1"]
    assert "target_product" not in json.dumps(stub.calls[0]["messages"]).lower()


def test_auditor_is_independent_and_normalizes_fail() -> None:
    stub = _StubClient(
        [
            {
                "groundedness": "fail",
                "hallucination": True,
                "contradiction": False,
                "query_quality": "bad",
                "unsupported_claims": ["anti-aging"],
                "supported_fact_ids": [],
                "audit_status": "REVIEW",
                "reason": "invented benefit",
            }
        ]
    )
    auditor = QueryAuditor(stub, EvaluationQueryConfig())
    query = SyntheticQuery(
        query_id="query_1",
        sample_id="sample_1",
        query="anti-aging moisturizer",
        used_fact_ids=["fact_1"],
        status="generated",
        generator_model="deepseek-flash",
        prompt_version="query_generator_v1",
    )
    audit = auditor.audit(_profile(), query)
    assert audit.audit_status == "FAIL"
    assert audit.hallucination is True
    assert "candidate_query" in stub.calls[0]["messages"][1]["content"]


def test_config_exposes_evaluation_section() -> None:
    config = load_config()
    assert config.evaluation.profile.version == "v1"
    assert config.evaluation.query.pilot_size == 100
