"""Unit tests for the E2E CLI/fallback surface (no models or network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from recommender.cli import build_parser
from recommender.e2e.service import E2EOptions, _FallbackGenerator, _hash_user
from recommender.schemas.common import StopReason

pytestmark = pytest.mark.unit


def test_recommend_cli_parses_required_and_optional_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "recommend",
            "--query",
            "fragrance-free moisturizer",
            "--user-id",
            "u1",
            "--top-k",
            "3",
            "--judgment-top-n",
            "5",
            "--rerank-top-n",
            "10",
            "--no-memory",
        ]
    )
    assert args.command == "recommend"
    assert args.query == "fragrance-free moisturizer"
    assert args.user_id == "u1"
    assert args.top_k == 3
    assert args.memory is False


def test_feedback_cli_parses_choices() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "feedback",
            "--user-id",
            "u1",
            "--product-id",
            "B0001",
            "--feedback",
            "positive",
            "--simulated",
        ]
    )
    assert args.command == "feedback"
    assert args.feedback == "positive"
    assert args.simulated is True


def test_e2e_options_are_explicit(tmp_path: Path) -> None:
    options = E2EOptions(
        catalog_dir=tmp_path / "catalog",
        bundle_dir=tmp_path / "bundle",
        trace_dir=tmp_path / "traces",
        top_k=4,
        memory_enabled=False,
    )
    assert options.top_k == 4
    assert options.memory_enabled is False
    assert options.judgment_top_n == 10


def test_fallback_generator_never_adds_products() -> None:
    generator = _FallbackGenerator()
    output = generator.generate(
        request_id="req_test",
        candidates=[
            {"product_id": "p1", "title": "Product 1", "price": 9.99},
            {"product_id": "p2", "title": "Product 2", "price": None},
        ],
        stop_reason=StopReason.QUALITY_SATISFIED,
        quality_satisfied=True,
        budget_exhausted=False,
    )
    assert output.product_ids == ["p1", "p2"]
    assert all(item.reasons for item in output.recommendations)
    assert output.model == "fallback_template"


def test_user_hash_is_stable_and_not_raw() -> None:
    first = _hash_user("demo_user")
    assert first == _hash_user("demo_user")
    assert first != "demo_user"
    assert len(first) == 12
