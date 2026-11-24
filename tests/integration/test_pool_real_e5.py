"""Integration: real E5 results -> cross-path fusion -> hard constraints -> pool.

Uses the real All_Beauty bundle on the S4000 host; skips automatically elsewhere.
TIGER is not available yet, so this test fuses the E5 path alone and validates that the
constraint partition and summaries behave on real catalog data (including unknown prices).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")

from recommender.catalog.build import load_products
from recommender.config import load_settings
from recommender.fusion import build_candidate_pool, fuse_paths
from recommender.retrieval.e5.artifacts import load_retriever
from recommender.retrieval.e5.encoder import E5Encoder
from recommender.schemas.common import ConstraintOperator
from recommender.schemas.plan import HardConstraint

pytestmark = pytest.mark.integration

BUNDLE = Path(
    os.environ.get(
        "AGR_REAL_BUNDLE", "/mnt/agr-data/artifacts/faiss/e5-base-v2_all_beauty_v2"
    )
)
CATALOG = Path(
    os.environ.get("AGR_REAL_CATALOG", "/mnt/agr-data/artifacts/catalog/all_beauty_v2")
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (BUNDLE.is_dir() and (CATALOG / "products.jsonl").is_file()),
        reason="real All_Beauty bundle/catalog not available on this host",
    ),
]


def _encoder(config) -> E5Encoder:
    e5 = config.e5
    return E5Encoder(
        model_id=e5.model_id,
        revision=e5.revision,
        device="cpu",
        batch_size=e5.batch_size,
        max_length=e5.max_length,
        query_prefix=e5.query_prefix,
        passage_prefix=e5.passage_prefix,
        normalize=e5.normalize,
        expected_dim=e5.embedding_dim,
    )


def test_real_e5_fusion_constraints_and_unknowns() -> None:
    settings = load_settings()
    config = settings.config
    e5 = config.retrieval.e5
    retriever = load_retriever(
        BUNDLE,
        _encoder(config),
        top_k_per_query=20,
        rrf_k=e5.rrf_k,
        max_queries=e5.max_queries,
    )
    output = retriever.search(
        [
            "lightweight daily moisturizer for sensitive skin",
            "fragrance free face cream under 30 dollars",
        ]
    )
    outcome = fuse_paths([output.fused], rrf_k=config.retrieval.fusion.rrf_k)
    assert outcome.candidates, "E5 returned no candidates"

    products = {
        product.product_id: product for product in load_products(CATALOG / "products.jsonl")
    }
    price_cap = HardConstraint(field="price", operator=ConstraintOperator.LTE, value=30, unit="USD")
    pool = build_candidate_pool(
        request_id="integration-1",
        outcome=outcome,
        constraints=[price_cap],
        products=products,
        rrf_k=config.retrieval.fusion.rrf_k,
    )
    summary = pool.constraint_summary
    assert summary.evaluated_products == len(outcome.candidates)
    assert summary.valid_product_count == summary.pass_count
    # All_Beauty price coverage is ~15.7%; unknown candidates must exist, not be "passed".
    assert summary.unknown_count > 0
    assert all(
        candidate.constraint_status is None or candidate.constraint_status.value == "pass"
        for candidate in pool.candidates
    )

    pool_without_constraints = build_candidate_pool(
        request_id="integration-2",
        outcome=outcome,
        constraints=[],
        products=products,
        rrf_k=config.retrieval.fusion.rrf_k,
    )
    assert pool_without_constraints.constraint_summary.not_configured_count == len(
        outcome.candidates
    )
    assert pool_without_constraints.constraint_summary.valid_product_count == len(
        outcome.candidates
    )
    assert all(
        candidate.constraint_status is None for candidate in pool_without_constraints.candidates
    )
