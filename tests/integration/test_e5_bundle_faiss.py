"""Integration test: build a real FAISS bundle from a tiny synthetic catalog.

Runs only where the retrieval extra is installed (the S4000 venv). It exercises the
same code path as the production index build (texts -> vectors -> FAISS -> manifest ->
load -> search) without downloading a model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")

from recommender.retrieval.e5.artifacts import build_bundle, load_retriever, verify_bundle
from recommender.utils.io import utc_now_iso, write_json

pytestmark = pytest.mark.integration

DIM = 16


class _DeterministicEncoder:
    """Deterministic stand-in for E5: identical text -> identical vector."""

    model_id = "test/deterministic-encoder"
    revision = "test"
    batch_size = 2
    # Empty prefixes so identical text yields an identical vector (self-retrieval check).
    query_prefix = ""
    passage_prefix = ""
    max_length = 64

    @property
    def embedding_dim(self) -> int:
        return DIM

    def _vector(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=DIM).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])


def _write_catalog(root: Path) -> Path:
    catalog_dir = root / "catalog" / "tiny_v1"
    catalog_dir.mkdir(parents=True)
    products = [
        {"product_id": "p1", "parent_asin": "p1", "title": "Daily Moisturizing Cream"},
        {"product_id": "p2", "parent_asin": "p2", "title": "Fragrance Free Face Cream"},
        {"product_id": "p3", "parent_asin": "p3", "title": "Heavy Night Balm"},
    ]
    with (catalog_dir / "products.jsonl").open("w", encoding="utf-8") as handle:
        for product in products:
            handle.write(json.dumps(product) + "\n")
    with (catalog_dir / "texts.jsonl").open("w", encoding="utf-8") as handle:
        for product in products:
            handle.write(
                json.dumps(
                    {
                        "product_id": product["product_id"],
                        "embedding_text": product["title"],
                        "rerank_text": product["title"],
                        "embedding_text_version": "test.embedding.v1",
                        "rerank_text_version": "test.rerank.v1",
                    }
                )
                + "\n"
            )
    manifest = {
        "catalog_version": "tiny_v1",
        "schema_version": "1.0",
        "source_dataset": "test",
        "source_category": "All_Beauty",
        "created_at": utc_now_iso(),
        "record_count": len(products),
        "product_id_field": "product_id",
        "source_id_field": "parent_asin",
        "checksum_sha256": "0" * 64,
        "text_builder_version": "test.embedding.v1;test.rerank.v1",
    }
    write_json(catalog_dir / "manifest.json", manifest)
    return catalog_dir


def test_build_verify_and_search_bundle(tmp_path: Path) -> None:
    catalog_dir = _write_catalog(tmp_path)
    encoder = _DeterministicEncoder()
    bundle_dir = tmp_path / "bundle"

    manifest = build_bundle(
        catalog_dir=catalog_dir,
        output_dir=bundle_dir,
        encoder=encoder,  # type: ignore[arg-type]
        index_type="IndexFlatIP",
        normalize_vectors=True,
    )
    assert manifest.num_vectors == 3
    assert manifest.embedding_dim == DIM
    assert (bundle_dir / "index.faiss").is_file()
    assert (bundle_dir / "embeddings.npy").is_file()

    verify_bundle(bundle_dir)
    retriever = load_retriever(
        bundle_dir,
        encoder,
        top_k_per_query=2,
        rrf_k=60,
        max_queries=3,  # type: ignore[arg-type]
    )
    output = retriever.search(["Daily Moisturizing Cream", "Heavy Night Balm"], top_k_per_query=2)
    assert output.fused.query_count == 2
    # Two queries x top-2 over three products: the union is exactly the two self-hits.
    assert {hit.product_id for hit in output.fused.hits} == {"p1", "p3"}
    assert all(hit.raw_score is not None for hit in output.fused.hits)
    # The exact-text query must retrieve its own product first within that query.
    assert output.per_query[0].hits[0].product_id == "p1"
    assert output.per_query[1].hits[0].product_id == "p3"
