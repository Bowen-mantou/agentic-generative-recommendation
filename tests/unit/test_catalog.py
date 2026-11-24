"""Amazon metadata parsing, text builders and catalog build."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recommender.catalog.amazon import (
    build_product_record,
    dataset_file_url,
    iter_jsonl_records,
    normalize_details,
    parse_price,
    resolve_endpoint,
)
from recommender.catalog.build import build_catalog, load_products, load_text_pairs
from recommender.catalog.text import build_embedding_text, build_rerank_text, has_searchable_text
from recommender.schemas.product import ProductRecord

pytestmark = pytest.mark.unit

RAW_PRODUCT = {
    "parent_asin": "B0001",
    "title": "Daily Moisturizing Cream",
    "store": "Brand A",
    "categories": ["Beauty & Personal Care", "Skin Care", "Moisturizers"],
    "price": "$25.99",
    "description": ["Lightweight moisturizing cream", "suitable for daily use."],
    "features": ["Fragrance-free", "Moisturizing"],
    "details": {"Scent": "Unscented", "Item Form": "Cream", "Nested": {"x": 1}},
    "average_rating": 4.6,
    "rating_number": 128,
}


def test_parse_price_variants() -> None:
    assert parse_price("$25.99") == 25.99
    assert parse_price("25.99 USD") == 25.99
    assert parse_price("$1,299.00") == 1299.0
    assert parse_price("from $9.99 to $19.99") == 9.99
    assert parse_price(31) == 31.0
    assert parse_price("") is None
    assert parse_price(None) is None
    assert parse_price(True) is None
    assert parse_price("Price unavailable") is None
    assert parse_price(-5) is None


def test_build_product_record_maps_parent_asin_to_product_id() -> None:
    product = build_product_record(RAW_PRODUCT)
    assert product is not None
    assert product.product_id == "B0001"
    assert product.parent_asin == "B0001"
    assert product.title == "Daily Moisturizing Cream"
    assert product.brand == "Brand A"
    assert product.main_category == "Beauty & Personal Care"
    assert product.price == 25.99
    assert "suitable for daily use." in (product.description or "")
    assert product.details["Scent"] == "Unscented"
    assert "Nested" not in product.details
    assert product.average_rating == 4.6
    assert product.rating_number == 128
    assert product.is_synthetic is False


def test_main_category_prefers_explicit_field() -> None:
    record = dict(RAW_PRODUCT)
    record["main_category"] = "All Beauty"
    record["categories"] = []
    product = build_product_record(record)
    assert product is not None
    assert product.main_category == "All Beauty"


def test_main_category_falls_back_to_categories() -> None:
    record = dict(RAW_PRODUCT)
    record.pop("main_category", None)
    product = build_product_record(record)
    assert product is not None
    assert product.main_category == "Beauty & Personal Care"


def test_build_product_record_requires_parent_asin() -> None:
    assert build_product_record({"title": "no id"}) is None


def test_out_of_range_rating_stays_unknown() -> None:
    record = dict(RAW_PRODUCT)
    record["average_rating"] = 7.5
    product = build_product_record(record)
    assert product is not None
    assert product.average_rating is None


def test_normalize_details_keeps_only_scalars() -> None:
    details = normalize_details({"A": 1, "B": None, "C": ["x"], "D": {"nested": 1}, "E": "y"})
    assert details == {"A": "1", "E": "y"}


def test_iter_jsonl_records_reports_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "meta.jsonl"
    path.write_text('{"a": 1}\n\nnot-json\n{"b": 2}\n', encoding="utf-8")
    records = list(iter_jsonl_records(path))
    assert [line for line, _ in records] == [1, 3, 4]
    assert records[0][1] == {"a": 1}
    assert records[1][1] is None
    assert records[2][1] == {"b": 2}


def test_endpoint_and_url_resolution() -> None:
    assert resolve_endpoint("https://hf-mirror.com/") == "https://hf-mirror.com"
    assert (
        resolve_endpoint(None, {"HF_ENDPOINT": "https://hf-mirror.com"}) == "https://hf-mirror.com"
    )
    url = dataset_file_url(
        "raw/meta_categories/meta_All_Beauty.jsonl", endpoint="https://hf-mirror.com"
    )
    assert url == (
        "https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
        "raw/meta_categories/meta_All_Beauty.jsonl"
    )


def test_text_builders_use_only_present_fields() -> None:
    product = build_product_record(RAW_PRODUCT)
    assert product is not None
    embedding_text = build_embedding_text(product)
    assert "Title: Daily Moisturizing Cream" in embedding_text
    assert "Category: Beauty & Personal Care" in embedding_text
    assert "Features: Fragrance-free; Moisturizing" in embedding_text
    assert "Price" not in embedding_text

    rerank_text = build_rerank_text(product)
    assert "Price: $25.99 USD" in rerank_text
    assert "Scent: Unscented" in rerank_text

    bare = ProductRecord(product_id="p1", parent_asin="p1", title="Plain item")
    bare_text = build_rerank_text(bare)
    assert "Attributes:" not in bare_text
    assert "fragrance" not in bare_text.lower()
    assert "Price:" not in bare_text


def test_has_searchable_text() -> None:
    assert has_searchable_text(ProductRecord(product_id="p", parent_asin="p", title="x"))
    assert not has_searchable_text(ProductRecord(product_id="p", parent_asin="p"))


def _write_meta(path: Path) -> None:
    lines = [
        json.dumps(RAW_PRODUCT),
        json.dumps({"parent_asin": "B0002", "title": "Cheap cream", "price": "$9.99"}),
        # duplicate id -> skipped
        json.dumps({**RAW_PRODUCT, "title": "Duplicate title"}),
        # missing id -> skipped
        json.dumps({"title": "No identifier"}),
        # no searchable text -> skipped
        json.dumps({"parent_asin": "B0003"}),
        "not-json",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_catalog_counts_and_artifacts(tmp_path: Path) -> None:
    meta = tmp_path / "meta_All_Beauty.jsonl"
    _write_meta(meta)
    result = build_catalog(
        meta,
        tmp_path / "catalog",
        catalog_version="all_beauty_v1",
        category="All_Beauty",
        log_every=1000,
    )
    assert result.record_count == 2
    stats = result.manifest.stats
    assert stats is not None
    assert stats.records_read == 6
    assert stats.records_written == 2
    assert stats.skipped_duplicate_id == 1
    assert stats.skipped_missing_id == 1
    assert stats.skipped_missing_text == 1
    assert stats.skipped_malformed_json == 1
    assert result.manifest.checksum_sha256
    assert result.manifest.source_file_sha256
    assert result.manifest.catalog_version == "all_beauty_v1"
    assert result.catalog_dir == tmp_path / "catalog" / "all_beauty_v1"
    products = load_products(result.products_path)
    texts = load_text_pairs(result.texts_path)
    assert [product.product_id for product in products] == ["B0001", "B0002"]
    assert texts[0].embedding_text.startswith("Product ID: B0001")
    assert texts[0].embedding_text_version == "product_text.embedding.v1"


def test_build_catalog_limit(tmp_path: Path) -> None:
    meta = tmp_path / "meta.jsonl"
    _write_meta(meta)
    result = build_catalog(
        meta,
        tmp_path / "catalog",
        catalog_version="limit1",
        category="All_Beauty",
        limit=1,
    )
    assert result.record_count == 1


def test_build_catalog_empty_source_is_an_error(tmp_path: Path) -> None:
    meta = tmp_path / "meta.jsonl"
    meta.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="zero products"):
        build_catalog(meta, tmp_path / "catalog", catalog_version="empty", category="All_Beauty")
