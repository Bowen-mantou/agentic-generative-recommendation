"""Build a versioned product catalog from raw Amazon Reviews 2023 metadata.

Outputs (inside ``<out_root>/<catalog_version>/``):

* ``products.jsonl`` — validated :class:`ProductRecord` rows (one per product).
* ``texts.jsonl`` — :class:`ProductTextPair` embedding/rerank texts.
* ``manifest.json`` — :class:`CatalogManifest` with hashes, counts and source info.

The source metadata file is never modified; the catalog is a derived artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from recommender.catalog.amazon import (
    DATASET_ID,
    build_product_record,
    iter_jsonl_records,
    metadata_url,
)
from recommender.catalog.text import (
    EMBEDDING_TEXT_VERSION,
    RERANK_TEXT_VERSION,
    build_embedding_text,
    build_rerank_text,
    has_searchable_text,
)
from recommender.observability.logging import get_logger
from recommender.schemas.artifacts import CatalogBuildStats
from recommender.schemas.product import CatalogManifest, ProductRecord, ProductTextPair
from recommender.utils.io import ensure_dir, file_sha256, utc_now_iso, write_json

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class CatalogBuildResult:
    catalog_dir: Path
    products_path: Path
    texts_path: Path
    manifest: CatalogManifest

    @property
    def record_count(self) -> int:
        return self.manifest.record_count


def build_catalog(
    meta_path: Path | str,
    output_root: Path | str,
    *,
    catalog_version: str,
    category: str,
    source_dataset: str = DATASET_ID,
    source_url: str | None = None,
    limit: int | None = None,
    log_every: int = 20000,
) -> CatalogBuildResult:
    """Parse metadata into a validated catalog and product texts.

    ``limit`` exists for fast smoke runs; the truncated catalog is still a real catalog
    (never mock data) and its manifest records the limit implicitly through
    ``record_count`` and the generation command.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    source = Path(meta_path)
    if not source.is_file():
        raise FileNotFoundError(f"metadata file not found: {source}")
    catalog_dir = ensure_dir(Path(output_root) / catalog_version)
    products_path = catalog_dir / "products.jsonl"
    texts_path = catalog_dir / "texts.jsonl"

    stats = CatalogBuildStats(records_read=0, records_written=0)
    seen: set[str] = set()
    with (
        products_path.open("w", encoding="utf-8", newline="\n") as products_file,
        texts_path.open("w", encoding="utf-8", newline="\n") as texts_file,
    ):
        for line_number, raw in iter_jsonl_records(source):
            stats.records_read += 1
            if raw is None:
                stats.skipped_malformed_json += 1
                _LOGGER.warning("malformed metadata line skipped", extra={"line": line_number})
                continue
            try:
                product = build_product_record(raw)
            except ValidationError as exc:
                stats.skipped_invalid_record += 1
                _LOGGER.warning(
                    "invalid metadata record skipped",
                    extra={"line": line_number, "error": str(exc)[:300]},
                )
                continue
            if product is None:
                stats.skipped_missing_id += 1
                continue
            if product.product_id in seen:
                stats.skipped_duplicate_id += 1
                continue
            if not has_searchable_text(product):
                stats.skipped_missing_text += 1
                continue
            seen.add(product.product_id)
            pair = ProductTextPair(
                product_id=product.product_id,
                embedding_text=build_embedding_text(product),
                rerank_text=build_rerank_text(product),
                embedding_text_version=EMBEDDING_TEXT_VERSION,
                rerank_text_version=RERANK_TEXT_VERSION,
            )
            products_file.write(product.model_dump_json() + "\n")
            texts_file.write(pair.model_dump_json() + "\n")
            stats.records_written += 1
            if stats.records_written % log_every == 0:
                _LOGGER.info(
                    "catalog build progress",
                    extra={"written": stats.records_written, "read": stats.records_read},
                )
            if limit is not None and stats.records_written >= limit:
                _LOGGER.info("catalog build stopped at limit", extra={"limit": limit})
                break

    if stats.records_written == 0:
        raise ValueError("catalog build produced zero products; check the source file")

    manifest = CatalogManifest(
        catalog_version=catalog_version,
        source_dataset=source_dataset,
        source_category=category,
        created_at=utc_now_iso(),
        record_count=stats.records_written,
        checksum_sha256=file_sha256(products_path),
        source_url=source_url or metadata_url(category),
        source_file_sha256=file_sha256(source),
        text_builder_version=f"embedding={EMBEDDING_TEXT_VERSION};rerank={RERANK_TEXT_VERSION}",
        stats=stats,
    )
    write_json(catalog_dir / "manifest.json", manifest.model_dump(mode="json"))
    _LOGGER.info(
        "catalog built",
        extra={
            "catalog_version": catalog_version,
            "records": stats.records_written,
            "read": stats.records_read,
            "skipped": stats.model_dump(),
        },
    )
    return CatalogBuildResult(
        catalog_dir=catalog_dir,
        products_path=products_path,
        texts_path=texts_path,
        manifest=manifest,
    )


def load_products(products_path: Path | str) -> list[ProductRecord]:
    """Load and re-validate a catalog (used by consumers/tests)."""
    products: list[ProductRecord] = []
    with Path(products_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                products.append(ProductRecord.model_validate_json(stripped))
    return products


def load_text_pairs(texts_path: Path | str) -> list[ProductTextPair]:
    pairs: list[ProductTextPair] = []
    with Path(texts_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                pairs.append(ProductTextPair.model_validate_json(stripped))
    return pairs


def catalog_manifest_dict(catalog_dir: Path | str) -> dict[str, Any]:
    """Read a catalog manifest as a plain dict."""
    import json

    return json.loads((Path(catalog_dir) / "manifest.json").read_text(encoding="utf-8"))
