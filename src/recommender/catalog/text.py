"""Deterministic product text builders.

Two templates with independent versions:

* embedding text — semantic retrieval input; no price, no inferred attributes.
* rerank text — cross-encoder document; includes price and catalog attributes.

Both only use fields that exist in the catalog. A missing attribute is omitted, never
reconstructed from the title (Reranker doc §5.3: "Do not assume missing product
attributes are satisfied").
"""

from __future__ import annotations

from recommender.schemas.product import ProductRecord

EMBEDDING_TEXT_VERSION = "product_text.embedding.v1"
RERANK_TEXT_VERSION = "product_text.rerank.v1"

MAX_EMBEDDING_CHARS = 1200
MAX_RERANK_CHARS = 2400
MAX_FEATURES_IN_TEXT = 8
MAX_ATTRIBUTES_IN_TEXT = 12


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = " ".join(value.split())
    return trimmed or None


def _squash(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def has_searchable_text(product: ProductRecord) -> bool:
    """Whether the product carries any field worth encoding."""
    return any(
        _clean(value) is not None
        for value in (product.title, product.description, product.brand, product.main_category)
    ) or bool(product.features)


def build_embedding_text(product: ProductRecord) -> str:
    """Semantic text for E5 passage encoding (module 3 §4.2)."""
    lines: list[str] = [f"Product ID: {product.product_id}"]
    title = _clean(product.title)
    if title:
        lines.append(f"Title: {title}")
    if product.main_category:
        categories = " / ".join(product.categories[:3])
        lines.append(f"Category: {categories}")
    if product.brand:
        lines.append(f"Brand: {product.brand}")
    if product.features:
        features = "; ".join(_clean(item) or "" for item in product.features[:MAX_FEATURES_IN_TEXT])
        if features.strip():
            lines.append(f"Features: {features}")
    description = _clean(product.description)
    if description:
        lines.append(f"Description: {description}")
    return _squash("\n".join(lines), MAX_EMBEDDING_CHARS)


def build_rerank_text(product: ProductRecord) -> str:
    """Document text for the cross-encoder (Reranker doc §5.3)."""
    lines: list[str] = [f"Product ID: {product.product_id}"]
    title = _clean(product.title)
    if title:
        lines.append(f"Title: {title}")
    if product.main_category:
        categories = " / ".join(product.categories[:3])
        lines.append(f"Category: {categories}")
    if product.brand:
        lines.append(f"Brand: {product.brand}")
    if product.price is not None:
        lines.append(f"Price: ${product.price:.2f} {product.currency}")
    attributes: list[str] = []
    for key, value in list(product.details.items())[:MAX_ATTRIBUTES_IN_TEXT]:
        clean_key, clean_value = _clean(key), _clean(value)
        if clean_key and clean_value:
            attributes.append(f"- {clean_key}: {clean_value}")
    if attributes:
        lines.append("Attributes:")
        lines.extend(attributes)
    if product.features:
        lines.append("Features:")
        lines.extend(
            f"- {feature}"
            for feature in (_clean(item) for item in product.features[:MAX_FEATURES_IN_TEXT])
            if feature
        )
    description = _clean(product.description)
    if description:
        lines.append(f"Description: {description}")
    return _squash("\n".join(lines), MAX_RERANK_CHARS)
