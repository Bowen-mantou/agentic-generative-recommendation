"""Amazon Reviews 2023 raw data access.

The dataset (McAuley-Lab/Amazon-Reviews-2023) exposes per-category JSONL metadata and
review files. Files are downloaded verbatim, hashed and recorded in the catalog
manifest so every artifact is reproducible. Direct huggingface.co access is blocked on
the lab network; ``hf-mirror.com`` works, so the endpoint is configurable via
``HF_ENDPOINT``.

Only review/rating proxy events exist in this dataset; nothing here infers clicks or
purchases (AGENTS.md §2.2).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from recommender.observability.logging import get_logger
from recommender.schemas.product import ProductRecord, ProductSource
from recommender.utils.errors import ExternalAPIError
from recommender.utils.io import ensure_dir

_LOGGER = get_logger(__name__)

DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"
META_TEMPLATE = "raw/meta_categories/meta_{category}.jsonl"
REVIEW_TEMPLATE = "raw/review_categories/{category}.jsonl"
DEFAULT_ENDPOINT = "https://huggingface.co"
_CHUNK_SIZE = 4 * 1024 * 1024
_PRICE_NUMBER = re.compile(r"(\d+(?:\.\d+)?)")
MAX_DETAILS = 40
MAX_DETAIL_CHARS = 300


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    url: str
    sha256: str
    size_bytes: int


def resolve_endpoint(explicit: str | None = None, environ: Mapping[str, str] | None = None) -> str:
    """Resolve the HuggingFace endpoint (explicit > env > default)."""
    env = os.environ if environ is None else environ
    endpoint = explicit or env.get("HF_ENDPOINT") or env.get("AGR_HF_ENDPOINT") or DEFAULT_ENDPOINT
    return endpoint.rstrip("/")


def dataset_file_url(
    relative_path: str,
    *,
    endpoint: str | None = None,
    revision: str = "main",
    environ: Mapping[str, str] | None = None,
) -> str:
    """Build a resolve URL for one dataset file."""
    base = resolve_endpoint(endpoint, environ)
    return f"{base}/datasets/{DATASET_ID}/resolve/{revision}/{quote(relative_path)}"


def metadata_url(category: str, *, endpoint: str | None = None) -> str:
    return dataset_file_url(META_TEMPLATE.format(category=category), endpoint=endpoint)


def reviews_url(category: str, *, endpoint: str | None = None) -> str:
    return dataset_file_url(REVIEW_TEMPLATE.format(category=category), endpoint=endpoint)


def download_file(
    url: str,
    destination: Path | str,
    *,
    timeout_s: float = 120.0,
    max_attempts: int = 3,
    force: bool = False,
) -> DownloadResult:
    """Stream a remote file to disk with a SHA-256 digest and bounded retries.

    A partially written ``.part`` file is never promoted to the final name.
    """
    target = Path(destination)
    ensure_dir(target.parent)
    if target.is_file() and not force:
        digest = hashlib.sha256()
        size = 0
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
                size += len(chunk)
        return DownloadResult(path=target, url=url, sha256=digest.hexdigest(), size_bytes=size)

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        temporary = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        size = 0
        try:
            timeout = httpx.Timeout(timeout_s, connect=20.0)
            with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
                if response.status_code >= 400:
                    raise ExternalAPIError(
                        f"download failed with HTTP {response.status_code}: {url}",
                        provider="huggingface",
                        status_code=response.status_code,
                        retryable=response.status_code >= 500 or response.status_code == 429,
                    )
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(_CHUNK_SIZE):
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            os.replace(temporary, target)
            _LOGGER.info(
                "downloaded dataset file",
                extra={"url": url, "size_bytes": size, "attempt": attempt},
            )
            return DownloadResult(path=target, url=url, sha256=digest.hexdigest(), size_bytes=size)
        except (httpx.HTTPError, ExternalAPIError, OSError) as exc:
            temporary.unlink(missing_ok=True)
            last_error = exc
            retryable = not isinstance(exc, ExternalAPIError) or exc.retryable
            if attempt >= max_attempts or not retryable:
                break
            backoff = 2.0 * attempt
            _LOGGER.warning(
                "download attempt failed; retrying",
                extra={"url": url, "attempt": attempt, "backoff_s": backoff, "error": str(exc)},
            )
            time.sleep(backoff)
    raise ExternalAPIError(
        f"could not download {url}: {last_error}",
        provider="huggingface",
        retryable=False,
    )


def iter_jsonl_records(path: Path | str) -> Iterator[tuple[int, dict[str, Any] | None]]:
    """Yield ``(line_number, record)``; malformed lines yield ``None``.

    Downstream ingestion counts malformed lines instead of silently dropping them.
    """
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
            if isinstance(record, dict):
                yield line_number, record
            else:
                yield line_number, None


def parse_price(raw: Any) -> float | None:
    """Parse Amazon's heterogeneous price field.

    Handles numeric values, ``"$25.99"``, ``"25.99 USD"`` and ranges such as
    ``"$9.99 - $19.99"`` (the first number is used). Returns ``None`` when no price can
    be determined — missing price must stay missing, never defaulted (AGENTS.md §2.4).
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if value >= 0 else None
    if isinstance(raw, str):
        cleaned = raw.strip().replace(",", "")
        if not cleaned:
            return None
        match = _PRICE_NUMBER.search(cleaned)
        if match is None:
            return None
        value = float(match.group(1))
        return value if value >= 0 else None
    return None


def _as_text(value: Any) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _join_text(value: Any) -> str | None:
    if isinstance(value, str):
        return _as_text(value)
    if isinstance(value, list):
        parts = [text for item in value if (text := _as_text(item))]
        return "\n".join(parts) if parts else None
    return None


def normalize_string_list(value: Any, *, limit: int = 64) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [text for item in value if (text := _as_text(item))]
    return items[:limit]


def normalize_details(raw: Any) -> dict[str, str]:
    """Flatten Amazon ``details`` into ``{attribute: value}`` strings."""
    if not isinstance(raw, Mapping):
        return {}
    details: dict[str, str] = {}
    for key, value in raw.items():
        key_text = _as_text(key)
        value_text = _as_text(value)
        if key_text is None or value_text is None:
            continue
        details[key_text] = value_text[:MAX_DETAIL_CHARS]
        if len(details) >= MAX_DETAILS:
            break
    return details


def build_product_record(record: Mapping[str, Any]) -> ProductRecord | None:
    """Map one raw Amazon metadata record to :class:`ProductRecord`.

    ``product_id`` is the raw ``parent_asin`` (the mapping happens here, at the data
    access boundary). Records without an id are skipped by the caller.
    """
    parent_asin = _as_text(record.get("parent_asin"))
    if parent_asin is None:
        return None
    categories = normalize_string_list(record.get("categories"), limit=32)
    description = _join_text(record.get("description"))
    features = normalize_string_list(record.get("features"), limit=64)
    title = _as_text(record.get("title")) or ""
    details = normalize_details(record.get("details"))
    store = _as_text(record.get("store"))
    average_rating: float | None = None
    raw_rating = record.get("average_rating")
    if isinstance(raw_rating, (int, float)) and not isinstance(raw_rating, bool):
        rating = float(raw_rating)
        average_rating = rating if 0.0 <= rating <= 5.0 else None
    rating_number: int | None = None
    raw_count = record.get("rating_number")
    if isinstance(raw_count, (int, float)) and not isinstance(raw_count, bool):
        count = int(raw_count)
        rating_number = count if count >= 0 else None
    # Amazon Reviews 2023 exposes a top-level `main_category` and (for this dump) an
    # always-empty `categories` list; prefer the explicit field, then the path.
    main_category = _as_text(record.get("main_category"))
    if main_category is None and categories:
        main_category = categories[0]
    return ProductRecord(
        product_id=parent_asin,
        parent_asin=parent_asin,
        title=title,
        brand=store,
        main_category=main_category,
        categories=categories,
        price=parse_price(record.get("price")),
        description=description,
        features=features,
        details=details,
        average_rating=average_rating,
        rating_number=rating_number,
        store=store,
        source=ProductSource.AMAZON_REVIEWS_2023,
        is_synthetic=False,
    )
