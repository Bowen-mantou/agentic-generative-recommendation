"""Product catalog contracts.

Internal primary key: ``product_id``. Amazon ``parent_asin`` is preserved for
traceability and mapped at the data-access layer (AGENTS.md §2.1). Missing attributes
stay missing: nothing here infers price, category or attributes.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from recommender.schemas.artifacts import CatalogBuildStats
from recommender.schemas.common import ContractModel, NonEmptyStr

MAX_TITLE_CHARS = 512
MAX_DESCRIPTION_CHARS = 8000
MAX_FEATURES = 64
MAX_CATEGORIES = 32


class ProductSource(str, Enum):
    AMAZON_REVIEWS_2023 = "amazon_reviews_2023"
    SYNTHETIC = "synthetic"


class ProductRecord(ContractModel):
    """One product in the unified catalog."""

    product_id: NonEmptyStr
    parent_asin: NonEmptyStr
    title: str = ""
    brand: str | None = None
    main_category: str | None = None
    categories: list[str] = Field(default_factory=list, max_length=MAX_CATEGORIES)
    price: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    description: str | None = None
    features: list[str] = Field(default_factory=list, max_length=MAX_FEATURES)
    # Store-specific structured attributes (e.g. {"Scent": "Unscented"}). A missing key
    # means "unknown", never "satisfied".
    details: dict[str, str] = Field(default_factory=dict)
    average_rating: float | None = Field(default=None, ge=0.0, le=5.0)
    rating_number: int | None = Field(default=None, ge=0)
    store: str | None = None
    source: ProductSource = ProductSource.AMAZON_REVIEWS_2023
    is_synthetic: bool = False

    @field_validator("product_id")
    @classmethod
    def _product_id_has_no_whitespace(cls, value: str) -> str:
        if any(char.isspace() for char in value):
            raise ValueError("product_id must not contain whitespace")
        return value

    @field_validator("parent_asin")
    @classmethod
    def _parent_asin_has_no_whitespace(cls, value: str) -> str:
        if any(char.isspace() for char in value):
            raise ValueError("parent_asin must not contain whitespace")
        return value

    @field_validator("title")
    @classmethod
    def _truncate_title(cls, value: str) -> str:
        return value.strip()[:MAX_TITLE_CHARS]

    @field_validator("description")
    @classmethod
    def _truncate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed[:MAX_DESCRIPTION_CHARS] if trimmed else None

    @field_validator("brand", "main_category", "store", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _synthetic_flag_matches_source(self) -> Self:
        if self.source is ProductSource.SYNTHETIC and not self.is_synthetic:
            raise ValueError("source=synthetic requires is_synthetic=true")
        if self.source is ProductSource.AMAZON_REVIEWS_2023 and self.is_synthetic:
            raise ValueError("amazon_reviews_2023 records cannot be synthetic")
        return self


class ProductText(ContractModel):
    """Catalog text prepared for an encoder or reranker, with provenance.

    ``builder_version`` identifies the text template so embeddings can be invalidated
    when the template changes.
    """

    product_id: NonEmptyStr
    builder_version: NonEmptyStr
    text: NonEmptyStr

    @field_validator("text")
    @classmethod
    def _reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("product text must not be blank")
        return value


class ProductTextPair(ContractModel):
    """Versioned embedding + rerank texts for one product."""

    product_id: NonEmptyStr
    embedding_text: NonEmptyStr
    rerank_text: NonEmptyStr
    embedding_text_version: NonEmptyStr
    rerank_text_version: NonEmptyStr


class CatalogManifest(ContractModel):
    """Versioned description of a built product catalog."""

    catalog_version: NonEmptyStr
    schema_version: str = "1.0"
    source_dataset: NonEmptyStr
    source_category: NonEmptyStr
    created_at: NonEmptyStr
    record_count: int = Field(ge=0)
    product_id_field: str = "product_id"
    source_id_field: str = "parent_asin"
    checksum_sha256: NonEmptyStr
    source_url: str | None = None
    source_file_sha256: str | None = None
    text_builder_version: str | None = None
    stats: CatalogBuildStats | None = None

    @field_validator("product_id_field")
    @classmethod
    def _product_key_is_product_id(cls, value: str) -> str:
        if value != "product_id":
            raise ValueError("catalog primary key must be product_id")
        return value
