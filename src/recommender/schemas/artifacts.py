"""Artifact manifests for catalog / embedding / index bundles.

Every artifact is versioned and carries hashes so an evaluation run can be tied to
exact inputs (AGENTS.md §6: experiments record data and model versions).
"""

from __future__ import annotations

from pydantic import Field, field_validator

from recommender.schemas.common import ContractModel, NonEmptyStr


class E5IndexManifest(ContractModel):
    """Description of one built E5 + FAISS bundle."""

    schema_version: str = "1.0"
    index_version: NonEmptyStr
    catalog_version: NonEmptyStr
    text_builder_version: NonEmptyStr
    model_id: NonEmptyStr
    model_revision: NonEmptyStr
    embedding_dim: int = Field(gt=0)
    normalize_vectors: bool
    metric: str
    index_type: str
    num_vectors: int = Field(gt=0)
    query_prefix: str = ""
    passage_prefix: str = ""
    max_length: int = Field(gt=0)
    created_at: NonEmptyStr
    index_sha256: NonEmptyStr
    product_ids_sha256: NonEmptyStr
    catalog_manifest_sha256: NonEmptyStr

    @field_validator("index_type")
    @classmethod
    def _supported_index_type(cls, value: str) -> str:
        supported = {"IndexFlatIP", "IndexFlatL2", "IndexIVFFlat"}
        if value not in supported:
            raise ValueError(f"unsupported FAISS index type: {value}")
        return value


class CatalogBuildStats(ContractModel):
    """Ingestion counters, useful for spotting silent data loss."""

    records_read: int = Field(ge=0)
    records_written: int = Field(ge=0)
    skipped_missing_id: int = Field(default=0, ge=0)
    skipped_missing_text: int = Field(default=0, ge=0)
    skipped_duplicate_id: int = Field(default=0, ge=0)
    skipped_malformed_json: int = Field(default=0, ge=0)
    skipped_invalid_record: int = Field(default=0, ge=0)
