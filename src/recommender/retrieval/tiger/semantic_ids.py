"""Semantic ID table: product_id <-> multi-level codes with collision statistics.

The table is built from RQ-VAE codes for the project's own catalog. Because multiple
products may share one Semantic ID (collision), decoding returns a *list* of products;
callers must disambiguate (e.g. by generation score and catalog evidence) instead of
pretending the mapping is one-to-one.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recommender.observability.logging import get_logger
from recommender.utils.errors import ArtifactError
from recommender.utils.io import ensure_dir, file_sha256, read_json, utc_now_iso, write_json

_LOGGER = get_logger(__name__)


def format_semantic_id(semantic_id: tuple[int, ...] | list[int]) -> str:
    return ",".join(str(int(token)) for token in semantic_id)


def parse_semantic_id(text: str) -> tuple[int, ...]:
    return tuple(int(token) for token in text.split(",") if token.strip())


@dataclass
class SemanticIdTable:
    product_ids: list[str]
    codes: np.ndarray
    id_to_products: dict[str, list[str]]

    @classmethod
    def build(cls, product_ids: list[str], codes: np.ndarray) -> SemanticIdTable:
        if len(product_ids) != len(codes):
            raise ArtifactError(f"product_ids ({len(product_ids)}) != codes ({len(codes)})")
        if len(set(product_ids)) != len(product_ids):
            raise ArtifactError("duplicate product_id in semantic ID table")
        id_to_products: dict[str, list[str]] = {}
        for product_id, code_row in zip(product_ids, codes, strict=True):
            key = format_semantic_id([int(token) for token in code_row])
            id_to_products.setdefault(key, []).append(product_id)
        return cls(
            product_ids=product_ids,
            codes=codes.astype(np.int32),
            id_to_products=id_to_products,
        )

    @property
    def num_products(self) -> int:
        return len(self.product_ids)

    @property
    def num_unique_ids(self) -> int:
        return len(self.id_to_products)

    def semantic_id_of(self, product_id: str) -> tuple[int, ...] | None:
        try:
            row = self.product_ids.index(product_id)
        except ValueError:
            return None
        return tuple(int(token) for token in self.codes[row])

    def products_for(self, semantic_id: tuple[int, ...] | list[int] | str) -> list[str]:
        key = semantic_id if isinstance(semantic_id, str) else format_semantic_id(semantic_id)
        return list(self.id_to_products.get(key, []))

    def save(
        self,
        directory: Path | str,
        *,
        extra_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = ensure_dir(directory)
        ids_path = target / "semantic_ids.jsonl"
        with ids_path.open("w", encoding="utf-8", newline="\n") as handle:
            for product_id, code_row in zip(self.product_ids, self.codes, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "product_id": product_id,
                            "semantic_id": [int(token) for token in code_row],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        mapping_path = target / "id_to_products.json"
        write_json(mapping_path, self.id_to_products)
        stats = collision_stats(self.codes)
        manifest = {
            "schema_version": "1.0",
            "created_at": utc_now_iso(),
            "num_products": self.num_products,
            "num_unique_ids": self.num_unique_ids,
            "num_levels": int(self.codes.shape[1]),
            "collision": stats,
            "semantic_ids_sha256": file_sha256(ids_path),
            "id_to_products_sha256": file_sha256(mapping_path),
            **(extra_manifest or {}),
        }
        write_json(target / "semantic_id_manifest.json", manifest)
        _LOGGER.info(
            "semantic ID table saved",
            extra={
                "products": self.num_products,
                "unique_ids": self.num_unique_ids,
                "collision_rate": stats["collision_rate"],
            },
        )
        return manifest

    @classmethod
    def load(cls, directory: Path | str) -> SemanticIdTable:
        target = Path(directory)
        ids_path = target / "semantic_ids.jsonl"
        if not ids_path.is_file():
            raise ArtifactError(f"semantic_ids.jsonl not found in {target}")
        product_ids: list[str] = []
        codes: list[list[int]] = []
        with ids_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                product_ids.append(row["product_id"])
                codes.append([int(token) for token in row["semantic_id"]])
        mapping = read_json(target / "id_to_products.json")
        return cls(
            product_ids=product_ids,
            codes=np.asarray(codes, dtype=np.int32),
            id_to_products={str(key): list(value) for key, value in mapping.items()},
        )


def collision_stats(codes: np.ndarray) -> dict[str, Any]:
    """Count how many products share each Semantic ID (prefix and full ID)."""
    if codes.ndim != 2:
        raise ArtifactError(f"codes must be 2D, got {codes.shape}")
    count = len(codes)
    full_ids = [format_semantic_id(row) for row in codes]
    full_counter = Counter(full_ids)
    full_collisions = {key: value for key, value in full_counter.items() if value > 1}
    collision_products = sum(full_collisions.values()) if full_collisions else 0
    prefix_stats: dict[str, Any] = {}
    for prefix_len in range(1, codes.shape[1] + 1):
        prefix_ids = [format_semantic_id(row[:prefix_len]) for row in codes]
        counter = Counter(prefix_ids)
        collisions = sum(value for value in counter.values() if value > 1)
        prefix_stats[f"prefix_{prefix_len}"] = {
            "unique_ids": len(counter),
            "collision_products": collisions,
            "collision_rate": round(collisions / count, 4) if count else 0.0,
            "max_products_per_id": max(counter.values()) if counter else 0,
        }
    return {
        "num_products": count,
        "unique_ids": len(full_counter),
        "collision_ids": len(full_collisions),
        "collision_products": collision_products,
        "collision_rate": round(collision_products / count, 4) if count else 0.0,
        "max_products_per_id": max(full_counter.values()) if full_counter else 0,
        "prefix_stats": prefix_stats,
    }
