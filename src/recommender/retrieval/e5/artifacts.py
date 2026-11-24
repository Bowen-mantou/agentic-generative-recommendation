"""Build / load versioned E5 + FAISS bundles.

Bundle layout::

    <bundle_dir>/
        index.faiss          # FAISS index (row order == product_ids.jsonl order)
        product_ids.jsonl    # one product_id per line
        embeddings.npy       # float32 [N, D] passage vectors (reproducible rebuilds)
        manifest.json        # E5IndexManifest with hashes
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from recommender.catalog.build import load_text_pairs
from recommender.observability.logging import get_logger
from recommender.retrieval.e5.encoder import E5Encoder
from recommender.retrieval.e5.index import FaissVectorIndex, build_index, save_index
from recommender.retrieval.e5.service import E5Retriever
from recommender.schemas.artifacts import E5IndexManifest
from recommender.schemas.product import CatalogManifest
from recommender.utils.errors import ArtifactError
from recommender.utils.io import ensure_dir, file_sha256, read_json, utc_now_iso, write_json

_LOGGER = get_logger(__name__)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32)


def build_bundle(
    *,
    catalog_dir: Path | str,
    output_dir: Path | str,
    encoder: E5Encoder,
    index_type: str = "IndexFlatIP",
    normalize_vectors: bool = True,
    encode_batch_size: int | None = None,
    log_every: int = 5000,
) -> E5IndexManifest:
    """Encode every catalog product and persist a FAISS bundle."""
    catalog_path = Path(catalog_dir)
    catalog_manifest = CatalogManifest.model_validate(read_json(catalog_path / "manifest.json"))
    pairs = load_text_pairs(catalog_path / "texts.jsonl")
    if len(pairs) != catalog_manifest.record_count:
        raise ArtifactError(
            f"catalog texts ({len(pairs)}) != manifest record_count "
            f"({catalog_manifest.record_count})"
        )
    bundle_dir = ensure_dir(Path(output_dir))
    batch = encode_batch_size or encoder.batch_size

    # Length-bucketed batching: sort by text length so each batch pads to a similar
    # length instead of the global worst case (5-10x faster on CPU, identical vectors).
    order = sorted(range(len(pairs)), key=lambda index: len(pairs[index].embedding_text))
    dimension = encoder.embedding_dim
    vectors = np.empty((len(pairs), dimension), dtype=np.float32)
    started = time.perf_counter()
    last_logged = 0
    for start in range(0, len(pairs), batch):
        indices = order[start : start + batch]
        chunk_vectors = encoder.encode_passages([pairs[index].embedding_text for index in indices])
        vectors[indices] = np.asarray(chunk_vectors, dtype=np.float32)
        done = min(start + batch, len(pairs))
        if done - last_logged >= log_every or done == len(pairs):
            elapsed = max(time.perf_counter() - started, 1e-9)
            _LOGGER.info(
                "encoding progress",
                extra={
                    "encoded": done,
                    "total": len(pairs),
                    "elapsed_s": round(elapsed, 1),
                    "texts_per_s": round(done / elapsed, 1),
                },
            )
            last_logged = done
    if normalize_vectors:
        vectors = _l2_normalize(vectors)

    index = build_index(vectors, index_type=index_type)
    index_path = save_index(index, bundle_dir / "index.faiss")
    np.save(bundle_dir / "embeddings.npy", vectors)

    product_ids_path = bundle_dir / "product_ids.jsonl"
    with product_ids_path.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps({"product_id": pair.product_id}, ensure_ascii=False) + "\n")

    manifest = E5IndexManifest(
        index_version=bundle_dir.name,
        catalog_version=catalog_manifest.catalog_version,
        text_builder_version=catalog_manifest.text_builder_version or "unknown",
        model_id=encoder.model_id,
        model_revision=encoder.revision,
        embedding_dim=int(vectors.shape[1]),
        normalize_vectors=normalize_vectors,
        metric="cosine" if normalize_vectors else "inner_product",
        index_type=index_type,
        num_vectors=int(vectors.shape[0]),
        query_prefix=encoder.query_prefix,
        passage_prefix=encoder.passage_prefix,
        max_length=encoder.max_length,
        created_at=utc_now_iso(),
        index_sha256=file_sha256(index_path),
        product_ids_sha256=file_sha256(product_ids_path),
        catalog_manifest_sha256=file_sha256(catalog_path / "manifest.json"),
    )
    write_json(bundle_dir / "manifest.json", manifest.model_dump(mode="json"))
    _LOGGER.info(
        "e5 bundle built",
        extra={
            "index_version": manifest.index_version,
            "vectors": manifest.num_vectors,
            "dim": manifest.embedding_dim,
            "index_mb": round(index_path.stat().st_size / 1e6, 1),
        },
    )
    return manifest


def read_manifest(bundle_dir: Path | str) -> E5IndexManifest:
    path = Path(bundle_dir) / "manifest.json"
    if not path.is_file():
        raise ArtifactError(f"manifest not found: {path}")
    return E5IndexManifest.model_validate(read_json(path))


def load_product_ids(bundle_dir: Path | str) -> list[str]:
    path = Path(bundle_dir) / "product_ids.jsonl"
    if not path.is_file():
        raise ArtifactError(f"product id mapping not found: {path}")
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                ids.append(json.loads(stripped)["product_id"])
    return ids


def verify_bundle(bundle_dir: Path | str) -> E5IndexManifest:
    """Check hashes and row counts; raise :class:`ArtifactError` on mismatch."""
    directory = Path(bundle_dir)
    manifest = read_manifest(directory)
    index_path = directory / "index.faiss"
    mapping_path = directory / "product_ids.jsonl"
    if not index_path.is_file() or not mapping_path.is_file():
        raise ArtifactError(f"bundle {directory} is incomplete")
    if file_sha256(index_path) != manifest.index_sha256:
        raise ArtifactError("index.faiss hash does not match manifest")
    if file_sha256(mapping_path) != manifest.product_ids_sha256:
        raise ArtifactError("product_ids.jsonl hash does not match manifest")
    product_ids = load_product_ids(directory)
    if len(product_ids) != manifest.num_vectors:
        raise ArtifactError(
            f"id mapping ({len(product_ids)}) != num_vectors ({manifest.num_vectors})"
        )
    return manifest


def load_retriever(
    bundle_dir: Path | str,
    encoder: E5Encoder,
    *,
    top_k_per_query: int,
    rrf_k: int,
    max_queries: int,
) -> E5Retriever:
    directory = Path(bundle_dir)
    manifest = verify_bundle(directory)
    if manifest.embedding_dim != encoder.embedding_dim:
        raise ArtifactError(
            f"bundle dim {manifest.embedding_dim} != encoder dim {encoder.embedding_dim}"
        )
    index = FaissVectorIndex.load(directory / "index.faiss")
    product_ids = load_product_ids(directory)
    if index.size != manifest.num_vectors:
        raise ArtifactError(
            f"index size {index.size} != manifest num_vectors {manifest.num_vectors}"
        )
    return E5Retriever(
        encoder=encoder,
        index=index,
        product_ids=product_ids,
        manifest=manifest,
        top_k_per_query=top_k_per_query,
        rrf_k=rrf_k,
        max_queries=max_queries,
    )


def bundle_product_ids_with_texts(
    bundle_dir: Path | str, catalog_dir: Path | str
) -> dict[str, str]:
    """Convenience helper: product_id -> rerank text for the bundle's products."""
    mapping = load_product_ids(bundle_dir)
    wanted = set(mapping)
    return {
        pair.product_id: pair.rerank_text
        for pair in load_text_pairs(Path(catalog_dir) / "texts.jsonl")
        if pair.product_id in wanted
    }


def manifest_to_dict(manifest: E5IndexManifest) -> dict[str, object]:
    return manifest.model_dump(mode="json")


def ids_for_rows(ids: Sequence[int], mapping: Sequence[str]) -> list[str]:
    return [mapping[row] for row in ids]
