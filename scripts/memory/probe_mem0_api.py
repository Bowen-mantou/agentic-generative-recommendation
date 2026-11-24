#!/usr/bin/env python3
"""Probe the mem0ai / qdrant-client interfaces the Phase 6 Memory adapter depends on.

Read-only w.r.t. the project repository. It records:
  * installed versions;
  * the exact ``Memory`` / ``Qdrant`` / ``EmbeddingBase`` signatures used by Phase 6;
  * the configuration model fields (embedder, vector store, LLM);
  * the provider registries (how a custom E5 embedder can be plugged in);
  * a live round-trip against the running Qdrant server using mem0's own Qdrant vector
    store class: collection creation (dense + BM25 sparse slot), insert, filtered search
    by ``user_id``, get, list, payload update, delete, cleanup.

The live section uses fake 768-dim vectors on purpose: it validates the *storage and
filter contract*, not embedding quality (that is ``bench_e5.py``/D9 report).

Usage:
    PYTHONPATH=/tmp/agr_d9/mem0site python3 scripts/memory/probe_mem0_api.py \
        --url http://127.0.0.1:6333 --json /tmp/mem0_probe.json
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import inspect
import json
import math
import random
import sys
import uuid
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def signature_of(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return "<unavailable>"


def dump_fields(model: Any) -> dict[str, str]:
    fields = getattr(model, "model_fields", {})
    return {name: str(field.annotation) for name, field in fields.items()}


def probe() -> dict[str, Any]:
    import torch  # noqa: F401  (version only)
    import transformers

    import mem0
    from mem0 import Memory
    from mem0.configs.base import MemoryConfig
    from mem0.embeddings.base import EmbeddingBase
    from mem0.embeddings.configs import EmbedderConfig
    from mem0.utils.factory import EmbedderFactory, VectorStoreFactory
    from mem0.vector_stores.configs import VectorStoreConfig
    from mem0.configs.vector_stores.qdrant import QdrantConfig
    from mem0.vector_stores.qdrant import Qdrant

    result: dict[str, Any] = {
        "versions": {
            "mem0ai": package_version("mem0ai"),
            "qdrant-client": package_version("qdrant-client"),
            "pydantic": package_version("pydantic"),
            "openai": package_version("openai"),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "mem0_module": getattr(mem0, "__version__", "<none>"),
        },
        "memory_methods": {
            "from_config": signature_of(Memory.from_config),
            "add": signature_of(Memory.add),
            "search": signature_of(Memory.search),
            "get": signature_of(Memory.get),
            "get_all": signature_of(Memory.get_all),
            "update": signature_of(Memory.update),
            "delete": signature_of(Memory.delete),
            "delete_all": signature_of(Memory.delete_all),
            "history": signature_of(Memory.history),
            "reset": signature_of(Memory.reset),
        },
        "embedding_base": {
            "embed": signature_of(EmbeddingBase.embed),
            "embed_batch": signature_of(EmbeddingBase.embed_batch),
            "providers": sorted(EmbedderFactory.provider_to_class),
        },
        "vector_store_providers": sorted(
            VectorStoreFactory.provider_to_class
        ),
        "config_fields": {
            "MemoryConfig": dump_fields(MemoryConfig),
            "EmbedderConfig": dump_fields(EmbedderConfig),
            "VectorStoreConfig_": dump_fields(VectorStoreConfig),
            "QdrantConfig": dump_fields(QdrantConfig),
        },
        "qdrant_store_methods": {
            "init": signature_of(Qdrant.__init__),
            "create_col": signature_of(Qdrant.create_col),
            "insert": signature_of(Qdrant.insert),
            "search": signature_of(Qdrant.search),
            "update": signature_of(Qdrant.update),
            "delete": signature_of(Qdrant.delete),
            "get": signature_of(Qdrant.get),
            "list": signature_of(Qdrant.list),
            "delete_col": signature_of(Qdrant.delete_col),
            "reset": signature_of(Qdrant.reset),
        },
    }
    return result


def live_roundtrip(url: str, collection: str, dim: int) -> dict[str, Any]:
    from qdrant_client import QdrantClient

    from mem0.vector_stores.qdrant import Qdrant

    parsed = url.split("://", 1)[-1]
    host, _, port_text = parsed.partition(":")
    client = QdrantClient(host=host, port=int(port_text or 6333), timeout=20)
    store = Qdrant(
        client=client,
        collection_name=collection,
        embedding_model_dims=dim,
        on_disk=False,
    )
    try:
        def unit(seed: int) -> list[float]:
            rng = random.Random(seed)
            values = [rng.uniform(-1, 1) for _ in range(dim)]
            norm = math.sqrt(sum(v * v for v in values))
            return [v / norm for v in values]

        ids = [str(uuid.uuid4()) for _ in range(6)]
        users = ["u001", "u001", "u001", "u002", "u002", "u002"]
        payloads = [
            {
                "data": f"memory text {i}",
                "text_lemmatized": f"memory text {i}",
                "hash": uuid.uuid5(uuid.NAMESPACE_DNS, f"memory {i}").hex,
                "user_id": users[i],
                "agent_id": None,
                "run_id": None,
                "created_at": "2026-11-24T08:00:00+00:00",
                "updated_at": "2026-11-24T08:00:00+00:00",
            }
            for i in range(6)
        ]
        store.insert(vectors=[unit(i) for i in range(6)], payloads=payloads, ids=ids)

        hits = store.search("probe", unit(0), top_k=5, filters={"user_id": "u001"})
        hit_payloads = [h.payload for h in hits]
        top = hit_payloads[0] if hit_payloads else {}
        isolation = all(p.get("user_id") == "u001" for p in hit_payloads) and len(hits) == 3

        got = store.get(ids[0])
        listed = store.list(filters={"user_id": "u002"}, top_k=10)

        store.update(vector_id=ids[0], payload={"data": "updated text"})
        updated = store.get(ids[0])

        store.delete(ids[5])
        after_delete = store.list(filters={"user_id": "u002"}, top_k=10)

        info = store.col_info()
        sparse = getattr(info.config.params, "sparse_vectors", None)
        payload_schema = list((info.payload_schema or {}).keys())

        return {
            "collection": collection,
            "dim": dim,
            "inserted": len(ids),
            "search_filter_u001": {
                "hits": len(hits),
                "isolation_ok": isolation,
                "top_payload": top,
                "score": getattr(hits[0], "score", None) if hits else None,
            },
            "get_ok": bool(got and got.payload.get("data") == "memory text 0"),
            "list_u002": len(listed[0]) if listed else 0,
            "update_ok": bool(updated and updated.payload.get("data") == "updated text"),
            "delete_ok": len(after_delete[0]) if after_delete else 0,
            "collection_sparse_vectors": sorted(sparse.keys()) if sparse else [],
            "payload_indexes": sorted(payload_schema),
            "checks": {
                "collection_has_bm25_slot": bool(sparse and "bm25" in sparse),
                "user_id_indexed": "user_id" in payload_schema,
                "search_isolation": isolation,
                "update_visible": bool(updated and updated.payload.get("data") == "updated text"),
                "delete_removed": (len(after_delete[0]) if after_delete else 0) == 2,
            },
        }
    finally:
        try:
            store.delete_col()
        except Exception:  # pragma: no cover - cleanup best effort
            pass
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="agr_d9_mem0_probe")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    report = probe()
    report["live"] = (
        {"skipped": True} if args.skip_live else live_roundtrip(args.url, args.collection, args.dim)
    )
    if not args.skip_live:
        report["ok"] = all(report["live"]["checks"].values())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    return 0 if report.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
