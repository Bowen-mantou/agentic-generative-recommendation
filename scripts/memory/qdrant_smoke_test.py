#!/usr/bin/env python3
"""Functional smoke test for the AGR Memory Qdrant server (Phase 6 / D9).

Runs against the HTTP API only (standard library, no qdrant-client), so it can be used
even when the project virtualenv is unavailable.  It creates a throw-away collection,
exercises the operations the Phase 6 Mem0 adapter depends on (create collection, payload
index on ``user_id``, upsert, vector search with/without ``user_id`` filter, retrieve,
delete-by-filter, delete-by-id) and removes the collection afterwards.

Usage:
    python3 scripts/memory/qdrant_smoke_test.py
    python3 scripts/memory/qdrant_smoke_test.py --url http://127.0.0.1:6333 --keep
    python3 scripts/memory/qdrant_smoke_test.py --collection agr_d9_smoke --points 64

Exit code 0 means every assertion passed; 1 means at least one check failed (details are
printed as ``FAIL <step>: ...``).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Any

FAILURES: list[str] = []
CHECKS = 0


def request(url: str, method: str = "GET", payload: dict[str, Any] | None = None,
            timeout: float = 15.0) -> tuple[int, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            if not body:
                return resp.status, None
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body  # /healthz answers with plain text
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        return exc.code, body
    except urllib.error.URLError as exc:
        return 0, str(exc)


def check(condition: bool, step: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"PASS {step}" + (f"  [{detail}]" if detail else ""))
    else:
        print(f"FAIL {step}" + (f"  [{detail}]" if detail else ""))
        FAILURES.append(step)


def unit_vector(seed: int, dim: int) -> list[float]:
    rng = random.Random(seed)
    values = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def wait_for_health(base_url: str, timeout_s: float = 90.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, _ = request(f"{base_url}/healthz", timeout=3)
        if status == 200:
            return True
        time.sleep(1)
    return False


def search(base_url: str, collection: str, vector: list[float], limit: int,
           payload_filter: dict[str, Any] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Search through the modern /points/query API, falling back to /points/search."""
    body: dict[str, Any] = {"query": vector, "limit": limit, "with_payload": True}
    if payload_filter:
        body["filter"] = payload_filter
    status, data = request(f"{base_url}/collections/{collection}/points/query", "POST", body)
    if status == 200:
        points = data.get("result", {}).get("points", [])
        return "query", points
    body = {"vector": vector, "limit": limit, "with_payload": True}
    if payload_filter:
        body["filter"] = payload_filter
    status, data = request(f"{base_url}/collections/{collection}/points/search", "POST", body)
    if status == 200:
        return "search", data.get("result", [])
    raise RuntimeError(f"both query and search endpoints failed: {status} {data}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="agr_d9_smoke")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--keep", action="store_true", help="keep the test collection")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    col = args.collection
    dim = args.dim
    n = args.points

    print(f"Qdrant smoke test @ {base}, collection={col}, dim={dim}, points={n}")
    check(wait_for_health(base), "healthz", base)
    status, version = request(f"{base}/")
    check(status == 200 and isinstance(version, dict), "server_info",
          str(version.get("version") if isinstance(version, dict) else version))

    # Clean up a leftover collection from an aborted previous run (ours only).
    request(f"{base}/collections/{col}", "DELETE")

    status, _ = request(f"{base}/collections/{col}", "PUT",
                        {"vectors": {"size": dim, "distance": "Cosine"}})
    check(status == 200, "create_collection", f"dim={dim} distance=Cosine")

    status, payload = request(f"{base}/collections/{col}/index", "PUT",
                              {"field_name": "user_id", "field_schema": "keyword"})
    check(status == 200, "create_payload_index_user_id", str(payload)[:120])

    points = []
    for i in range(1, n + 1):
        user = "u001" if i % 2 else "u002"
        points.append({
            "id": i,
            "vector": unit_vector(i, dim),
            "payload": {"user_id": user, "text": f"示例记忆 {i} / memory {i}",
                        "seq": i},
        })
    status, _ = request(f"{base}/collections/{col}/points?wait=true", "PUT", {"points": points})
    check(status == 200, "upsert_points", f"{n} points")

    status, info = request(f"{base}/collections/{col}")
    count = info.get("result", {}).get("points_count") if status == 200 else None
    check(count == n, "count_after_upsert", f"points_count={count}")

    # Nearest neighbour of point 1 must be point 1 itself with score ~1.0.
    api, hits = search(base, col, unit_vector(1, dim), 3)
    top = hits[0] if hits else {}
    check(top.get("id") == 1 and abs(float(top.get("score", 0)) - 1.0) < 1e-3,
          "search_self_hit", f"api={api} top_id={top.get('id')} score={top.get('score')}")

    filtered = {"must": [{"key": "user_id", "match": {"value": "u001"}}]}
    _, hits = search(base, col, unit_vector(3, dim), 5, filtered)
    check(len(hits) > 0 and all(h["payload"]["user_id"] == "u001" for h in hits),
          "search_with_user_id_filter", f"hits={len(hits)}")

    status, point = request(f"{base}/collections/{col}/points/1")
    check(status == 200 and point.get("result", {}).get("payload", {}).get("user_id") == "u001",
          "retrieve_point_by_id", str(point.get("result", {}).get("payload"))[:100])

    status, _ = request(f"{base}/collections/{col}/points/delete?wait=true", "POST",
                        {"filter": filtered})
    check(status == 200, "delete_points_by_filter")
    status, info = request(f"{base}/collections/{col}")
    count = info.get("result", {}).get("points_count") if status == 200 else None
    check(count == n // 2, "count_after_filter_delete", f"points_count={count}")

    status, _ = request(f"{base}/collections/{col}/points/delete?wait=true", "POST",
                        {"points": [2]})
    check(status == 200, "delete_point_by_id")
    _, hits = search(base, col, unit_vector(1, dim), 5, filtered)
    check(len(hits) == 0, "deleted_user_returns_no_hits", f"hits={len(hits)}")

    status, info = request(f"{base}/collections/{col}")
    vector_cfg = info.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
    check(vector_cfg.get("size") == dim and vector_cfg.get("distance") == "Cosine",
          "collection_vector_config", str(vector_cfg))

    if not args.keep:
        status, _ = request(f"{base}/collections/{col}", "DELETE")
        check(status == 200, "delete_collection")

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED: " + ", ".join(FAILURES))
        return 1
    print("SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
