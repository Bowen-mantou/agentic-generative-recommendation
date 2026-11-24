#!/usr/bin/env python3
"""Benchmark ``intfloat/multilingual-e5-base`` for the AGR Memory embedder candidate.

The script mirrors the loader used by ``src/recommender/retrieval/e5/encoder.py``
(Sentence-Transformers + explicit ``query: `` / ``passage: `` prefixes) and records the
numbers D9 needs: embedding dimension, normalisation, Chinese/English encode latency,
batch throughput and process memory.

It is intentionally read-only: it loads the model from the local HF cache (no download)
and never writes into the shared project virtualenv.

Usage:
    HF_HOME=/mnt/agr-data/hf HF_HUB_OFFLINE=1 \
        python3 scripts/memory/bench_e5.py --json /tmp/e5_bench.json

Exit code 0 when all assertions pass (dimension 768, unit norm, deterministic output),
1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

QUERIES = {
    "zh": "推荐一款适合敏感肌的无香精保湿面霜",
    "en": "a fragrance-free moisturizer for sensitive skin",
}
PASSAGES = {
    "zh_match": "这款保湿面霜不含香精，配方温和，适合敏感肌肤使用。",
    "en_match": "This fragrance-free moisturizer is gentle and suitable for sensitive skin.",
    "zh_other": "机械键盘 87 键 青轴 有线 游戏电竞",
    "en_other": "A 27-inch 4K monitor with USB-C hub and height adjustment",
}
EXPECTED_DIM = 768


def rss_mb() -> float:
    """Resident set size of this process in MiB (Linux)."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def peak_rss_mb() -> float:
    return resource_maxrss_kb() / 1024.0


def resource_maxrss_kb() -> float:
    import resource

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def l2_norm(vector: list[float]) -> float:
    return sum(x * x for x in vector) ** 0.5


def latency_stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)

    def pct(p: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return ordered[index]

    return {
        "runs": len(samples),
        "min_ms": round(min(samples) * 1000, 2),
        "median_ms": round(statistics.median(samples) * 1000, 2),
        "p95_ms": round(pct(0.95) * 1000, 2),
        "max_ms": round(max(samples) * 1000, 2),
    }


def main() -> int:
    start_import = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=20, help="single-query latency runs")
    parser.add_argument("--batch-repeat", type=int, default=5)
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--threads", type=int, default=0,
                        help="torch intra-op threads; 0 keeps the library default")
    parser.add_argument("--skip-prefix-comparison", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"FATAL embedding stack unavailable: {exc}", file=sys.stderr)
        return 2
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    import_time = time.perf_counter() - start_import

    print(f"model={args.model} revision={args.revision} device={args.device}")
    print(f"torch={torch.__version__} numpy={np.__version__} torch_threads={torch.get_num_threads()}")

    load_start = time.perf_counter()
    model = SentenceTransformer(args.model, revision=args.revision, device=args.device)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = args.max_length
    load_s = time.perf_counter() - load_start
    rss_after_load = rss_mb()
    dimension_getter = (
        getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    )
    dim = int(dimension_getter())
    print(f"load_time_s={load_s:.2f} dim={dim} rss_after_load_mb={rss_after_load:.0f}")
    if dim != EXPECTED_DIM:
        print(f"FAIL dimension {dim} != {EXPECTED_DIM}", file=sys.stderr)
        return 1

    def encode(texts: list[str], prefix: str) -> list[list[float]]:
        prepared = [f"{prefix}{text}" if prefix else text for text in texts]
        vectors = model.encode(
            prepared,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    # Warm-up (excluded from latency numbers): page-in weights, allocate buffers.
    encode(list(QUERIES.values()), args.query_prefix)
    encode(list(PASSAGES.values()), args.passage_prefix)

    results: dict[str, Any] = {
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "dim": dim,
        "max_seq_length": args.max_length,
        "query_prefix": args.query_prefix,
        "passage_prefix": args.passage_prefix,
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__,
            "transformers": __import__("transformers").__version__,
        },
        "load_time_s": round(load_s, 2),
        "import_time_s": round(import_time, 2),
        "torch_threads": torch.get_num_threads(),
        "rss_after_load_mb": round(rss_after_load, 1),
    }

    # -- encode quality sanity checks ---------------------------------------
    zh_vec = encode([QUERIES["zh"]], args.query_prefix)[0]
    en_vec = encode([QUERIES["en"]], args.query_prefix)[0]
    passage_vecs = {name: encode([text], args.passage_prefix)[0] for name, text in PASSAGES.items()}
    zh_again = encode([QUERIES["zh"]], args.query_prefix)[0]

    checks = {
        "dim_768": dim == EXPECTED_DIM,
        "unit_norm_query": abs(l2_norm(zh_vec) - 1.0) < 1e-3,
        "unit_norm_passage": abs(l2_norm(passage_vecs["zh_match"]) - 1.0) < 1e-3,
        "deterministic": all(abs(a - b) < 1e-6 for a, b in zip(zh_vec, zh_again)),
    }
    cosines = {
        f"zh_query~{name}": round(cosine(zh_vec, vec), 4) for name, vec in passage_vecs.items()
    }
    cosines.update({
        f"en_query~{name}": round(cosine(en_vec, vec), 4) for name, vec in passage_vecs.items()
    })
    checks["zh_match_ranked_above_other"] = (
        cosines["zh_query~zh_match"] > cosines["zh_query~zh_other"]
    )
    checks["zh_query_close_to_en_passage"] = (
        cosines["zh_query~en_match"] > cosines["zh_query~en_other"]
    )
    results["cosine"] = cosines

    # -- latency: single Chinese query --------------------------------------
    timings: list[float] = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        encode([QUERIES["zh"]], args.query_prefix)
        timings.append(time.perf_counter() - started)
    results["single_query_latency"] = latency_stats(timings)

    timings_en: list[float] = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        encode([QUERIES["en"]], args.query_prefix)
        timings_en.append(time.perf_counter() - started)
    results["single_query_latency_en"] = latency_stats(timings_en)

    # -- throughput: batch of 32 passages -----------------------------------
    batch_texts = [PASSAGES["zh_match"]] * args.batch_size
    batch_times: list[float] = []
    for _ in range(args.batch_repeat):
        started = time.perf_counter()
        encode(batch_texts, args.passage_prefix)
        batch_times.append(time.perf_counter() - started)
    stats = latency_stats(batch_times)
    stats["texts_per_s"] = round(args.batch_size / (stats["median_ms"] / 1000.0), 1)
    results["batch_latency"] = stats

    # -- prefix importance (documented for the Phase 6 adapter) -------------
    if not args.skip_prefix_comparison:
        no_prefix_query = encode([QUERIES["zh"]], "")[0]
        no_prefix_passage = encode([PASSAGES["zh_match"]], "")[0]
        results["prefix_comparison"] = {
            "cosine_with_prefix": round(cosine(zh_vec, passage_vecs["zh_match"]), 4),
            "cosine_without_prefix": round(cosine(no_prefix_query, no_prefix_passage), 4),
        }

    results["rss_after_bench_mb"] = round(rss_mb(), 1)
    results["peak_rss_mb"] = round(peak_rss_mb(), 1)
    results["checks"] = checks
    results["ok"] = all(checks.values())

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_path}")
    if not results["ok"]:
        print("FAILED checks: " + ", ".join(k for k, v in checks.items() if not v), file=sys.stderr)
        return 1
    print("E5 BENCH OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
