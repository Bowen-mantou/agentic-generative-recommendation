#!/usr/bin/env python3
"""P1: E5-only vs E5 + Qwen3-Reranker on the frozen v1.0 evaluation set.

The reranker operates on the E5 top-N candidate budget (default 10) and produces
Recall@10 / NDCG@10 / MRR plus user-cluster bootstrap intervals. E5-only is evaluated
on the full top-100 candidate list; reranked Recall@50/100 is intentionally not reported
because the reranker only sees top-N.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _metrics(rankings: list[list[str]], targets: list[str]) -> dict[str, Any]:
    def recall(k: int) -> float:
        return sum(1 for r, t in zip(rankings, targets, strict=True) if t in r[:k]) / len(rankings)

    ndcg = 0.0
    reciprocal = 0.0
    for ranked, target in zip(rankings, targets, strict=True):
        if target in ranked[:10]:
            ndcg += 1.0 / np.log2(ranked.index(target) + 2)
        rank = ranked.index(target) + 1 if target in ranked else None
        reciprocal += 1.0 / rank if rank else 0.0
    n = len(rankings)
    return {
        "samples": n,
        "recall@10": round(recall(10), 4),
        "recall@50": round(recall(50), 4),
        "recall@100": round(recall(100), 4),
        "ndcg@10": round(ndcg / n, 4),
        "mrr": round(reciprocal / n, 4),
    }


def _per_sample_hits(rankings: list[list[str]], targets: list[str], k: int) -> list[float]:
    return [
        1.0 if target in ranked[:k] else 0.0
        for ranked, target in zip(rankings, targets, strict=True)
    ]


def _bootstrap_delta(
    rankings_a: list[list[str]],
    rankings_b: list[list[str]],
    targets: list[str],
    users: list[str],
    *,
    metric: str,
    k: int,
    rounds: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    by_user: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        by_user[user].append(index)
    user_keys = list(by_user)
    rng = np.random.default_rng(seed)

    def value(ranking: list[str], target: str) -> float:
        if metric == "recall":
            return 1.0 if target in ranking[:k] else 0.0
        if metric == "ndcg":
            return 1.0 / np.log2(ranking.index(target) + 2) if target in ranking[:10] else 0.0
        if metric == "mrr":
            return 1.0 / (ranking.index(target) + 1) if target in ranking else 0.0
        raise ValueError(metric)

    deltas: list[float] = []
    for _ in range(rounds):
        sampled = rng.choice(user_keys, size=len(user_keys), replace=True)
        indexes = [index for user in sampled for index in by_user[user]]
        a = np.mean([value(rankings_a[i], targets[i]) for i in indexes])
        b = np.mean([value(rankings_b[i], targets[i]) for i in indexes])
        deltas.append(float(a - b))
    values = np.asarray(deltas)
    observed = float(
        np.mean([value(rankings_a[i], targets[i]) for i in range(len(targets))])
        - np.mean([value(rankings_b[i], targets[i]) for i in range(len(targets))])
    )
    return {
        "metric": metric,
        "k": k,
        "delta": round(observed, 4),
        "ci95": [
            round(float(np.percentile(values, 2.5)), 4),
            round(float(np.percentile(values, 97.5)), 4),
        ],
        "p_boot_gt_0": round(float((values > 0).mean()), 4),
        "clusters": len(user_keys),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True, help="frozen v1.0 dataset dir")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 = all final samples")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    import faiss

    from recommender.catalog.build import load_text_pairs
    from recommender.config import load_settings
    from recommender.reranking.qwen import Qwen3Reranker
    from recommender.retrieval.e5.encoder import E5Encoder

    settings = load_settings()
    config = settings.config
    samples = _read_jsonl(args.pilot / "samples" / "final_samples.jsonl")
    queries = {
        row["sample_id"]: row["query"]
        for row in _read_jsonl(args.pilot / "queries" / "test_queries.jsonl")
    }
    labels = {
        row["sample_id"]: row["target_product_id"]
        for row in _read_jsonl(args.pilot / "labels" / "final_target_labels.jsonl")
    }
    rows = [row for row in samples if row["sample_id"] in labels and queries.get(row["sample_id"])]
    rows.sort(key=lambda row: row["sample_id"])
    if args.limit:
        rows = rows[: args.limit]
    target = [labels[row["sample_id"]] for row in rows]
    users = [row["user_id"] for row in rows]
    print(
        f"samples={len(rows)} top_n={args.top_n} batch={args.batch_size} max_len={args.max_length}",
        flush=True,
    )

    encoder = E5Encoder(
        model_id=config.e5.model_id,
        revision=config.e5.revision,
        device=config.e5.device,
        batch_size=config.e5.batch_size,
        max_length=config.e5.max_length,
        query_prefix=config.e5.query_prefix,
        passage_prefix=config.e5.passage_prefix,
        normalize=config.e5.normalize,
        expected_dim=config.e5.embedding_dim,
    )
    index = faiss.read_index(str(args.bundle / "index.faiss"))
    product_ids = [
        json.loads(line)["product_id"]
        for line in (args.bundle / "product_ids.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    started = time.perf_counter()
    vectors = encoder.encode_queries([queries[row["sample_id"]] for row in rows])
    scores, found = index.search(np.asarray(vectors, dtype=np.float32), 100)
    e5_latency = (time.perf_counter() - started) * 1000 / max(1, len(rows))
    e5_rankings = [
        [product_ids[int(index_)] for index_ in found[i] if index_ >= 0] for i in range(len(rows))
    ]
    e5_metrics = _metrics(e5_rankings, target)
    print("E5", json.dumps(e5_metrics), flush=True)

    texts = {
        pair.product_id: pair.rerank_text for pair in load_text_pairs(args.catalog / "texts.jsonl")
    }
    reranker = Qwen3Reranker(
        device=config.reranker.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        instruction=config.reranker.instruction,
    )
    checkpoint = args.checkpoint or (args.out.parent / "reranker_checkpoint.json")
    existing: dict[str, list[str]] = {}
    if checkpoint.is_file():
        try:
            existing = json.loads(checkpoint.read_text(encoding="utf-8")).get("rankings", {})
        except (OSError, ValueError):
            existing = {}
        print(f"resume checkpoint: {len(existing)}", flush=True)

    reranked: list[list[str]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        cached = existing.get(sample_id)
        if cached is not None:
            reranked.append(cached)
            continue
        candidate_ids = e5_rankings[index - 1][: args.top_n]
        valid = [(pid, texts[pid]) for pid in candidate_ids if pid in texts]
        if not valid:
            ranking = candidate_ids
        else:
            pids = [pid for pid, _ in valid]
            docs = [doc for _, doc in valid]
            scores = reranker.score_pairs(queries[sample_id], docs)
            order = sorted(range(len(pids)), key=lambda i: (-scores[i], pids[i]))
            ranking = [pids[i] for i in order]
        reranked.append(ranking)
        existing[sample_id] = ranking
        if index % 50 == 0:
            checkpoint.write_text(
                json.dumps({"rankings": existing}, ensure_ascii=False), encoding="utf-8"
            )
            elapsed = time.perf_counter() - started
            done = sum(1 for r in reranked if r)
            print(f"reranked {index}/{len(rows)} ({done}) elapsed={elapsed:.1f}s", flush=True)
    if checkpoint.is_file():
        checkpoint.unlink()
    rerank_latency = (time.perf_counter() - started) * 1000 / max(1, len(rows))
    rerank_metrics = _metrics(reranked, target)

    deltas = [
        _bootstrap_delta(
            reranked,
            e5_rankings,
            target,
            users,
            metric=metric,
            k=10,
            rounds=args.bootstrap_rounds,
        )
        for metric in ("ndcg", "mrr", "recall")
    ]
    payload = {
        "dataset": "v1.0 final (2,224)",
        "samples": len(rows),
        "candidate_budget": args.top_n,
        "methods": {"e5_only": e5_metrics, "e5_plus_reranker": rerank_metrics},
        "bootstrap_reranker_minus_e5": deltas,
        "latency_ms_per_sample": {"e5": round(e5_latency, 1), "rerank": round(rerank_latency, 1)},
        "note": (
            f"Reranker sees E5 top-{args.top_n}; reranked Recall@50/100 is not comparable. "
            "Read it from e5_only. Reranker max_length="
            f"{args.max_length}, batch={args.batch_size}."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
