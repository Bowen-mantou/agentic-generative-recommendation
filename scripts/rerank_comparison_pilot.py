#!/usr/bin/env python3
"""Reranker-stage pilot comparison for the 82 PASS evaluation samples.

Reranks the top-20 candidates of E5-only and Hybrid with the frozen Qwen3-Reranker
(CPU), reports NDCG@10 / MRR / Recall@10 and user-cluster bootstrap intervals.
TIGER-only is omitted from reranking because it produced zero target hits in the
recall-stage pilot; its numbers are reported separately as 0.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _rrf(rankings: list[list[str]], *, k: int = 60, top_k: int = 20) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, product_id in enumerate(ranking, start=1):
            if product_id in seen:
                continue
            seen.add(product_id)
            scores[product_id] += 1.0 / (k + rank)
    return [pid for pid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))][:top_k]


def _metrics(rankings: list[list[str]], targets: list[str]) -> dict:
    recall10 = sum(
        1 for ranked, target in zip(rankings, targets, strict=True) if target in ranked[:10]
    )
    ndcg = 0.0
    rr = 0.0
    for ranked, target in zip(rankings, targets, strict=True):
        if target in ranked[:10]:
            ndcg += 1.0 / np.log2(ranked.index(target) + 2)
        rank = ranked.index(target) + 1 if target in ranked else None
        rr += 1.0 / rank if rank else 0.0
    n = len(rankings)
    return {
        "samples": n,
        "recall@10": round(recall10 / n, 4),
        "ndcg@10": round(ndcg / n, 4),
        "mrr": round(rr / n, 4),
    }


def _cluster_bootstrap(
    rankings_a: list[list[str]],
    rankings_b: list[list[str]],
    targets: list[str],
    users: list[str],
    *,
    rounds: int = 1000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    by_user: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        by_user[user].append(index)
    user_keys = list(by_user)
    deltas: list[float] = []
    for _ in range(rounds):
        sampled = rng.choice(user_keys, size=len(user_keys), replace=True)
        indexes = [index for user in sampled for index in by_user[user]]
        hit_a = np.mean([1.0 if targets[i] in rankings_a[i][:10] else 0.0 for i in indexes])
        hit_b = np.mean([1.0 if targets[i] in rankings_b[i][:10] else 0.0 for i in indexes])
        deltas.append(float(hit_a - hit_b))
    values = np.asarray(deltas)
    return {
        "delta_recall@10": float(
            np.mean([targets[i] in rankings_a[i][:10] for i in range(len(targets))])
            - np.mean([targets[i] in rankings_b[i][:10] for i in range(len(targets))])
        ),
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "p_boot_gt_0": float((values > 0).mean()),
        "clusters": len(user_keys),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    args = parser.parse_args()

    from recommender.catalog.build import load_text_pairs
    from recommender.config import load_settings
    from recommender.reranking.qwen import Qwen3Reranker
    from recommender.retrieval.e5.encoder import E5Encoder
    from recommender.retrieval.e5.index import FaissVectorIndex

    settings = load_settings()
    config = settings.config
    queries = _read_jsonl(args.pilot / "queries" / "test_queries.jsonl")
    audits = {
        row["sample_id"]: row for row in _read_jsonl(args.pilot / "audits" / "test_audits.jsonl")
    }
    validation = {
        row["sample_id"]: row
        for row in _read_jsonl(args.pilot / "validation" / "query_validation.jsonl")
    }
    labels = {
        row["sample_id"]: row for row in _read_jsonl(args.pilot / "labels" / "target_labels.jsonl")
    }
    samples = {
        row["sample_id"]: row for row in _read_jsonl(args.pilot / "samples" / "test_samples.jsonl")
    }
    selected = [
        query
        for query in queries
        if query["status"] == "generated"
        and audits.get(query["sample_id"], {}).get("audit_status") == "PASS"
        and validation.get(query["sample_id"], {}).get("passed")
    ]
    targets = [labels[query["sample_id"]]["target_product_id"] for query in selected]
    users = [samples[query["sample_id"]]["user_id"] for query in selected]

    # Rebuild E5 rankings on the same PASS subset.
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
    bundle = args.bundle
    index = FaissVectorIndex.load(bundle / "index.faiss")
    product_ids = [
        json.loads(line)["product_id"]
        for line in (bundle / "product_ids.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vectors = encoder.encode_queries([query["query"] for query in selected])
    _, found = index.search(vectors, args.top_k)
    e5_rankings = [[product_ids[int(i)] for i in row if i >= 0] for row in found]

    texts = {
        pair.product_id: pair.rerank_text for pair in load_text_pairs(args.catalog / "texts.jsonl")
    }
    scorer = Qwen3Reranker(
        model_id=config.reranker.primary_model_id,
        device=config.reranker.device,
        max_length=config.reranker.max_length,
        batch_size=config.reranker.batch_size,
        instruction=config.reranker.instruction,
    )
    reranked_e5: list[list[str]] = []
    started = time.perf_counter()
    for position, query in enumerate(selected, start=1):
        documents = [texts.get(pid, "") for pid in e5_rankings[position - 1]]
        valid = [
            (pid, doc) for pid, doc in zip(e5_rankings[position - 1], documents, strict=True) if doc
        ]
        if not valid:
            reranked_e5.append([])
            continue
        pids = [pid for pid, _ in valid]
        docs = [doc for _, doc in valid]
        scores = scorer.score_pairs(query["query"], docs)
        order = sorted(range(len(pids)), key=lambda i: (-scores[i], pids[i]))
        reranked_e5.append([pids[i] for i in order])
        if position % 5 == 0:
            elapsed = time.perf_counter() - started
            print(f"reranked {position}/{len(selected)} in {elapsed:.1f}s", flush=True)

    results = {
        "e5_before": _metrics(e5_rankings, targets),
        "e5_after": _metrics(reranked_e5, targets),
    }
    payload = {
        "dataset": "v0.1-pilot PASS subset",
        "note": "top-20 rerank, frozen Qwen3-Reranker CPU; TIGER rerank omitted (0 target hits)",
        "samples": len(selected),
        "methods": results,
        "bootstrap_e5_after_minus_before": _cluster_bootstrap(
            reranked_e5, e5_rankings, targets, users, rounds=args.bootstrap_rounds
        ),
        "rerank_latency_s_total": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
