#!/usr/bin/env python3
"""D11 gate: TIGER vs popularity vs E5-profile vs hybrid on one fixed test split.

Computes per-sample rankings for four methods on the same samples and reports
Recall@K / NDCG@10 / MRR plus paired-bootstrap confidence intervals for the D11
conditions:

* TIGER must beat the popularity baseline (Recall@10 and Recall@50);
* hybrid (RRF of TIGER + E5-profile) must improve over E5-only without a clear NDCG drop.

This is an offline next-item evaluation on Amazon review/rating proxy events; it is not a
purchase-conversion measurement.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_samples(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def metrics(rankings: list[list[str]], targets: list[str], ks: tuple[int, ...]) -> dict:
    per_sample: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
    ndcg: list[float] = []
    reciprocal_ranks: list[float] = []
    for ranked, target in zip(rankings, targets, strict=True):
        for k in ks:
            per_sample[f"recall@{k}"].append(1.0 if target in ranked[:k] else 0.0)
        ndcg.append(
            1.0 / np.log2(ranked.index(target) + 2) if target in ranked[:10] else 0.0
        )
        rank = ranked.index(target) + 1 if target in ranked else None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    count = len(rankings)
    return {
        "samples": count,
        **{key: float(np.mean(values)) for key, values in per_sample.items()},
        "ndcg@10": float(np.mean(ndcg)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "_per_sample": {**per_sample, "ndcg@10": ndcg, "mrr": reciprocal_ranks},
    }


def bootstrap_delta(
    a: list[float], b: list[float], *, rounds: int = 1000, seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    count = len(a_arr)
    deltas = []
    for _ in range(rounds):
        index = rng.integers(0, count, size=count)
        deltas.append(float((a_arr[index] - b_arr[index]).mean()))
    deltas_arr = np.asarray(deltas)
    return {
        "delta": float(a_arr.mean() - b_arr.mean()),
        "ci95": [
            float(np.percentile(deltas_arr, 2.5)),
            float(np.percentile(deltas_arr, 97.5)),
        ],
        "p_boot_gt_0": float((deltas_arr > 0).mean()),
    }


def rrf_fuse(rankings: list[list[str]], *, k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, product_id in enumerate(ranking, start=1):
            if product_id in seen:
                continue
            seen.add(product_id)
            scores[product_id] += 1.0 / (k + rank)
    return [
        product_id
        for product_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-dir", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    args = parser.parse_args()

    import faiss

    from recommender.interaction_history import InteractionStore
    from recommender.retrieval.tiger.generator import (
        generate_ids,
        load_generator,
        popularity_from_store,
        rank_products_for_ids,
    )
    from recommender.retrieval.tiger.semantic_ids import SemanticIdTable

    rows = load_samples(args.samples_dir / f"{args.split}.jsonl")
    checkpoint = args.generator_dir / "generator_best.pt"
    if not checkpoint.is_file():
        checkpoint = args.generator_dir / "generator.pt"
    model, config = load_generator(checkpoint)
    table = SemanticIdTable.load(args.semantic_ids)
    with InteractionStore(args.db) as store:
        popularity = popularity_from_store(store)
    connection = sqlite3.connect(args.db)
    # Keep timestamps so the E5 profile only sees events strictly before the target
    # (no target leakage).
    history: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        user_id = row["user_id"]
        if user_id not in history:
            history[user_id] = [
                (int(record[0]), str(record[1]))
                for record in connection.execute(
                    "SELECT timestamp, product_id FROM interaction_events "
                    "WHERE user_id=? ORDER BY timestamp ASC",
                    (user_id,),
                )
            ]
    connection.close()

    product_ids = [
        json.loads(line)["product_id"]
        for line in (args.bundle / "product_ids.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    row_of = {product_id: index for index, product_id in enumerate(product_ids)}
    vectors = np.load(args.bundle / "embeddings.npy", mmap_mode="r")
    index = faiss.read_index(str(args.bundle / "index.faiss"))

    top_popularity = [
        product_id
        for product_id, _ in sorted(popularity.items(), key=lambda kv: (-kv[1], kv[0]))
    ][: args.top_k]
    targets = [row["target_product_id"] for row in rows]

    tiger_rankings: list[list[str]] = []
    e5_rankings: list[list[str]] = []
    for row in rows:
        decoded = generate_ids(model, config, row["history_ids"])
        tiger_rankings.append(
            [
                product_id
                for product_id, _ in rank_products_for_ids(
                    decoded, table, popularity, top_k=args.top_k
                )
            ]
        )
        user_history = [
            pid
            for timestamp, pid in history.get(row["user_id"], [])
            if timestamp < row["timestamp"] and pid in row_of
        ]
        if not user_history:
            e5_rankings.append([])
            continue
        profile = vectors[[row_of[pid] for pid in user_history]].mean(axis=0, dtype=np.float32)
        profile /= np.linalg.norm(profile) + 1e-9
        _, found = index.search(np.asarray([profile], dtype=np.float32), args.top_k)
        e5_rankings.append([product_ids[int(i)] for i in found[0] if i >= 0])

    hybrid_rankings = [
        rrf_fuse([tiger, e5])
        for tiger, e5 in zip(tiger_rankings, e5_rankings, strict=True)
    ]
    popularity_rankings = [top_popularity for _ in rows]

    results = {
        "tiger": metrics(tiger_rankings, targets, (10, 50)),
        "popularity": metrics(popularity_rankings, targets, (10, 50)),
        "e5_profile": metrics(e5_rankings, targets, (10, 50)),
        "hybrid_rrf": metrics(hybrid_rankings, targets, (10, 50)),
    }
    condition_one = {
        f"recall@{k}": bootstrap_delta(
            results["tiger"]["_per_sample"][f"recall@{k}"],
            results["popularity"]["_per_sample"][f"recall@{k}"],
            rounds=args.bootstrap_rounds,
        )
        for k in (10, 50)
    }
    condition_two = {
        "recall@10": bootstrap_delta(
            results["hybrid_rrf"]["_per_sample"]["recall@10"],
            results["e5_profile"]["_per_sample"]["recall@10"],
            rounds=args.bootstrap_rounds,
        ),
        "ndcg@10": bootstrap_delta(
            results["hybrid_rrf"]["_per_sample"]["ndcg@10"],
            results["e5_profile"]["_per_sample"]["ndcg@10"],
            rounds=args.bootstrap_rounds,
        ),
    }
    payload = {
        "split": args.split,
        "generator_dir": str(args.generator_dir),
        "samples": len(rows),
        "methods": {
            name: {key: value for key, value in result.items() if key != "_per_sample"}
            for name, result in results.items()
        },
        "d11_condition_one_tiger_minus_popularity": condition_one,
        "d11_condition_two_hybrid_minus_e5": condition_two,
        "caveat": "offline next-item proxy evaluation; not purchase conversion",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
