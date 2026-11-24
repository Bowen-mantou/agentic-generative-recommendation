#!/usr/bin/env python3
"""Recall-stage three-way comparison for the evaluation pilot (protocol §10).

Compares TIGER-only, E5-only and Hybrid RRF on the same PASS queries/samples and target
labels. The Reranker stage is intentionally not included here; this script reports
recall-stage metrics only and labels that fact in the output.
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


def _load_pass(pilot: Path) -> tuple[list[dict], list[dict]]:
    queries = _read_jsonl(pilot / "queries" / "test_queries.jsonl")
    audits = {row["sample_id"]: row for row in _read_jsonl(pilot / "audits" / "test_audits.jsonl")}
    validation = {
        row["sample_id"]: row
        for row in _read_jsonl(pilot / "validation" / "query_validation.jsonl")
    }
    labels = {
        row["sample_id"]: row for row in _read_jsonl(pilot / "labels" / "target_labels.jsonl")
    }
    selected = [
        query
        for query in queries
        if query["status"] == "generated"
        and audits.get(query["sample_id"], {}).get("audit_status") == "PASS"
        and validation.get(query["sample_id"], {}).get("passed")
    ]
    return selected, [labels[query["sample_id"]] for query in selected]


def _metrics(rankings: list[list[str]], targets: list[str]) -> dict:
    def recall(k: int) -> float:
        return sum(
            1 for ranked, target in zip(rankings, targets, strict=True) if target in ranked[:k]
        ) / len(rankings)

    ndcg = 0.0
    rr = 0.0
    for ranked, target in zip(rankings, targets, strict=True):
        if target in ranked[:10]:
            ndcg += 1.0 / np.log2(ranked.index(target) + 2)
        rank = ranked.index(target) + 1 if target in ranked else None
        rr += 1.0 / rank if rank else 0.0
    return {
        "samples": len(rankings),
        "recall@10": round(recall(10), 4),
        "recall@50": round(recall(50), 4),
        "recall@100": round(recall(100), 4),
        "ndcg@10": round(ndcg / len(rankings), 4),
        "mrr": round(rr / len(rankings), 4),
        "coverage": round(sum(1 for ranked in rankings if ranked) / len(rankings), 4),
    }


def _rrf(rankings: list[list[str]], *, k: int = 60, top_k: int = 100) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, product_id in enumerate(ranking, start=1):
            if product_id in seen:
                continue
            seen.add(product_id)
            scores[product_id] += 1.0 / (k + rank)
    return [pid for pid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))][:top_k]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--semantic-ids", type=Path, required=True)
    parser.add_argument("--generator-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-version", default="v1.0")
    args = parser.parse_args()

    from recommender.config import load_settings
    from recommender.interaction_history import InteractionStore
    from recommender.retrieval.e5.encoder import E5Encoder
    from recommender.retrieval.e5.index import FaissVectorIndex
    from recommender.retrieval.tiger.generator import (
        generate_ids,
        load_generator,
        popularity_from_store,
        rank_products_for_ids,
    )
    from recommender.retrieval.tiger.semantic_ids import SemanticIdTable

    settings = load_settings()
    config = settings.config
    selected, labels = _load_pass(args.pilot)
    samples = {
        row["sample_id"]: row for row in _read_jsonl(args.pilot / "samples" / "test_samples.jsonl")
    }
    targets = [label["target_product_id"] for label in labels]

    table = SemanticIdTable.load(args.semantic_ids)
    model, generator_config = load_generator(args.generator_dir / "generator_best.pt")
    with InteractionStore(args.db) as store:
        popularity = popularity_from_store(store)

    tiger_rankings: list[list[str]] = []
    started = time.perf_counter()
    for query in selected:
        sample = samples[query["sample_id"]]
        history_sids = [
            table.semantic_id_of(product_id) for product_id in sample["history_product_ids"]
        ]
        history_sids = [item for item in history_sids if item is not None]
        decoded = generate_ids(model, generator_config, history_sids)
        tiger_rankings.append(
            [pid for pid, _ in rank_products_for_ids(decoded, table, popularity, top_k=args.top_k)]
        )
    tiger_latency = (time.perf_counter() - started) * 1000 / max(1, len(selected))

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
    index = FaissVectorIndex.load(args.bundle / "index.faiss")
    product_ids = [
        json.loads(line)["product_id"]
        for line in (args.bundle / "product_ids.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queries = [query["query"] for query in selected]
    started = time.perf_counter()
    vectors = encoder.encode_queries(queries)
    _, found = index.search(vectors, args.top_k)
    e5_latency = (time.perf_counter() - started) * 1000 / max(1, len(selected))
    e5_rankings = [[product_ids[int(index_)] for index_ in row if index_ >= 0] for row in found]

    hybrid_rankings = [
        _rrf([tiger, e5], top_k=args.top_k)
        for tiger, e5 in zip(tiger_rankings, e5_rankings, strict=True)
    ]

    results = {
        "tiger": _metrics(tiger_rankings, targets),
        "e5": _metrics(e5_rankings, targets),
        "hybrid": _metrics(hybrid_rankings, targets),
    }
    tiger_only = sum(
        1
        for tiger, e5, target in zip(tiger_rankings, e5_rankings, targets, strict=True)
        if target in tiger[: args.top_k] and target not in e5[: args.top_k]
    )
    e5_misses = sum(
        1 for e5, target in zip(e5_rankings, targets, strict=True) if target not in e5[: args.top_k]
    )
    rescue = (
        sum(
            1
            for tiger, e5, target in zip(tiger_rankings, e5_rankings, targets, strict=True)
            if target in tiger[: args.top_k] and target not in e5[: args.top_k]
        )
        / e5_misses
        if e5_misses
        else None
    )
    overlap = [
        len(set(tiger[: args.top_k]) & set(e5[: args.top_k])) / max(1, min(len(tiger), len(e5)))
        for tiger, e5 in zip(tiger_rankings, e5_rankings, strict=True)
    ]
    payload = {
        "dataset": f"{args.dataset_version} PASS subset",
        "note": "recall-stage only; Reranker comparison not included in this pilot run",
        "samples": len(selected),
        "methods": results,
        "complementarity": {
            "tiger_only_targets": tiger_only,
            "e5_missed_targets": e5_misses,
            "rescue_rate": round(rescue, 4) if rescue is not None else None,
            "mean_candidate_overlap": round(float(np.mean(overlap)) if overlap else 0.0, 4),
            "incremental_recall@10": round(
                results["hybrid"]["recall@10"] - results["e5"]["recall@10"], 4
            ),
            "incremental_recall@50": round(
                results["hybrid"]["recall@50"] - results["e5"]["recall@50"], 4
            ),
        },
        "latency_ms_per_sample": {
            "tiger": round(tiger_latency, 1),
            "e5": round(e5_latency, 1),
        },
        "target_leakage": "queries are synthetic PASS queries built without target labels",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
