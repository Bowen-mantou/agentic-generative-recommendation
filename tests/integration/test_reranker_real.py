"""Real Qwen3-Reranker-0.6B smoke test (opt-in: AGR_RUN_RERANKER=1).

Loads the real model on CPU (Q2) and reranks a few hand-written documents. Hand-written
texts are smoke-test only and must not enter evaluation datasets.
"""

from __future__ import annotations

import os
import time

import pytest

RUN = os.environ.get("AGR_RUN_RERANKER") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RUN, reason="set AGR_RUN_RERANKER=1 to run"),
]


def test_qwen3_reranker_scores_and_orders() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from recommender.config import load_settings
    from recommender.reranking import Qwen3Reranker

    config = load_settings().config
    reranker = Qwen3Reranker(
        model_id=config.reranker.primary_model_id,
        device="cpu",
        max_length=512,
        batch_size=4,
        instruction=config.reranker.instruction,
    )
    query = "moisturizing face cream for dry sensitive skin"
    documents = [
        "Daily Face Moisturizer Serum - Hydrating Vitamin Lotion with Aloe Vera and Shea Butter for Sensitive Dry Skin",
        "Maybelline Lash Sensational Mascara, Very Black",
        "Dyson V8 cordless vacuum cleaner",
        "Sony WH-1000XM5 wireless headphones",
    ]
    started = time.perf_counter()
    scores = reranker.score_pairs(query, documents)
    elapsed = time.perf_counter() - started

    assert len(scores) == len(documents)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores[0] == max(scores), f"relevant document was not ranked first: {scores}"
    assert scores[0] > 0.5, f"relevant score unexpectedly low: {scores[0]}"
    print(f"reranker smoke scores={scores} elapsed={elapsed:.2f}s")
