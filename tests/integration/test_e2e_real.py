"""Real E2E integration test (opt-in; requires data, models, Qdrant and DeepSeek key)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _enabled() -> bool:
    return os.environ.get("AGR_RUN_E2E") == "1"


@pytest.mark.skipif(not _enabled(), reason="set AGR_RUN_E2E=1 to run the real E2E test")
def test_real_e2e_stop_and_trace(tmp_path: Path) -> None:
    from recommender.config import load_settings
    from recommender.e2e import E2EOptions, run_e2e
    from recommender.schemas.common import ControllerAction, StopReason

    settings = load_settings()
    if not settings.has_deepseek_api_key():
        pytest.skip("DEEPSEEK_API_KEY is not configured")
    catalog = Path(os.environ.get("AGR_E2E_CATALOG", "artifacts/catalog/all_beauty_v2"))
    bundle = Path(os.environ.get("AGR_E2E_BUNDLE", "artifacts/faiss/e5-base-v2_all_beauty_v2"))
    if not (catalog / "products.jsonl").is_file() or not (bundle / "index.faiss").is_file():
        pytest.skip("real catalog/index not available")
    options = E2EOptions(
        catalog_dir=catalog,
        bundle_dir=bundle,
        trace_dir=tmp_path / "e2e",
        top_k=3,
        judgment_top_n=5,
        rerank_top_n=10,
        memory_enabled=False,
        device="cpu",
    )
    result = run_e2e(
        query="fragrance-free moisturizer for sensitive skin",
        user_id="e2e_test_user",
        options=options,
        settings=settings,
    )
    assert result.trace_path.is_file()
    assert result.summary["request_id"] == result.request_id
    assert result.state.action in (ControllerAction.STOP, ControllerAction.CLARIFY)
    assert result.state.stop_reason in (StopReason.QUALITY_SATISFIED, StopReason.RETRIEVAL_DEGRADED)
    assert result.recommendation.product_ids == result.state.selected_products
    assert all(item.reasons for item in result.recommendation.recommendations)
