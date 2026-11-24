"""D6 Judgment Evidence Builder tests (Catalog lookup, truncation, budget, no inference).

Covers: description truncation, relevance filtering of details (case-insensitive,
key- and value-based), missing products, empty description/details, token and character
budgets, product_id alignment/deduplication, read-only constraint status, and the
additive ``Judge.judge_ranked`` entry point with a stub DeepSeek client (no real API).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from recommender.api.deepseek import DeepSeekClient
from recommender.judgment import (
    Judge,
    JudgmentEvidenceBuilder,
)
from recommender.schemas.common import ConstraintStatus
from recommender.schemas.plan import RecommendationPlan
from recommender.schemas.product import ProductRecord
from recommender.utils.errors import ContractError

pytestmark = pytest.mark.unit

Handler = Callable[[httpx.Request], httpx.Response]


def _plan(
    *,
    hard: list[dict[str, Any]] | None = None,
    soft: list[dict[str, Any]] | None = None,
    current: list[str] | None = None,
    historical: list[str] | None = None,
    negative: list[str] | None = None,
    priority: list[str] | None = None,
    priority_attributes: list[str] | None = None,
) -> RecommendationPlan:
    return RecommendationPlan.model_validate(
        {
            "schema_version": "1.0",
            "plan_id": "plan_evidence",
            "parent_plan_id": None,
            "plan_version": 1,
            "mode": "initial",
            "intent": {
                "intent": "product_recommendation",
                "category": "Beauty",
                "product_type": "moisturizer",
                "usage_scenario": "daily_use",
            },
            "constraints": {
                "hard": (
                    hard
                    if hard is not None
                    else [
                        {
                            "field": "price",
                            "operator": "lte",
                            "value": 30,
                            "unit": "USD",
                            "source": "current_request",
                        }
                    ]
                ),
                "soft": (
                    soft
                    if soft is not None
                    else [
                        {
                            "attribute": "lightweight",
                            "source": "current_request",
                            "priority": "high",
                        }
                    ]
                ),
            },
            "preferences": {
                "current": current if current is not None else ["lightweight"],
                "historical": (
                    historical if historical is not None else ["fragrance_free"]
                ),
                "negative": (
                    negative if negative is not None else ["strong_fragrance"]
                ),
                "priority": priority if priority is not None else ["price", "lightweight"],
            },
            "retrieval": {
                "strategy": "semantic_priority",
                "tiger": {"enabled": False, "candidate_budget": "standard"},
                "e5_faiss": {
                    "enabled": True,
                    "queries": ["daily moisturizer under 30"],
                    "candidate_budget": "standard",
                },
            },
            "ranking": {
                "rerank_query": "daily moisturizer under $30",
                "priority_attributes": (
                    priority_attributes
                    if priority_attributes is not None
                    else ["price", "lightweight"]
                ),
            },
            "clarification": {"required": False, "question": None, "reason": None},
            "assumptions": [],
            "change_summary": [],
        }
    )


def _empty_signal_plan() -> RecommendationPlan:
    return _plan(
        hard=[],
        soft=[],
        current=[],
        historical=[],
        negative=[],
        priority=[],
        priority_attributes=[],
    )


def _product(
    product_id: str = "p1",
    *,
    title: str = "Daily Moisturizing Cream",
    price: float | None = 25.99,
    description: str | None = "Lightweight daily moisturizing cream",
    details: dict[str, str] | None = None,
    brand: str | None = "Acme",
    main_category: str | None = "Beauty",
) -> ProductRecord:
    return ProductRecord(
        product_id=product_id,
        parent_asin=product_id,
        title=title,
        brand=brand,
        main_category=main_category,
        price=price,
        description=description,
        details=details if details is not None else {},
    )


def _detail_keys(candidate_attributes: dict[str, str]) -> set[str]:
    return set(candidate_attributes) - {"brand", "main_category"}


# -- description -------------------------------------------------------------


def test_description_is_truncated_to_limit_and_recorded() -> None:
    long_description = "A" * 50 + " " + "B" * 50
    builder = JudgmentEvidenceBuilder(
        catalog={"p1": _product(description=long_description)},
        description_max_chars=40,
    )
    result = builder.build(["p1"], plan=_plan())

    candidate = result.candidates[0]
    assert candidate.description is not None
    assert len(candidate.description) <= 40
    assert candidate.description.endswith("…")
    assert candidate.description.startswith("A" * 39)
    assert result.stats.truncated_fields == {"p1": ["description"]}


def test_short_description_is_kept_verbatim() -> None:
    builder = JudgmentEvidenceBuilder(catalog={"p1": _product(description="Soft cream")})

    result = builder.build(["p1"], plan=_plan())

    assert result.candidates[0].description == "Soft cream"
    assert result.stats.truncated_fields == {}


def test_empty_description_stays_none_and_is_not_inferred() -> None:
    builder = JudgmentEvidenceBuilder(catalog={"p1": _product(description="   ")})

    result = builder.build(["p1"], plan=_plan())

    assert result.candidates[0].description is None
    assert result.stats.truncated_fields == {}


# -- detail relevance --------------------------------------------------------


def test_relevant_details_are_kept_case_insensitively() -> None:
    product = _product(details={"SCENT": "Unscented", "Material": "Plastic"})
    builder = JudgmentEvidenceBuilder(catalog={"p1": product})

    result = builder.build(["p1"], plan=_plan(current=["Scent"]))

    candidate = result.candidates[0]
    assert candidate.attributes["SCENT"] == "Unscented"
    assert "Material" not in candidate.attributes
    reasons = {(detail.key, detail.reason) for detail in result.stats.dropped_details}
    assert ("Material", "irrelevant") in reasons


def test_detail_value_can_match_a_preference() -> None:
    product = _product(details={"Special Feature": "Unscented"})
    builder = JudgmentEvidenceBuilder(catalog={"p1": product})

    result = builder.build(["p1"], plan=_plan(current=["unscented"]))

    assert result.candidates[0].attributes["Special Feature"] == "Unscented"


def test_detail_key_count_is_limited() -> None:
    details = {
        "Scent": "Unscented",
        "Material": "Plastic",
        "Color": "White",
        "Size": "50 ml",
        "Weight": "60 g",
    }
    builder = JudgmentEvidenceBuilder(
        catalog={"p1": _product(details=details)},
        max_detail_keys=2,
    )
    plan = _plan(
        soft=[],
        priority=[],
        priority_attributes=[],
        current=["scent", "material", "color", "size", "weight"],
    )

    result = builder.build(["p1"], plan=plan)

    candidate = result.candidates[0]
    assert _detail_keys(candidate.attributes) == {"Color", "Material"}
    limit_drops = [
        detail for detail in result.stats.dropped_details
        if detail.reason == "max_detail_keys"
    ]
    assert len(limit_drops) == 3


def test_without_relevance_signal_details_are_withheld() -> None:
    product = _product(details={"Scent": "Unscented"})
    builder = JudgmentEvidenceBuilder(catalog={"p1": product})

    result = builder.build(["p1"], plan=_empty_signal_plan())

    candidate = result.candidates[0]
    assert _detail_keys(candidate.attributes) == set()
    assert candidate.description == "Lightweight daily moisturizing cream"
    assert any(
        detail.key == "Scent" and detail.reason == "irrelevant"
        for detail in result.stats.dropped_details
    )


def test_brand_category_title_and_price_are_preserved() -> None:
    builder = JudgmentEvidenceBuilder(catalog={"p1": _product()})

    result = builder.build(["p1"], plan=_plan())

    candidate = result.candidates[0]
    assert candidate.title == "Daily Moisturizing Cream"
    assert candidate.price == pytest.approx(25.99)
    assert candidate.attributes["brand"] == "Acme"
    assert candidate.attributes["main_category"] == "Beauty"


# -- missing products / alignment -------------------------------------------


def test_missing_product_is_reported_and_not_fabricated() -> None:
    builder = JudgmentEvidenceBuilder(catalog={"p1": _product()})

    result = builder.build(["p2", "p1"], plan=_plan())

    assert result.product_ids == ["p1"]
    assert result.stats.missing_products == ["p2"]
    assert result.stats.requested_products == 2
    assert result.stats.returned_products == 1


def test_order_follows_ranked_ids_and_duplicates_are_deduplicated() -> None:
    catalog = {"p2": _product("p2"), "p1": _product("p1")}
    builder = JudgmentEvidenceBuilder(catalog=catalog)

    result = builder.build(["p2", "p1", "p2"], plan=_plan())

    assert result.product_ids == ["p2", "p1"]
    assert result.stats.duplicate_products == ["p2"]


def test_catalog_lookup_is_called_once_with_ranked_ids() -> None:
    calls: list[list[str]] = []

    def lookup(product_ids: list[str]) -> dict[str, ProductRecord]:
        calls.append(list(product_ids))
        return {"p1": _product("p1"), "p2": _product("p2")}

    builder = JudgmentEvidenceBuilder(catalog_lookup=lookup)

    result = builder.build(["p2", "p1"], plan=_plan())

    assert calls == [["p2", "p1"]]
    assert result.product_ids == ["p2", "p1"]


def test_lookup_record_with_wrong_product_id_counts_as_missing() -> None:
    builder = JudgmentEvidenceBuilder(
        catalog_lookup=lambda _ids: {"p1": _product("p2")}
    )

    result = builder.build(["p1"], plan=_plan())

    assert result.candidates == []
    assert result.stats.missing_products == ["p1"]


def test_empty_product_ids_raise_contract_error() -> None:
    builder = JudgmentEvidenceBuilder(catalog={"p1": _product()})

    with pytest.raises(ContractError):
        builder.build([], plan=_plan())


def test_missing_catalog_source_raises_contract_error() -> None:
    builder = JudgmentEvidenceBuilder()

    with pytest.raises(ContractError):
        builder.build(["p1"], plan=_plan())


# -- constraint status (read-only upstream context, Q3) ----------------------


def test_constraint_status_is_passed_through_and_never_inferred() -> None:
    builder = JudgmentEvidenceBuilder(catalog={"p1": _product(price=25.99)})

    without_status = builder.build(["p1"], plan=_plan())
    with_status = builder.build(
        ["p1"],
        plan=_plan(),
        constraint_status_by_id={"p1": ConstraintStatus.UNKNOWN},
    )

    # A cheap product must not be auto-marked "pass" against plan.constraints.hard.
    assert without_status.candidates[0].constraint_status is None
    assert with_status.candidates[0].constraint_status is ConstraintStatus.UNKNOWN


# -- budgets -----------------------------------------------------------------


def test_char_budget_truncates_description_before_identity_fields() -> None:
    long_description = "moisturizing cream " * 60
    builder = JudgmentEvidenceBuilder(
        catalog={
            "p1": _product(
                description=long_description,
                details={"Scent": "Unscented"},
            )
        },
        max_total_chars=400,
    )
    plan = _plan(current=["scent"])

    result = builder.build(["p1"], plan=plan)

    candidate = result.candidates[0]
    assert result.stats.budget_exhausted is True
    assert result.stats.estimated_chars <= result.stats.char_limit == 400
    assert candidate.title == "Daily Moisturizing Cream"
    assert candidate.price == pytest.approx(25.99)
    assert candidate.attributes["brand"] == "Acme"
    assert candidate.attributes["Scent"] == "Unscented"
    assert candidate.description is not None
    assert len(candidate.description) < len(long_description)
    assert "description" in result.stats.truncated_fields["p1"]


def test_budget_overflow_of_core_fields_keeps_alignment_and_reports() -> None:
    builder = JudgmentEvidenceBuilder(
        catalog={
            "p1": _product(
                description="moisturizing cream " * 30,
                details={"Scent": "Unscented"},
            ),
            "p2": _product("p2"),
        },
        max_total_chars=1,
    )

    result = builder.build(["p1", "p2"], plan=_plan(current=["scent"]))

    assert result.product_ids == ["p1", "p2"]
    assert result.stats.budget_exhausted is True
    assert result.stats.estimated_chars > result.stats.char_limit
    assert result.candidates[0].description is None
    assert result.candidates[0].attributes["brand"] == "Acme"
    assert "description" in result.stats.dropped_fields["p1"]
    assert any(
        detail.reason == "budget" for detail in result.stats.dropped_details
    )


def test_token_budget_sets_effective_char_limit() -> None:
    builder = JudgmentEvidenceBuilder(
        catalog={
            "p1": _product(
                title="",
                price=None,
                description="x" * 500,
                details={},
                brand=None,
                main_category=None,
            )
        },
        max_total_chars=10_000,
        max_total_tokens=40,
        chars_per_token=4,
    )

    result = builder.build(["p1"], plan=_empty_signal_plan())

    assert result.stats.char_limit == 160
    assert result.stats.token_limit == 40
    assert result.stats.estimated_chars <= 160
    assert result.stats.estimated_tokens <= 40
    assert result.candidates[0].description is not None
    assert len(result.candidates[0].description) <= 160


def test_builder_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        JudgmentEvidenceBuilder(max_total_chars=0)
    with pytest.raises(ValueError):
        JudgmentEvidenceBuilder(max_total_tokens=0)
    with pytest.raises(ValueError):
        JudgmentEvidenceBuilder(chars_per_token=0)
    with pytest.raises(ValueError):
        JudgmentEvidenceBuilder(
            catalog={},
            catalog_lookup=lambda _ids: {},
        )


# -- Judge integration (stub client, no real API) ----------------------------


def _completion(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-evidence",
        "object": "chat.completion",
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 500, "completion_tokens": 300, "total_tokens": 800},
    }


def _response(content: str) -> httpx.Response:
    return httpx.Response(200, json=_completion(content))


def _item(product_id: str) -> dict[str, Any]:
    return {
        "product_id": product_id,
        "preference_fit": {
            "level": "high",
            "matched_preferences": ["lightweight"],
            "unmatched_preferences": [],
            "unknown_preferences": [],
        },
        "attribute_evidence": {
            "lightweight": {"status": "supported", "source": "description"}
        },
        "match_assessment": "partial_match",
        "uncertainty": "low",
        "warnings": [],
    }


def _items_content(items: list[dict[str, Any]]) -> str:
    return json.dumps({"items": items})


class _ScriptedServer:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected extra DeepSeek request")
        return self.responses.pop(0)


def make_client(handler: Handler) -> tuple[DeepSeekClient, httpx.Client]:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = DeepSeekClient(
        base_url="https://api.test/v1",
        model="deepseek-flash",
        api_key="test-key-not-real",
        timeout_s=5.0,
        max_retries=2,
        max_concurrency=4,
        temperature=0.2,
        max_tokens=2048,
        client=http,
        sleeper=lambda _: None,
        random_source=lambda: 0.5,
    )
    return client, http


def test_judge_ranked_uses_builder_and_keeps_ranked_order() -> None:
    server = _ScriptedServer([_response(_items_content([_item("p1"), _item("p2")]))])
    client, http = make_client(server)
    judge = Judge(client)
    catalog = {
        "p1": _product(
            "p1",
            details={"Scent": "Unscented", "Material": "Plastic"},
            description="Lightweight daily cream",
        ),
        "p2": _product("p2", description="Rich night cream"),
    }

    result = judge.judge_ranked(
        request_id="req-evidence",
        user_request="unscented daily moisturizer",
        plan=_plan(current=["scent"]),
        ranked_product_ids=["p2", "p1"],
        catalog=catalog,
        constraint_status_by_id={"p1": ConstraintStatus.UNKNOWN},
    )
    http.close()

    assert result.product_ids == ["p2", "p1"]
    assert result.evidence.product_ids == ["p2", "p1"]
    assert len(server.requests) == 1
    body = json.loads(server.requests[0].content)
    payload = json.loads(body["messages"][1]["content"])
    candidates = payload["candidates"]
    assert [candidate["product_id"] for candidate in candidates] == ["p2", "p1"]
    assert "Material" not in candidates[1]["attributes"]
    assert candidates[1]["attributes"]["Scent"] == "Unscented"
    assert candidates[1]["attributes"]["brand"] == "Acme"
    assert candidates[1]["constraint_status"] == "unknown"


def test_judge_ranked_skips_missing_products() -> None:
    server = _ScriptedServer([_response(_items_content([_item("p1")]))])
    client, http = make_client(server)
    judge = Judge(client)

    result = judge.judge_ranked(
        request_id="req-missing",
        user_request="daily moisturizer",
        plan=_plan(),
        ranked_product_ids=["p2", "p1"],
        catalog={"p1": _product()},
    )
    http.close()

    assert result.product_ids == ["p1"]
    assert result.evidence.stats.missing_products == ["p2"]


def test_judge_ranked_all_missing_raises_without_api_call() -> None:
    server = _ScriptedServer([])
    client, http = make_client(server)
    judge = Judge(client)

    with pytest.raises(ContractError):
        judge.judge_ranked(
            request_id="req-empty",
            user_request="daily moisturizer",
            plan=_plan(),
            ranked_product_ids=["p-missing"],
            catalog={},
        )
    http.close()

    assert server.requests == []
