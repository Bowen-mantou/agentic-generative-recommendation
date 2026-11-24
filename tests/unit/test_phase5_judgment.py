"""Phase 5 Judgment tests (httpx.MockTransport, no real API).

Covers: one aligned result per product, missing/extra/duplicate products, schema and
evidence-source validation (no inferred attributes), batching, repair round, timeout and
the read-only hard-constraint context.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from recommender.api.deepseek import DeepSeekClient
from recommender.judgment import (
    JUDGMENT_PROMPT_VERSION,
    Judge,
    JudgmentBatchError,
    available_evidence_sources,
)
from recommender.schemas.common import ConstraintStatus, PlanMode, RecallPath
from recommender.schemas.judgment import JudgmentCandidate
from recommender.schemas.plan import RecommendationPlan
from recommender.utils.errors import ContractError

pytestmark = pytest.mark.unit

Handler = Callable[[httpx.Request], httpx.Response]


def _completion(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-judgment",
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


def _response(content: str, *, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=_completion(content))


def _plan() -> RecommendationPlan:
    return RecommendationPlan.model_validate(
        {
            "schema_version": "1.0",
            "plan_id": "plan_judgment",
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
                "hard": [
                    {
                        "field": "price",
                        "operator": "lte",
                        "value": 30,
                        "unit": "USD",
                        "source": "current_request",
                    }
                ],
                "soft": [
                    {
                        "attribute": "lightweight",
                        "source": "current_request",
                        "priority": "high",
                    }
                ],
            },
            "preferences": {
                "current": ["lightweight"],
                "historical": ["fragrance_free"],
                "negative": ["strong_fragrance"],
                "priority": ["price", "lightweight"],
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
                "priority_attributes": ["price", "lightweight"],
            },
            "clarification": {"required": False, "question": None, "reason": None},
            "assumptions": [],
            "change_summary": [],
        }
    )


def _candidate(
    product_id: str = "p1",
    *,
    title: str = "Daily Moisturizing Cream",
    price: float | None = 25.99,
    description: str | None = "Lightweight daily moisturizing cream",
    attributes: dict[str, str] | None = None,
    constraint_status: ConstraintStatus | None = ConstraintStatus.PASS,
) -> JudgmentCandidate:
    return JudgmentCandidate(
        product_id=product_id,
        title=title,
        price=price,
        description=description,
        attributes=attributes if attributes is not None else {"Scent": "Unscented"},
        constraint_status=constraint_status,
    )


def _item(
    product_id: str = "p1",
    *,
    level: str = "high",
    match: str = "partial_match",
    uncertainty: str = "medium",
    evidence: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    matched: list[str] | None = None,
    unknown: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "product_id": product_id,
        "preference_fit": {
            "level": level,
            "matched_preferences": matched if matched is not None else ["lightweight"],
            "unmatched_preferences": [],
            "unknown_preferences": unknown or [],
        },
        "attribute_evidence": (
            evidence
            if evidence is not None
            else {"lightweight": {"status": "supported", "source": "description"}}
        ),
        "match_assessment": match,
        "uncertainty": uncertainty,
        "warnings": warnings or [],
    }


def _items_content(items: list[dict[str, Any]]) -> str:
    return json.dumps({"items": items})


class _ScriptedServer:
    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected extra DeepSeek request")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(handler: Handler, *, max_retries: int = 2) -> tuple[DeepSeekClient, httpx.Client]:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = DeepSeekClient(
        base_url="https://api.test/v1",
        model="deepseek-flash",
        api_key="test-key-not-real",
        timeout_s=5.0,
        max_retries=max_retries,
        max_concurrency=4,
        temperature=0.2,
        max_tokens=2048,
        client=http,
        sleeper=lambda _: None,
        random_source=lambda: 0.5,
    )
    return client, http


def test_judge_returns_one_aligned_record_per_candidate() -> None:
    server = _ScriptedServer([_response(_items_content([_item("p2"), _item("p1")]))])
    client, http = make_client(server)
    batch = Judge(client).judge(
        request_id="req_judge",
        user_request="daily moisturizer under $30",
        plan=_plan(),
        candidates=[_candidate("p1"), _candidate("p2")],
    )
    assert [record.result.product_id for record in batch.records] == ["p1", "p2"]
    for record in batch.records:
        assert record.model == "deepseek-flash"
        assert record.prompt_version == JUDGMENT_PROMPT_VERSION
        assert record.latency_ms is not None
        assert record.input_tokens == 500
        assert record.output_tokens == 300
        assert record.attempts == 1
    assert batch.request_id == "req_judge"
    http.close()


def test_judge_prompt_carries_read_only_constraint_status() -> None:
    server = _ScriptedServer([_response(_items_content([_item("p1"), _item("p2")]))])
    client, http = make_client(server)
    Judge(client).judge(
        request_id="req_judge",
        user_request="daily moisturizer under $30",
        plan=_plan(),
        candidates=[
            _candidate("p1"),
            _candidate("p2", constraint_status=None),
        ],
    )
    body = json.loads(server.requests[0].content)
    payload = json.loads(body["messages"][1]["content"])
    statuses = {
        candidate["product_id"]: candidate["constraint_status"]
        for candidate in payload["candidates"]
    }
    assert statuses == {"p1": "pass", "p2": None}
    assert "constraint_status" in body["messages"][0]["content"]
    http.close()


def test_judge_missing_product_fails_with_structured_error() -> None:
    server = _ScriptedServer(
        [_response(_items_content([_item("p1")])) for _ in range(2)]
    )
    client, http = make_client(server)
    with pytest.raises(JudgmentBatchError) as excinfo:
        Judge(client).judge(
            request_id="req_judge",
            user_request="daily moisturizer under $30",
            plan=_plan(),
            candidates=[_candidate("p1"), _candidate("p2")],
        )
    error = excinfo.value
    assert error.missing_products == ["p2"]
    assert error.attempts == 2
    payload = error.to_dict()
    assert payload["degraded"] is True
    assert payload["judgment_available"] is False
    assert payload["missing_products"] == ["p2"]
    assert len(server.requests) == 2
    http.close()


def test_judge_extra_product_fails() -> None:
    server = _ScriptedServer(
        [_response(_items_content([_item("p1"), _item("p2"), _item("p3")])) for _ in range(2)]
    )
    client, http = make_client(server)
    with pytest.raises(JudgmentBatchError) as excinfo:
        Judge(client).judge(
            request_id="req_judge",
            user_request="daily moisturizer under $30",
            plan=_plan(),
            candidates=[_candidate("p1"), _candidate("p2")],
        )
    assert excinfo.value.extra_products == ["p3"]
    http.close()


def test_judge_duplicate_product_results_fail() -> None:
    server = _ScriptedServer(
        [_response(_items_content([_item("p1"), _item("p1"), _item("p2")])) for _ in range(2)]
    )
    client, http = make_client(server)
    with pytest.raises(JudgmentBatchError):
        Judge(client).judge(
            request_id="req_judge",
            user_request="daily moisturizer under $30",
            plan=_plan(),
            candidates=[_candidate("p1"), _candidate("p2")],
        )
    http.close()


def test_judge_invalid_enum_fails_after_repair_round() -> None:
    bad = _item("p1", match="super_match")
    server = _ScriptedServer([_response(_items_content([bad])) for _ in range(2)])
    client, http = make_client(server)
    with pytest.raises(JudgmentBatchError) as excinfo:
        Judge(client).judge(
            request_id="req_judge",
            user_request="daily moisturizer under $30",
            plan=_plan(),
            candidates=[_candidate("p1")],
        )
    assert "match_assessment" in excinfo.value.last_error
    http.close()


def test_judge_rejects_supported_evidence_for_missing_field() -> None:
    candidate = _candidate("p1", description=None)
    item = _item(
        "p1",
        evidence={"fragrance_free": {"status": "supported", "source": "description"}},
    )
    server = _ScriptedServer([_response(_items_content([item])) for _ in range(2)])
    client, http = make_client(server)
    with pytest.raises(JudgmentBatchError) as excinfo:
        Judge(client).judge(
            request_id="req_judge",
            user_request="daily moisturizer under $30",
            plan=_plan(),
            candidates=[candidate],
        )
    assert "does not exist on the product" in excinfo.value.last_error
    http.close()


def test_judge_rejects_unknown_evidence_that_cites_a_source() -> None:
    item = _item(
        "p1",
        evidence={"fragrance_free": {"status": "unknown", "source": "title"}},
    )
    server = _ScriptedServer([_response(_items_content([item])) for _ in range(2)])
    client, http = make_client(server)
    with pytest.raises(JudgmentBatchError) as excinfo:
        Judge(client).judge(
            request_id="req_judge",
            user_request="daily moisturizer under $30",
            plan=_plan(),
            candidates=[_candidate("p1")],
        )
    assert "unknown" in excinfo.value.last_error
    http.close()


def test_judge_rejects_constraint_status_field_in_model_output() -> None:
    item = _item("p1")
    item["constraint_status"] = "pass"
    server = _ScriptedServer([_response(_items_content([item])) for _ in range(2)])
    client, http = make_client(server)
    with pytest.raises(JudgmentBatchError):
        Judge(client).judge(
            request_id="req_judge",
            user_request="daily moisturizer under $30",
            plan=_plan(),
            candidates=[_candidate("p1")],
        )
    http.close()


def test_judge_accepts_attribute_evidence_from_catalog_attributes() -> None:
    item = _item(
        "p1",
        evidence={
            "fragrance_free": {"status": "supported", "source": "attributes.Scent"},
            "lightweight": {"status": "unknown", "source": None},
        },
    )
    server = _ScriptedServer([_response(_items_content([item]))])
    client, http = make_client(server)
    batch = Judge(client).judge(
        request_id="req_judge",
        user_request="daily moisturizer under $30",
        plan=_plan(),
        candidates=[_candidate("p1")],
    )
    assert batch.records[0].result.attribute_evidence["fragrance_free"].status.value == (
        "supported"
    )
    assert available_evidence_sources(_candidate("p1")) == {
        "title",
        "description",
        "price",
        "scent",
        "attributes.scent",
    }
    http.close()


def test_judge_repairs_invalid_json_then_succeeds() -> None:
    server = _ScriptedServer(
        [_response("not json"), _response(_items_content([_item("p1")]))]
    )
    client, http = make_client(server)
    batch = Judge(client).judge(
        request_id="req_judge",
        user_request="daily moisturizer under $30",
        plan=_plan(),
        candidates=[_candidate("p1")],
    )
    assert len(batch.records) == 1
    second_messages = json.loads(server.requests[1].content)["messages"]
    assert "不合法" in second_messages[-1]["content"]
    http.close()


def test_judge_splits_candidates_into_batches() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        ids = [candidate["product_id"] for candidate in payload["candidates"]]
        calls.append(len(ids))
        return _response(_items_content([_item(product_id) for product_id in ids]))

    client, http = make_client(handler)
    judge = Judge(client, batch_size=2)
    candidates = [_candidate(f"p{index}") for index in range(1, 6)]
    batch = judge.judge(
        request_id="req_judge",
        user_request="daily moisturizer under $30",
        plan=_plan(),
        candidates=candidates,
    )
    assert judge.batch_count(5) == 3
    assert calls == [2, 2, 1]
    assert [record.result.product_id for record in batch.records] == [
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
    ]
    http.close()


def test_judge_rejects_duplicate_input_candidates() -> None:
    server = _ScriptedServer([])
    client, http = make_client(server)
    with pytest.raises(ContractError):
        Judge(client).judge(
            request_id="req_judge",
            user_request="daily moisturizer under $30",
            plan=_plan(),
            candidates=[_candidate("p1"), _candidate("p1")],
        )
    http.close()


def test_judge_rejects_empty_candidates() -> None:
    server = _ScriptedServer([])
    client, http = make_client(server)
    with pytest.raises(ContractError):
        Judge(client).judge(
            request_id="req_judge",
            user_request="daily moisturizer under $30",
            plan=_plan(),
            candidates=[],
        )
    http.close()


def test_judge_timeout_is_structured_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client, http = make_client(handler, max_retries=0)
    with pytest.raises(JudgmentBatchError) as excinfo:
        Judge(client).judge(
            request_id="req_judge",
            user_request="daily moisturizer under $30",
            plan=_plan(),
            candidates=[_candidate("p1")],
        )
    assert excinfo.value.error_kind == "timeout"
    http.close()


def test_judgment_candidate_allows_no_configured_constraint() -> None:
    candidate = _candidate("p1", constraint_status=None)
    assert candidate.constraint_status is None


def test_recommendation_plan_mode_is_initial() -> None:
    # Guard against a helper drifting into an invalid plan that would mask judge tests.
    assert _plan().mode is PlanMode.INITIAL
    assert _plan().retrieval.enabled_paths() == [RecallPath.E5_FAISS]
