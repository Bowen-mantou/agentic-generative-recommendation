"""Phase 5 Planner + DeepSeek client tests (httpx.MockTransport, no real API).

Covers: success path, JSON-mode request shape, invalid JSON -> bounded repair, schema
violation, HTTP 429/500/auth, timeout, JSON-mode compatibility fallback, concurrency
ceiling, secret redaction, Replan hard-constraint preservation, capability/limit checks.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest

from recommender.api.deepseek import (
    DeepSeekAuthError,
    DeepSeekClient,
    DeepSeekJSONError,
    DeepSeekRateLimitError,
    DeepSeekServerError,
    DeepSeekTimeoutError,
    DeepSeekTruncatedError,
    parse_json_content,
)
from recommender.config import PlannerFallbackConfig, load_settings
from recommender.planner import (
    FALLBACK_ASSUMPTIONS,
    INITIAL_PROMPT_VERSION,
    REPLAN_PROMPT_VERSION,
    ControllerFeedback,
    ExecutionFeedback,
    PlanFallbackError,
    PlanGenerationError,
    Planner,
    PlannerInput,
    PlannerLimits,
    ReplanInput,
    SystemCapabilities,
)
from recommender.schemas.common import PlanMode, RecallPath
from recommender.schemas.controller import BudgetState
from recommender.schemas.plan import RecommendationPlan
from recommender.utils.errors import ConfigError

pytestmark = pytest.mark.unit

Handler = Callable[[httpx.Request], httpx.Response]


def _completion(
    content: str,
    *,
    model: str = "deepseek-flash",
    prompt_tokens: int = 120,
    completion_tokens: int = 240,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _response(
    content: str,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(status, json=_completion(content), headers=headers)


def _error_response(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": "test_error"}})


def _truncated_response(reasoning: str = "reasoning content " * 100) -> httpx.Response:
    payload = _completion("")
    payload["choices"][0]["finish_reason"] = "length"
    payload["choices"][0]["message"]["reasoning_content"] = reasoning
    return httpx.Response(200, json=payload)


def _hard_constraint() -> dict[str, Any]:
    return {
        "field": "price",
        "operator": "lte",
        "value": 30,
        "unit": "USD",
        "source": "current_request",
    }


def plan_payload(
    *,
    mode: str = "initial",
    plan_id: str = "plan_001",
    parent_plan_id: str | None = None,
    plan_version: int = 1,
    queries: list[str] | None = None,
    hard: list[dict[str, Any]] | None = None,
    tiger_enabled: bool = False,
    change_summary: list[str] | None = None,
    clarification_required: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "plan_id": plan_id,
        "parent_plan_id": parent_plan_id,
        "plan_version": plan_version,
        "mode": mode,
        "intent": {
            "intent": "product_recommendation",
            "category": "Beauty",
            "product_type": "moisturizer",
            "usage_scenario": "daily_use",
        },
        "constraints": {
            "hard": [_hard_constraint()] if hard is None else hard,
            "soft": [
                {"attribute": "lightweight", "source": "current_request", "priority": "high"}
            ],
        },
        "preferences": {
            "current": ["lightweight"],
            "historical": ["fragrance_free"],
            "negative": ["strong_fragrance"],
            "priority": ["price", "lightweight"],
        },
        "retrieval": {
            "strategy": "hybrid_balanced",
            "tiger": {"enabled": tiger_enabled, "candidate_budget": "standard"},
            "e5_faiss": {
                "enabled": True,
                "queries": queries or ["daily facial moisturizer under 30"],
                "candidate_budget": "standard",
            },
        },
        "ranking": {
            "rerank_query": "daily moisturizer under $30, lightweight texture",
            "priority_attributes": ["price", "lightweight"],
        },
        "clarification": {
            "required": clarification_required,
            "question": "您更偏好无香型还是有香型？" if clarification_required else None,
            "reason": "缺少香型偏好" if clarification_required else None,
        },
        "assumptions": [],
        "change_summary": change_summary or [],
    }
    return payload


def e5_only_capabilities() -> SystemCapabilities:
    return SystemCapabilities(retrieval_modules=[RecallPath.E5_FAISS])


def make_client(
    handler: Handler,
    *,
    max_retries: int = 2,
    max_concurrency: int = 4,
    use_json_mode: bool = True,
    reasoning_effort: str | None = None,
    allow_reasoning_fallback: bool = True,
) -> tuple[DeepSeekClient, list[float], httpx.Client]:
    delays: list[float] = []
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = DeepSeekClient(
        base_url="https://api.test/v1",
        model="deepseek-flash",
        api_key="test-key-not-real",
        timeout_s=5.0,
        max_retries=max_retries,
        max_concurrency=max_concurrency,
        temperature=0.2,
        max_tokens=2048,
        use_json_mode=use_json_mode,
        reasoning_effort=reasoning_effort,
        allow_reasoning_fallback=allow_reasoning_fallback,
        client=http,
        sleeper=delays.append,
        random_source=lambda: 0.5,
    )
    return client, delays, http


class _PlanServer:
    """Scripted HTTP handler; records requests and replays responses in order."""

    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected extra DeepSeek request")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _planner_input(plan_id: str = "plan_fixed") -> PlannerInput:
    return PlannerInput(
        user_request="我想买一款30美元以内、质地清爽的日常面霜",
        plan_id=plan_id,
        user_memory={"long_term_preferences": ["fragrance_free"]},
        session_history=[],
        system_capabilities=e5_only_capabilities(),
    )


def _planner(client: DeepSeekClient, **kwargs: Any) -> Planner:
    return Planner(client, limits=PlannerLimits(max_e5_queries=5, max_query_chars=300), **kwargs)


# -- client -----------------------------------------------------------------


def test_client_sends_json_mode_and_returns_metadata() -> None:
    server = _PlanServer([_response(json.dumps(plan_payload()))])
    client, _, http = make_client(server)

    completion = client.chat_json([{"role": "user", "content": "hello"}])
    assert completion.data["plan_id"] == "plan_001"
    assert completion.completion.model == "deepseek-flash"
    assert completion.completion.prompt_tokens == 120
    assert completion.completion.completion_tokens == 240
    assert completion.completion.latency_ms >= 0.0
    assert completion.completion.request_id.startswith("req_")
    assert completion.completion.trace_id.startswith("tr_")

    request = server.requests[0]
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key-not-real"
    body = json.loads(request.content)
    assert body["response_format"] == {"type": "json_object"}
    assert body["model"] == "deepseek-flash"
    assert body["temperature"] == 0.2
    assert body["stream"] is False
    http.close()


def test_client_describe_and_logs_never_expose_the_key(caplog: pytest.LogCaptureFixture) -> None:
    server = _PlanServer([_response(json.dumps(plan_payload()))])
    client, _, http = make_client(server)
    caplog.set_level("INFO", logger="recommender.api.deepseek")

    client.chat_json([{"role": "user", "content": "hello"}])
    assert "test-key-not-real" not in caplog.text
    assert "test-key-not-real" not in json.dumps(client.describe())
    assert client.describe()["model"] == "deepseek-flash"
    http.close()


def test_parse_json_content_handles_markdown_fence() -> None:
    data = parse_json_content('```json\n{"items": [{"product_id": "p1"}]}\n```')
    assert data["items"][0]["product_id"] == "p1"
    with pytest.raises(DeepSeekJSONError):
        parse_json_content("no json here")


def test_client_retries_429_then_succeeds() -> None:
    server = _PlanServer(
        [_error_response(429, "rate limited"), _response(json.dumps(plan_payload()))]
    )
    client, delays, http = make_client(server)
    completion = client.chat_json([{"role": "user", "content": "hello"}])
    assert completion.completion.attempts == 2
    assert len(delays) == 1 and delays[0] > 0
    assert len(server.requests) == 2
    http.close()


def test_client_retry_after_header_is_honored() -> None:
    server = _PlanServer(
        [
            httpx.Response(
                429,
                json={"error": {"message": "rate limited"}},
                headers={"retry-after": "0.25"},
            ),
            _response(json.dumps(plan_payload())),
        ]
    )
    client, delays, http = make_client(server)
    client.chat_json([{"role": "user", "content": "hello"}])
    assert delays == [0.25]
    http.close()


def test_client_retries_500_until_exhausted() -> None:
    server = _PlanServer([_error_response(500, "boom") for _ in range(3)])
    client, delays, http = make_client(server, max_retries=2)
    with pytest.raises(DeepSeekServerError):
        client.chat_json([{"role": "user", "content": "hello"}])
    assert len(server.requests) == 3 == len(delays) + 1
    http.close()


def test_client_auth_error_is_not_retried() -> None:
    server = _PlanServer([_error_response(401, "invalid api key")])
    client, delays, http = make_client(server, max_retries=2)
    with pytest.raises(DeepSeekAuthError):
        client.chat_json([{"role": "user", "content": "hello"}])
    assert len(server.requests) == 1
    assert delays == []
    http.close()


def test_client_rate_limit_error_is_classified() -> None:
    server = _PlanServer([_error_response(429, "slow down")] * 3)
    client, _, http = make_client(server, max_retries=1)
    with pytest.raises(DeepSeekRateLimitError) as excinfo:
        client.chat_json([{"role": "user", "content": "hello"}])
    assert excinfo.value.retryable is True
    assert excinfo.value.status_code == 429
    http.close()


def test_client_timeout_is_retried_then_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client, delays, http = make_client(handler, max_retries=1)
    with pytest.raises(DeepSeekTimeoutError):
        client.chat_json([{"role": "user", "content": "hello"}])
    assert len(delays) == 1
    http.close()


def test_client_json_mode_rejected_falls_back_without_response_format() -> None:
    server = _PlanServer(
        [
            _error_response(400, "response_format is not supported for this model"),
            _response(json.dumps(plan_payload())),
        ]
    )
    client, _, http = make_client(server)
    completion = client.chat_json([{"role": "user", "content": "hello"}])
    assert completion.data["plan_id"] == "plan_001"
    assert client.json_mode_enabled is False
    first_body = json.loads(server.requests[0].content)
    second_body = json.loads(server.requests[1].content)
    assert "response_format" in first_body
    assert "response_format" not in second_body
    http.close()


def test_client_retries_truncated_reasoning_with_reasoning_disabled() -> None:
    server = _PlanServer(
        [
            _truncated_response(),
            _response(json.dumps(plan_payload())),
        ]
    )
    client, _, http = make_client(server)
    completion = client.chat_json([{"role": "user", "content": "hello"}])
    assert completion.completion.attempts == 2
    first_body = json.loads(server.requests[0].content)
    second_body = json.loads(server.requests[1].content)
    assert "reasoning_effort" not in first_body
    assert second_body["reasoning_effort"] == "none"
    http.close()


def test_client_truncated_reasoning_without_retry_raises_classified_error() -> None:
    server = _PlanServer([_truncated_response()])
    client, _, http = make_client(server, max_retries=0)
    with pytest.raises(DeepSeekTruncatedError) as excinfo:
        client.chat_json([{"role": "user", "content": "hello"}])
    assert excinfo.value.kind.value == "truncated"
    assert excinfo.value.retryable is True
    assert excinfo.value.reasoning_chars > 0
    http.close()


def test_client_explicit_reasoning_effort_is_forwarded_and_not_overridden() -> None:
    server = _PlanServer([_truncated_response()])
    client, _, http = make_client(server, reasoning_effort="low")
    with pytest.raises(DeepSeekTruncatedError):
        client.chat_json([{"role": "user", "content": "hello"}])
    body = json.loads(server.requests[0].content)
    assert body["reasoning_effort"] == "low"
    assert len(server.requests) == 1
    http.close()


def test_client_enforces_concurrency_ceiling() -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return _response("{}")

    client, _, http = make_client(handler, max_concurrency=2)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(
                lambda _: client.chat_json([{"role": "user", "content": "hi"}]), range(6)
            )
        )
    assert len(results) == 6
    assert peak <= 2
    assert peak >= 1
    http.close()


# -- planner ----------------------------------------------------------------


def test_planner_initial_success_and_identity_enforcement() -> None:
    model_payload = plan_payload(plan_id="plan_from_model")
    server = _PlanServer([_response(json.dumps(model_payload))])
    client, _, http = make_client(server)

    result = _planner(client).plan_initial(_planner_input(plan_id="plan_fixed"))
    assert result.plan.plan_id == "plan_fixed"
    assert result.plan.mode is PlanMode.INITIAL
    assert result.plan.plan_version == 1
    assert result.plan.retrieval.e5_faiss.queries == ["daily facial moisturizer under 30"]
    assert result.prompt_version == INITIAL_PROMPT_VERSION
    assert result.attempts == 1
    assert result.repaired is False
    assert result.identity_overrides  # model plan_id was overwritten, and it is recorded
    assert result.input_tokens == 120
    assert result.output_tokens == 240

    request_body = json.loads(server.requests[0].content)
    assert request_body["messages"][0]["role"] == "system"
    assert "planner.initial.v1" in request_body["messages"][0]["content"]
    user_payload = json.loads(request_body["messages"][1]["content"])
    assert user_payload["plan_identity"]["plan_id"] == "plan_fixed"
    assert user_payload["system_capabilities"]["retrieval_modules"] == ["e5_faiss"]
    http.close()


def test_planner_invalid_json_triggers_bounded_repair() -> None:
    server = _PlanServer(
        [_response("this is not json"), _response(json.dumps(plan_payload()))]
    )
    client, _, http = make_client(server)
    result = _planner(client).plan_initial(_planner_input())
    assert result.attempts == 2
    assert result.repaired is True
    second_messages = json.loads(server.requests[1].content)["messages"]
    assert second_messages[-2]["role"] == "assistant"
    assert "不合法" in second_messages[-1]["content"]
    http.close()


def test_planner_invalid_json_twice_raises_structured_error() -> None:
    server = _PlanServer([_response("not json"), _response("{broken")])
    client, _, http = make_client(server)
    with pytest.raises(PlanGenerationError) as excinfo:
        _planner(client).plan_initial(_planner_input())
    error = excinfo.value
    payload = error.to_dict()
    assert payload["degraded"] is True
    assert payload["plan_generated"] is False
    assert payload["mode"] == "initial"
    assert payload["attempts"] == 2
    assert not hasattr(error, "plan")
    assert error.last_error
    assert len(server.requests) == 2
    http.close()


def test_planner_schema_violation_is_repaired_then_failed() -> None:
    invalid = plan_payload()
    invalid["retrieval"]["e5_faiss"]["enabled"] = False  # no retrieval path enabled
    invalid["retrieval"]["tiger"]["enabled"] = False
    server = _PlanServer(
        [_response(json.dumps(invalid)), _response(json.dumps(invalid))]
    )
    client, _, http = make_client(server)
    with pytest.raises(PlanGenerationError) as excinfo:
        _planner(client).plan_initial(_planner_input())
    assert "retrieval path" in excinfo.value.last_error
    http.close()


def test_planner_capability_violation_is_rejected() -> None:
    payload = plan_payload(tiger_enabled=True)  # TIGER not in capabilities
    server = _PlanServer([_response(json.dumps(payload)) for _ in range(2)])
    client, _, http = make_client(server)
    with pytest.raises(PlanGenerationError) as excinfo:
        _planner(client).plan_initial(_planner_input())
    assert "does not provide" in excinfo.value.last_error
    http.close()


def test_planner_query_limit_violation_is_rejected() -> None:
    payload = plan_payload(queries=["query one", "query two"])
    server = _PlanServer([_response(json.dumps(payload)) for _ in range(2)])
    client, _, http = make_client(server)
    planner = Planner(client, limits=PlannerLimits(max_e5_queries=1, max_query_chars=300))
    with pytest.raises(PlanGenerationError) as excinfo:
        planner.plan_initial(_planner_input())
    assert "operational maximum" in excinfo.value.last_error
    http.close()


def test_planner_api_failure_is_structured_and_not_fabricated() -> None:
    server = _PlanServer([_error_response(401, "invalid api key")])
    client, _, http = make_client(server, max_retries=0)
    with pytest.raises(PlanGenerationError) as excinfo:
        _planner(client).plan_initial(_planner_input())
    assert excinfo.value.to_dict()["error_kind"] == "auth"
    http.close()


# -- replanner --------------------------------------------------------------


def _parent_plan() -> RecommendationPlan:
    return RecommendationPlan.model_validate(plan_payload())


def replan_payload(
    *,
    parent: RecommendationPlan | None = None,
    queries: list[str] | None = None,
    hard: list[dict[str, Any]] | None = None,
    change_summary: list[str] | None = None,
) -> dict[str, Any]:
    parent = parent or _parent_plan()
    return plan_payload(
        mode="replan",
        plan_id="plan_replan_001",
        parent_plan_id=parent.plan_id,
        plan_version=parent.plan_version + 1,
        queries=queries or ["fragrance-free lightweight daily moisturizer"],
        hard=hard,
        change_summary=change_summary or ["增加无香型检索方向", "保留原始硬约束"],
    )


def _replan_input(plan_id: str = "plan_replan_fixed") -> ReplanInput:
    return ReplanInput(
        user_request="我想买一款30美元以内、质地清爽的日常面霜",
        previous_plan=_parent_plan(),
        execution_feedback=ExecutionFeedback(
            retrieved_candidates=100,
            after_deduplication=72,
            constraint_satisfied=65,
            judgment_passed=28,
            missing_attributes=["fragrance_free"],
        ),
        controller_feedback=ControllerFeedback(
            reason="INSUFFICIENT_ATTRIBUTE_EVIDENCE",
            suggested_direction="SEARCH_FOR_FRAGRANCE_FREE_PRODUCTS",
        ),
        budget_state=BudgetState(
            remaining_replan_rounds=1,
            remaining_retrieval_calls=4,
            remaining_api_calls=12,
        ),
        plan_id=plan_id,
        system_capabilities=e5_only_capabilities(),
    )


def test_replanner_success_preserves_hard_constraints_and_changes_queries() -> None:
    server = _PlanServer([_response(json.dumps(replan_payload()))])
    client, _, http = make_client(server)
    result = _planner(client).plan_replan(_replan_input())
    assert result.plan.mode is PlanMode.REPLAN
    assert result.plan.parent_plan_id == "plan_001"
    assert result.plan.plan_version == 2
    assert result.prompt_version == REPLAN_PROMPT_VERSION
    assert result.plan.retrieval.e5_faiss.queries == [
        "fragrance-free lightweight daily moisturizer"
    ]
    request_body = json.loads(server.requests[0].content)
    user_payload = json.loads(request_body["messages"][1]["content"])
    assert user_payload["previous_plan"]["constraints"]["hard"] == [_hard_constraint()]
    assert user_payload["budget_state"]["remaining_replan_rounds"] == 1
    http.close()


def test_replanner_cannot_drop_parent_hard_constraint() -> None:
    server = _PlanServer([_response(json.dumps(replan_payload(hard=[]))) for _ in range(2)])
    client, _, http = make_client(server)
    with pytest.raises(PlanGenerationError) as excinfo:
        _planner(client).plan_replan(_replan_input())
    assert "dropped or rewrote hard constraints" in excinfo.value.last_error
    http.close()


def test_replanner_must_change_at_least_one_query() -> None:
    payload = replan_payload(queries=["daily facial moisturizer under 30"])
    server = _PlanServer([_response(json.dumps(payload)) for _ in range(2)])
    client, _, http = make_client(server)
    with pytest.raises(PlanGenerationError) as excinfo:
        _planner(client).plan_replan(_replan_input())
    assert "at least one new query" in excinfo.value.last_error
    http.close()


def test_replanner_requires_change_summary() -> None:
    payload = replan_payload()
    payload["change_summary"] = []
    server = _PlanServer([_response(json.dumps(payload)) for _ in range(2)])
    client, _, http = make_client(server)
    with pytest.raises(PlanGenerationError) as excinfo:
        _planner(client).plan_replan(_replan_input())
    assert "change_summary" in excinfo.value.last_error
    http.close()


def test_unified_generate_dispatch() -> None:
    server = _PlanServer([_response(json.dumps(plan_payload()))])
    client, _, http = make_client(server)
    planner = _planner(client)
    result = planner.generate(PlanMode.INITIAL, _planner_input())
    assert result.plan.mode is PlanMode.INITIAL
    with pytest.raises(ValueError):
        planner.generate(PlanMode.REPLAN, _planner_input())
    http.close()


def test_missing_api_key_raises_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    missing_env = tmp_path / "does-not-exist.env"
    settings = load_settings(env_file=missing_env)
    with pytest.raises(ConfigError):
        DeepSeekClient(
            base_url="https://api.test",
            model="deepseek-flash",
            api_key=None,
            settings=settings,
        )


def test_planner_uses_request_context_ids() -> None:
    from recommender.observability.context import request_context

    server = _PlanServer([_response(json.dumps(plan_payload()))])
    client, _, http = make_client(server)
    with request_context(request_id="req_ctx", trace_id="tr_ctx"):
        result = _planner(client).plan_initial(_planner_input())
    assert result.request_id == "req_ctx"
    assert result.trace_id == "tr_ctx"
    http.close()


# -- D5: strict query normalisation on the Replanner's "new query" check ------


def test_replanner_rejects_full_width_case_variant_of_previous_query() -> None:
    """D5: a full-width/whitespace/case variant is still the same query."""
    parent = _parent_plan()
    payload = replan_payload(
        parent=parent,
        queries=["ＤＡＩＬＹ  ＦＡＣＩＡＬ\tmoisturizer   under 30"],
    )
    server = _PlanServer([_response(json.dumps(payload)) for _ in range(2)])
    client, _, http = make_client(server)
    with pytest.raises(PlanGenerationError) as excinfo:
        _planner(client).plan_replan(_replan_input())
    assert "at least one new query" in excinfo.value.last_error
    http.close()


# -- D7: Planner degradation policy (default structured failure, opt-in fallback) --


def test_planner_fallback_is_disabled_by_default() -> None:
    server = _PlanServer([_response("not json"), _response("still not json")])
    client, _, http = make_client(server)
    planner = _planner(client)
    assert planner.fallback.enabled is False
    with pytest.raises(PlanGenerationError):
        planner.plan_initial(_planner_input())
    http.close()


def test_planner_fallback_enabled_returns_deterministic_marked_plan() -> None:
    server = _PlanServer(
        [_response("not json"), _response("still not json")] * 2
    )
    client, _, http = make_client(server)
    planner = Planner(
        client,
        limits=PlannerLimits(max_e5_queries=5, max_query_chars=300),
        fallback=PlannerFallbackConfig(enabled=True, strategy="balanced"),
    )

    first = planner.plan_initial(_planner_input(plan_id="plan_fixed"))
    second = planner.plan_initial(_planner_input(plan_id="plan_fixed"))

    assert first.degraded is True
    assert first.plan_source == "system_default"
    assert first.attempts == 2
    assert first.input_tokens is None and first.output_tokens is None
    assert first.plan.plan_id == "plan_fixed"
    assert first.plan.mode is PlanMode.INITIAL
    assert first.plan.retrieval.strategy.value == "hybrid_balanced"
    assert first.plan.retrieval.e5_faiss.enabled is True
    assert first.plan.retrieval.e5_faiss.queries == ["我想买一款30美元以内、质地清爽的日常面霜"]
    # The degradation is visible inside the plan itself (Plan has no `degraded` field).
    for assumption in FALLBACK_ASSUMPTIONS:
        assert assumption in first.plan.assumptions
    assert any("Hard constraints were not extracted" in note for note in first.plan.assumptions)
    assert first.plan.constraints.hard == []
    assert first.plan.change_summary == []
    # Deterministic: same input -> byte-identical plan.
    assert first.plan.model_dump(mode="json") == second.plan.model_dump(mode="json")
    http.close()


def test_planner_fallback_uses_deterministic_plan_id_when_input_has_none() -> None:
    server = _PlanServer([_response("not json"), _response("still not json")] * 2)
    client, _, http = make_client(server)
    planner = Planner(
        client,
        limits=PlannerLimits(max_e5_queries=5, max_query_chars=300),
        fallback=PlannerFallbackConfig(enabled=True),
    )

    first = planner.plan_initial(_planner_input(plan_id=""))
    second = planner.plan_initial(_planner_input(plan_id=""))

    assert first.plan.plan_id.startswith("plan_fallback_")
    assert first.plan.plan_id == second.plan.plan_id
    http.close()


def test_planner_fallback_applies_to_api_errors_too() -> None:
    server = _PlanServer([_error_response(401, "invalid api key")])
    client, _, http = make_client(server, max_retries=0)
    planner = Planner(
        client,
        limits=PlannerLimits(max_e5_queries=5, max_query_chars=300),
        fallback=PlannerFallbackConfig(enabled=True),
    )

    result = planner.plan_initial(_planner_input(plan_id="plan_fixed"))

    assert result.degraded is True
    assert result.plan_source == "system_default"
    assert "Planner API unavailable" in result.plan.assumptions
    http.close()


def test_planner_fallback_replan_preserves_parent_constraints_and_changes_query() -> None:
    server = _PlanServer([_response("not json"), _response("still not json")] * 2)
    client, _, http = make_client(server)
    planner = Planner(
        client,
        limits=PlannerLimits(max_e5_queries=5, max_query_chars=300),
        fallback=PlannerFallbackConfig(enabled=True),
    )

    first = planner.plan_replan(_replan_input())
    second = planner.plan_replan(_replan_input())

    parent = _parent_plan()
    plan = first.plan
    assert first.degraded is True
    assert plan.mode is PlanMode.REPLAN
    assert plan.parent_plan_id == parent.plan_id
    assert plan.plan_version == parent.plan_version + 1
    assert plan.constraints.model_dump() == parent.constraints.model_dump()
    assert plan.preferences.model_dump() == parent.preferences.model_dump()
    assert plan.change_summary
    assert plan.retrieval.e5_faiss.queries
    assert plan.retrieval.e5_faiss.queries != parent.retrieval.e5_faiss.queries
    assert first.plan.model_dump(mode="json") == second.plan.model_dump(mode="json")
    http.close()


def test_planner_fallback_without_retrieval_modules_raises_plan_fallback_error() -> None:
    server = _PlanServer([_response("not json"), _response("still not json")])
    client, _, http = make_client(server)
    planner = Planner(
        client,
        limits=PlannerLimits(max_e5_queries=5, max_query_chars=300),
        fallback=PlannerFallbackConfig(enabled=True),
    )

    with pytest.raises(PlanFallbackError) as excinfo:
        planner.plan_initial(
            PlannerInput(
                user_request="我想买一款30美元以内的日常面霜",
                plan_id="plan_fixed",
                system_capabilities=SystemCapabilities(retrieval_modules=[]),
            )
        )
    assert isinstance(excinfo.value.__cause__, PlanGenerationError)
    http.close()


def test_planner_retry_success_is_not_marked_degraded() -> None:
    """D4/D7: a retried-then-successful API call must not stop or degrade the plan."""
    server = _PlanServer(
        [_error_response(429, "rate limited"), _response(json.dumps(plan_payload()))]
    )
    client, _, http = make_client(server)
    planner = Planner(
        client,
        limits=PlannerLimits(max_e5_queries=5, max_query_chars=300),
        fallback=PlannerFallbackConfig(enabled=True),
    )

    result = planner.plan_initial(_planner_input(plan_id="plan_fixed"))

    assert result.degraded is False
    assert result.plan_source is None
    assert result.plan.plan_id == "plan_fixed"
    assert result.plan.retrieval.e5_faiss.queries == ["daily facial moisturizer under 30"]
    assert len(server.requests) == 2
    http.close()
