"""Phase 6 DeepSeek Generator tests (stub client; no real API).

Covers: successful grounded generation, candidate-ID subset enforcement, catalog
truth over model text, evidence-path validation, bounded API/format retries, order
preservation and the explicit ``fallback_template`` degradation path.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from recommender.api.deepseek import (
    ChatCompletion,
    DeepSeekTimeoutError,
    JSONCompletion,
)
from recommender.config import GenerationConfig
from recommender.generation import (
    FALLBACK_MODEL_NAME,
    GENERATOR_PROMPT_VERSION,
    GeneratorCandidate,
    RecommendationGenerator,
    candidate_evidence_paths,
)
from recommender.schemas.common import StopReason
from recommender.schemas.generation import RecommendationOutput

pytestmark = pytest.mark.unit


def _candidate(
    product_id: str = "B001",
    *,
    title: str = "Fragrance Free Daily Moisturizer",
    price: float | None = 12.99,
    evidence: dict[str, Any] | None = None,
    judgment: dict[str, Any] | None = None,
) -> GeneratorCandidate:
    return GeneratorCandidate(
        product_id=product_id,
        title=title,
        price=price,
        evidence=evidence
        if evidence is not None
        else {
            "brand": "ExampleBrand",
            "main_category": "Beauty",
            "details": {"Scent": "Fragrance Free", "Skin Type": "Sensitive"},
        },
        judgment=judgment
        if judgment is not None
        else {
            "preference_fit": {"level": "high", "matched_preferences": ["fragrance_free"]},
            "match_assessment": "direct_match",
            "uncertainty": "low",
            "warnings": [],
        },
    )


def _completion(
    payload: dict[str, Any],
    *,
    model: str = "deepseek-flash",
    prompt_tokens: int = 640,
    completion_tokens: int = 220,
) -> JSONCompletion:
    content = json.dumps(payload, ensure_ascii=False)
    return JSONCompletion(
        data=payload,
        completion=ChatCompletion(
            content=content,
            model=model,
            request_id="req_generator",
            trace_id="trace_generator",
            latency_ms=321.0,
            attempts=1,
            http_status=200,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop",
        ),
    )


def _valid_payload(
    *candidates: GeneratorCandidate,
    evidence_path: str = "evidence.brand",
) -> dict[str, Any]:
    return {
        "recommendations": [
            {
                "product_id": candidate.product_id,
                "reasons": [
                    {
                        "text": f"「{candidate.title}」符合当前需求，品牌信息来自目录证据。",
                        "evidence": evidence_path,
                    }
                ],
                "limitations": [],
            }
            for candidate in candidates
        ]
    }


class StubClient:
    model = "deepseek-flash"

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def chat_json(self, messages: list[dict[str, str]], **_: Any) -> JSONCompletion:
        self.calls.append([dict(message) for message in messages])
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _generator(client: StubClient, *, top_k: int = 5) -> RecommendationGenerator:
    return RecommendationGenerator(client, GenerationConfig(top_k=top_k))  # type: ignore[arg-type]


def _generate(
    generator: RecommendationGenerator,
    candidates: list[GeneratorCandidate] | None = None,
    *,
    stop_reason: StopReason | None = StopReason.QUALITY_SATISFIED,
    quality_satisfied: bool | None = True,
    budget_exhausted: bool | None = False,
) -> RecommendationOutput:
    return generator.generate(
        request_id="req_generator",
        user_request="fragrance-free moisturizer for sensitive skin under $20",
        candidates=candidates if candidates is not None else [_candidate(), _candidate("B002")],
        stop_reason=stop_reason,
        quality_satisfied=quality_satisfied,
        budget_exhausted=budget_exhausted,
    )


# -- success ------------------------------------------------------------------
def test_generate_returns_grounded_recommendations_in_input_order() -> None:
    first = _candidate()
    second = _candidate("B002", title="Gentle Cleanser", price=None)
    payload = {
        "recommendations": [
            {
                "product_id": "B002",
                "reasons": [
                    {
                        "text": "温和洁面，适合敏感肌日常清洁。",
                        "evidence": "judgment.preference_fit.level",
                    }
                ],
                "limitations": ["价格未知"],
                "title": "FABRICATED TITLE",
                "price": 0.01,
            },
            {
                "product_id": "B001",
                "reasons": [
                    {
                        "text": "无香精保湿，符合敏感肌需求。",
                        "evidence": "evidence.details.Scent",
                    }
                ],
                "limitations": [],
            },
        ]
    }
    client = StubClient([_completion(payload)])
    output = _generate(_generator(client), [first, second])

    assert output.model == "deepseek-flash"
    assert output.product_ids == ["B001", "B002"]  # Controller order, not model order
    assert [item.rank for item in output.recommendations] == [1, 2]
    # Catalog facts win: model-supplied title/price are ignored.
    assert output.recommendations[0].title == first.title
    assert output.recommendations[0].price == pytest.approx(12.99)
    assert output.recommendations[1].price is None
    assert all(item.reasons and item.reasons[0].strip() for item in output.recommendations)
    assert output.recommendations[0].evidence_refs == ["evidence.details.Scent"]
    assert output.recommendations[1].evidence_refs == ["judgment.preference_fit.level"]
    assert output.generation_metadata is not None
    metadata = output.generation_metadata
    assert metadata.prompt_version == GENERATOR_PROMPT_VERSION
    assert metadata.fallback_used is False
    assert metadata.attempts == 1
    assert metadata.format_retries == 0
    assert metadata.api_retries == 0
    assert metadata.prompt_tokens == 640
    assert metadata.completion_tokens == 220
    assert metadata.candidate_count == 2
    assert len(client.calls) == 1


def test_generate_allows_candidate_subset_and_keeps_evidence_citation() -> None:
    first = _candidate()
    second = _candidate("B002", title="Gentle Cleanser")
    client = StubClient([_completion(_valid_payload(second))])
    output = _generate(_generator(client), [first, second])
    assert output.product_ids == ["B002"]
    assert output.recommendations[0].rank == 1


def test_prompt_contains_candidates_and_controller_context() -> None:
    candidate = _candidate()
    client = StubClient([_completion(_valid_payload(candidate))])
    _generate(
        _generator(client),
        [candidate],
        stop_reason=StopReason.BUDGET_EXHAUSTED,
        quality_satisfied=False,
        budget_exhausted=True,
    )
    messages = client.calls[0]
    assert messages[0]["role"] == "system"
    assert GENERATOR_PROMPT_VERSION in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["prompt_version"] == GENERATOR_PROMPT_VERSION
    assert payload["controller_context"]["stop_reason"] == "budget_exhausted"
    assert payload["controller_context"]["budget_exhausted"] is True
    assert payload["candidates"][0]["product_id"] == candidate.product_id
    assert "evidence.brand" in payload["candidates"][0]["allowed_evidence"]


def test_prompt_requires_reason_language_to_match_request() -> None:
    candidate = _candidate()
    client = StubClient([_completion(_valid_payload(candidate))])
    _generate(_generator(client), [candidate])  # user_request is English
    system = client.calls[0][0]["content"]
    assert "与 current_request 相同的语言" in system
    assert "英文请求输出英文理由" in system
    assert "中文请求输出中文理由" in system
    # The English answer must pass; the prompt is what steers the language.
    exchange = client.calls[0]
    user_payload = json.loads(exchange[1]["content"])
    assert user_payload["current_request"].startswith("fragrance-free")


# -- format repair ------------------------------------------------------------
def test_invalid_first_answer_is_repaired_once() -> None:
    candidate = _candidate()
    client = StubClient(
        [_completion({"recommendations": []}), _completion(_valid_payload(candidate))]
    )
    output = _generate(_generator(client), [candidate])
    assert output.model == "deepseek-flash"
    assert output.product_ids == [candidate.product_id]
    assert len(client.calls) == 2
    # The repair round echoes the invalid answer and the validator error.
    assert client.calls[1][-2]["role"] == "assistant"
    assert "不合法" in client.calls[1][-1]["content"]
    metadata = output.generation_metadata
    assert metadata is not None and metadata.format_retries == 1 and metadata.attempts == 2


def test_out_of_candidate_product_is_rejected_then_repaired() -> None:
    candidate = _candidate()
    bad = _valid_payload(candidate)
    bad["recommendations"][0]["product_id"] = "NOT_A_CANDIDATE"
    client = StubClient([_completion(bad), _completion(_valid_payload(candidate))])
    output = _generate(_generator(client), [candidate])
    assert output.product_ids == [candidate.product_id]
    assert output.generation_metadata is not None
    assert output.generation_metadata.format_retries == 1


def test_reason_without_evidence_path_is_rejected() -> None:
    candidate = _candidate()
    bad = _valid_payload(candidate, evidence_path="evidence.made_up_attribute")
    client = StubClient([_completion(bad), _completion(_valid_payload(candidate))])
    output = _generate(_generator(client), [candidate])
    assert output.product_ids == [candidate.product_id]
    assert output.generation_metadata is not None
    assert output.generation_metadata.format_retries == 1


def test_empty_reason_is_rejected_and_eventually_falls_back() -> None:
    candidate = _candidate()
    bad = _valid_payload(candidate)
    bad["recommendations"][0]["reasons"][0]["text"] = "   "
    client = StubClient([_completion(bad), _completion(bad)])
    output = _generate(_generator(client), [candidate])
    assert output.model == FALLBACK_MODEL_NAME
    assert output.recommendations[0].reasons
    assert output.generation_metadata is not None
    assert output.generation_metadata.fallback_used is True
    assert output.generation_metadata.error_kind == "invalid_output"
    assert len(client.calls) == 2


# -- API retry / fallback -----------------------------------------------------
def test_transient_api_error_is_retried_once() -> None:
    candidate = _candidate()
    client = StubClient(
        [
            DeepSeekTimeoutError("timed out", retryable=True),
            _completion(_valid_payload(candidate)),
        ]
    )
    output = _generate(_generator(client), [candidate])
    assert output.model == "deepseek-flash"
    assert output.generation_metadata is not None
    assert output.generation_metadata.api_retries == 1
    assert output.generation_metadata.attempts == 2
    assert output.generation_metadata.fallback_used is False


def test_two_api_errors_fall_back_without_faking_success() -> None:
    candidate = _candidate()
    client = StubClient(
        [
            DeepSeekTimeoutError("timed out", retryable=True),
            DeepSeekTimeoutError("timed out again", retryable=True),
        ]
    )
    output = _generate(_generator(client), [candidate])
    assert output.model == FALLBACK_MODEL_NAME
    assert output.product_ids == [candidate.product_id]
    metadata = output.generation_metadata
    assert metadata is not None
    assert metadata.error_kind == "api_error:timeout"
    assert metadata.api_retries == 1
    assert metadata.fallback_used is True
    assert len(client.calls) == 2


def test_persistent_invalid_output_falls_back_with_template_reasons() -> None:
    candidates = [_candidate(), _candidate("B002", title="Gentle Cleanser")]
    bad = {"recommendations": [{"product_id": "B001", "reasons": []}]}
    client = StubClient([_completion(bad), _completion(bad)])
    output = _generate(_generator(client), candidates)
    assert output.model == FALLBACK_MODEL_NAME
    assert output.product_ids == ["B001", "B002"]
    assert all(item.reasons for item in output.recommendations)
    assert all(item.limitations for item in output.recommendations)
    metadata = output.generation_metadata
    assert metadata is not None
    assert metadata.fallback_used is True
    assert metadata.error_kind == "invalid_output"
    assert metadata.attempts == 2


# -- input handling -----------------------------------------------------------
def test_empty_candidates_return_marked_empty_fallback() -> None:
    client = StubClient([])
    output = _generate(_generator(client), [], stop_reason=StopReason.NO_VALID_CANDIDATES)
    assert output.model == FALLBACK_MODEL_NAME
    assert output.recommendations == []
    assert output.stop_reason is StopReason.NO_VALID_CANDIDATES
    assert output.generation_metadata is not None
    assert output.generation_metadata.error_kind == "no_candidates"
    assert client.calls == []


def test_empty_candidates_without_stop_reason_is_rejected() -> None:
    with pytest.raises(ValueError):
        _generate(_generator(StubClient([])), [], stop_reason=None)


def test_candidates_are_deduped_and_limited_to_top_k() -> None:
    first = _candidate()
    second = _candidate("B002")
    client = StubClient([_completion(_valid_payload(first))])
    output = _generate(_generator(client, top_k=1), [first, second, first])
    assert output.product_ids == ["B001"]
    payload = json.loads(client.calls[0][1]["content"])
    assert [item["product_id"] for item in payload["candidates"]] == ["B001"]
    assert output.generation_metadata is not None
    assert output.generation_metadata.candidate_count == 1


def test_invalid_request_arguments_are_rejected() -> None:
    generator = _generator(StubClient([]))
    with pytest.raises(ValueError):
        generator.generate(
            request_id="",
            user_request="q",
            candidates=[],
            stop_reason=StopReason.NO_VALID_CANDIDATES,
            quality_satisfied=False,
            budget_exhausted=False,
        )
    with pytest.raises(ValueError):
        generator.generate(
            request_id="r",
            user_request="   ",
            candidates=[],
            stop_reason=StopReason.NO_VALID_CANDIDATES,
            quality_satisfied=False,
            budget_exhausted=False,
        )
    with pytest.raises(ValueError):
        generator.generate(
            request_id="r",
            user_request="q",
            candidates=[],
            stop_reason="not_a_stop_reason",  # type: ignore[arg-type]
            quality_satisfied=False,
            budget_exhausted=False,
        )


# -- evidence paths -----------------------------------------------------------
def test_candidate_evidence_paths_only_include_existing_scalar_values() -> None:
    candidate = GeneratorCandidate(
        product_id="B001",
        title="Moisturizer",
        price=None,
        evidence={"brand": "Acme", "empty": "", "details": {"Scent": "None"}},
        judgment={"preference_fit": {"level": "high"}, "warnings": []},
    )
    paths = candidate_evidence_paths(candidate)
    assert paths["title"] == "Moisturizer"
    assert paths["evidence.brand"] == "Acme"
    assert paths["evidence.details.Scent"] == "None"
    assert "price" not in paths
    assert "evidence.empty" not in paths
    assert "judgment.preference_fit.level" in paths
    assert "judgment.warnings" not in paths


def test_generator_candidate_validates_inputs() -> None:
    with pytest.raises(ValueError):
        GeneratorCandidate(product_id=" ")
    with pytest.raises(ValueError):
        GeneratorCandidate(product_id="B001", price=-1.0)
