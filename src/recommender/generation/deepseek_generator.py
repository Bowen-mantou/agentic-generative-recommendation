"""DeepSeek recommendation Generator (Phase 6).

The Generator is the last language-model stage. It only *expresses* the products the
Controller already selected: it may not add products, change their order relative to
the Controller/Reranker ranking, invent attributes or replace catalog facts. Every
reason must cite an allowed evidence path that exists on that candidate
(catalog facts, D6 evidence payload or Judgment output). ``GeneratorCandidate`` is the
stable integration contract used by ``recommender.e2e.service``.

Retry policy (bounded, no silent success):

* at most one *API* retry after a ``DeepSeekAPIError`` (the shared client already does
  its own bounded transport retries);
* at most one *format* repair round after invalid JSON / contract violation;
* after those, :meth:`RecommendationGenerator.generate` returns a clearly marked
  template output with ``model="fallback_template"``. The E2E Gate must not count a
  fallback as a real Generator pass.

No network call or model load happens at import time.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from recommender.api.deepseek import DeepSeekAPIError, DeepSeekClient, JSONCompletion
from recommender.config import GenerationConfig
from recommender.observability.logging import get_logger
from recommender.schemas.common import StopReason
from recommender.schemas.generation import (
    GenerationMetadata,
    RecommendationItem,
    RecommendationOutput,
)

_LOGGER = get_logger(__name__)

GENERATOR_PROMPT_VERSION = "recommendation_generator_v1"
FALLBACK_MODEL_NAME = "fallback_template"
GENERATOR_MAX_API_RETRIES = 1
GENERATOR_MAX_FORMAT_RETRIES = 1
_MAX_REPAIR_ECHO_CHARS = 6000

# The Generator prompt sentences are intentionally long Chinese lines; wrapping them
# would change the text sent to the model.
# ruff: noqa: E501

OUTPUT_SCHEMA_EXAMPLE = """{
  "recommendations": [
    {
      "product_id": "<原样复制输入候选的 product_id>",
      "reasons": [
        {
          "text": "<一句与 current_request 同语言的推荐理由，只包含输入候选中有证据支持的内容>",
          "evidence": "<该理由引用的 allowed_evidence 路径，例如 title、price、evidence.brand、judgment.preference_fit.level>"
        }
      ],
      "limitations": ["<可选的证据缺失/不确定说明，没有则为空数组>"]
    }
  ]
}"""

GENERATOR_SYSTEM_PROMPT = (
    """你是 Agentic Generative Recommendation 系统的 Recommendation Generator。
Prompt 版本：recommendation_generator_v1。

你的唯一任务：为 Controller 已选定的候选商品生成面向前用户的推荐表达和可追溯理由。
候选商品、排序、硬约束结论、Judgment 结果都由上游系统给出，你不得改变它们。

输出必须是一个 JSON 对象，且严格符合以下结构（字段名不得改动，禁止新增字段）：
"""
    + OUTPUT_SCHEMA_EXAMPLE
    + """

规则（违反任何一条都会被视为无效输出并触发重试）：
1. 只能输出输入 candidates 中存在的 product_id；不得新增、猜测或编造任何商品。可以少于候选数量，但不得重复。
2. 必须保持输入 candidates 的相对顺序；不得擅自改变排序（排序由上游 Reranker/Controller 决定）。
3. 所有面向用户的文本（reasons 的 text 与 limitations）必须使用与 current_request 相同的语言：英文请求输出英文理由，中文请求输出中文理由，其他语言同理；不得混用语言，也不得固定输出中文。文本要具体可读，不得只说"很好""适合你"这类无信息量的套话。
4. 每条 reason 必须带 evidence 字段，且 evidence 必须精确等于该商品 allowed_evidence 列表中的某个路径（不要加引号、不要写别的路径）。
   只有当该路径在候选中真实存在且值非空时才可以引用；缺失的属性就是未知，不得把未知写成满足。
5. 不得编造商品的品牌、材质、功效、价格、评分、折扣、库存、物流或任何输入中不存在的信息。
6. reasons 只能使用该商品 title / price / evidence / judgment 中出现的事实；不得把 Judgment 的 uncertain/unknown 说成已证实。
7. limitations 用于说明证据不足、不确定点或受预算限制的影响；没有则输出空数组。
8. 适当参考 controller_context：stop_reason=budget_exhausted 时不得声称已做穷尽比较；quality_satisfied=false 时不得声称完全满足。
9. 只输出 JSON 对象：禁止 markdown 代码块、注释和任何解释文字。"""
)


class GeneratorOutputError(ValueError):
    """The model answer violated the Generator contract (never auto-accepted)."""


@dataclass(frozen=True)
class GeneratorCandidate:
    """One Controller-selected product handed to the Generator.

    ``evidence`` is the D6 evidence payload (catalog facts used for grounding) and
    ``judgment`` is the per-product Judgment result. Both stay opaque dicts so the
    integration contract does not depend on their internal schema version.
    """

    product_id: str
    title: str = ""
    price: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    judgment: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.product_id, str) or not self.product_id.strip():
            raise ValueError("GeneratorCandidate.product_id must be a non-empty string")
        if self.price is not None:
            if isinstance(self.price, bool) or not isinstance(self.price, (int, float)):
                raise ValueError("GeneratorCandidate.price must be a number or None")
            if self.price < 0:
                raise ValueError("GeneratorCandidate.price must be >= 0")
        if not isinstance(self.evidence, Mapping):
            raise ValueError("GeneratorCandidate.evidence must be a mapping")
        if not isinstance(self.judgment, Mapping):
            raise ValueError("GeneratorCandidate.judgment must be a mapping")


# -- evidence paths -----------------------------------------------------------
def _flatten_paths(prefix: str, node: Any, out: dict[str, Any]) -> None:
    """Collect leaf scalar paths so a reason can cite exactly one existing field."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten_paths(child, value, out)
        return
    if isinstance(node, (list, tuple)):
        scalar_items = [
            item
            for item in node
            if isinstance(item, (str, int, float)) and not isinstance(item, bool)
        ]
        if scalar_items and len(scalar_items) == len(node):
            out.setdefault(prefix, list(node))
        for index, value in enumerate(node):
            _flatten_paths(f"{prefix}.{index}", value, out)
        return
    if node is None:
        return
    if isinstance(node, str):
        if node.strip():
            out[prefix] = node
        return
    out[prefix] = node


def candidate_evidence_paths(candidate: GeneratorCandidate) -> dict[str, Any]:
    """Return the evidence paths the model is allowed to cite for ``candidate``."""
    paths: dict[str, Any] = {}
    if candidate.title.strip():
        paths["title"] = candidate.title
    if candidate.price is not None:
        paths["price"] = candidate.price
    for key, value in candidate.evidence.items():
        _flatten_paths(f"evidence.{key}", value, paths)
    for key, value in candidate.judgment.items():
        _flatten_paths(f"judgment.{key}", value, paths)
    return paths


def _candidate_payload(candidate: GeneratorCandidate) -> dict[str, Any]:
    return {
        "product_id": candidate.product_id,
        "title": candidate.title,
        "price": candidate.price,
        "evidence": candidate.evidence,
        "judgment": candidate.judgment,
        "allowed_evidence": sorted(candidate_evidence_paths(candidate)),
    }


# -- prompts ------------------------------------------------------------------
def _json_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def build_generator_messages(
    *,
    request_id: str,
    user_request: str,
    candidates: Sequence[GeneratorCandidate],
    stop_reason: StopReason | None,
    quality_satisfied: bool | None,
    budget_exhausted: bool | None,
) -> list[dict[str, str]]:
    """Render the versioned Generator request (system + one JSON user message)."""
    payload = {
        "task": "generate_recommendation_text",
        "prompt_version": GENERATOR_PROMPT_VERSION,
        "request_id": request_id,
        "current_request": user_request,
        "controller_context": {
            "stop_reason": stop_reason.value
            if isinstance(stop_reason, StopReason)
            else stop_reason,
            "quality_satisfied": quality_satisfied,
            "budget_exhausted": budget_exhausted,
        },
        "candidates": [_candidate_payload(candidate) for candidate in candidates],
        "output_schema": OUTPUT_SCHEMA_EXAMPLE,
        "required_output": "严格按 output_schema 输出 recommendations 数组，只使用输入候选。",
    }
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": _json_payload(payload)},
    ]


def build_generator_repair_messages(
    messages: Sequence[Mapping[str, str]],
    *,
    invalid_output: str,
    error: str,
) -> list[dict[str, str]]:
    """Bounded repair round: echo the invalid answer plus the validator message."""
    trimmed = invalid_output.strip()
    if len(trimmed) > _MAX_REPAIR_ECHO_CHARS:
        trimmed = trimmed[:_MAX_REPAIR_ECHO_CHARS] + "...(truncated)"
    return [dict(message) for message in messages] + [
        {"role": "assistant", "content": trimmed},
        {
            "role": "user",
            "content": (
                "上面的输出不合法，错误为："
                + error[:1200]
                + "\n请只返回修正后的完整 JSON 对象（recommendations 数组），不要输出任何解释文字。"
            ),
        },
    ]


# -- parsing / validation -----------------------------------------------------
def _parse_llm_output(
    data: Mapping[str, Any],
    candidates: Sequence[GeneratorCandidate],
) -> list[RecommendationItem]:
    """Validate the model answer and rebuild items from catalog facts only."""
    raw_items = data.get("recommendations")
    if not isinstance(raw_items, list) or not raw_items:
        raise GeneratorOutputError("model output must contain a non-empty recommendations array")
    allowed = {candidate.product_id: candidate for candidate in candidates}
    order = [candidate.product_id for candidate in candidates]
    parsed: dict[str, tuple[list[str], list[str], list[str]]] = {}
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, Mapping):
            raise GeneratorOutputError(f"recommendations[{index}] is not an object")
        product_id = raw.get("product_id")
        if not isinstance(product_id, str) or not product_id.strip():
            raise GeneratorOutputError(f"recommendations[{index}].product_id is missing")
        product_id = product_id.strip()
        if product_id not in allowed:
            raise GeneratorOutputError(
                f"recommendations[{index}] uses product_id {product_id!r} outside the candidate set"
            )
        if product_id in parsed:
            raise GeneratorOutputError(f"duplicate recommendation for product {product_id!r}")
        allowed_paths = candidate_evidence_paths(allowed[product_id])
        raw_reasons = raw.get("reasons")
        if not isinstance(raw_reasons, list) or not raw_reasons:
            raise GeneratorOutputError(
                f"recommendations[{index}].reasons must be a non-empty array"
            )
        reasons: list[str] = []
        evidence_refs: list[str] = []
        for reason_index, reason in enumerate(raw_reasons):
            if not isinstance(reason, Mapping):
                raise GeneratorOutputError(
                    f"recommendations[{index}].reasons[{reason_index}] is not an object"
                )
            text = reason.get("text")
            if not isinstance(text, str) or not text.strip():
                raise GeneratorOutputError(
                    f"recommendations[{index}].reasons[{reason_index}].text is empty"
                )
            evidence = reason.get("evidence")
            if not isinstance(evidence, str) or evidence.strip() not in allowed_paths:
                hint = ", ".join(sorted(allowed_paths)[:12])
                raise GeneratorOutputError(
                    f"recommendations[{index}] reason cites evidence {evidence!r} which does "
                    f"not exist on product {product_id!r}; allowed paths: {hint}"
                )
            reasons.append(text.strip())
            evidence_refs.append(evidence.strip())
        raw_limitations = raw.get("limitations", []) or []
        if not isinstance(raw_limitations, list) or any(
            not isinstance(item, str) for item in raw_limitations
        ):
            raise GeneratorOutputError(
                f"recommendations[{index}].limitations must be an array of strings"
            )
        parsed[product_id] = (
            reasons,
            evidence_refs,
            [item.strip() for item in raw_limitations if item.strip()],
        )

    # Emit in Controller/Reranker order; never reorder because the model chose to.
    items: list[RecommendationItem] = []
    for product_id in order:
        if product_id not in parsed:
            continue
        candidate = allowed[product_id]
        reasons, evidence_refs, limitations = parsed[product_id]
        items.append(
            RecommendationItem(
                rank=len(items) + 1,
                product_id=product_id,
                title=candidate.title,
                price=candidate.price,
                reasons=reasons,
                evidence_refs=evidence_refs,
                limitations=limitations,
            )
        )
    if not items:
        raise GeneratorOutputError("model returned no usable recommendation")
    return items


def _normalise_stop_reason(value: StopReason | str | None) -> StopReason | None:
    if value is None or isinstance(value, StopReason):
        return value
    try:
        return StopReason(value)
    except ValueError as exc:
        raise ValueError(f"unknown stop_reason {value!r}") from exc


def _dedupe_candidates(candidates: Sequence[GeneratorCandidate]) -> list[GeneratorCandidate]:
    seen: set[str] = set()
    unique: list[GeneratorCandidate] = []
    for candidate in candidates:
        if not isinstance(candidate, GeneratorCandidate):
            raise TypeError("candidates must be GeneratorCandidate instances")
        product_id = candidate.product_id.strip()
        if product_id in seen:
            continue
        seen.add(product_id)
        unique.append(candidate)
    return unique


class RecommendationGenerator:
    """Bounded DeepSeek Generator over Controller-selected candidates."""

    def __init__(self, client: DeepSeekClient, config: GenerationConfig) -> None:
        self._client = client
        self._config = config

    @property
    def prompt_version(self) -> str:
        return GENERATOR_PROMPT_VERSION

    @property
    def top_k(self) -> int:
        return self._config.top_k

    # -- fallback ------------------------------------------------------------
    def _fallback_items(
        self,
        candidates: Sequence[GeneratorCandidate],
    ) -> list[RecommendationItem]:
        items: list[RecommendationItem] = []
        for candidate in candidates[: self._config.top_k]:
            label = candidate.title.strip() or candidate.product_id
            reason = (
                f"「{label}」由 Controller 从有效候选中选定，并已通过上游硬约束与 Judgment 流程。"
            )
            evidence_refs = ["title"]
            if candidate.price is not None:
                reason += f" 目录价格为 {candidate.price:.2f}。"
                evidence_refs.append("price")
            items.append(
                RecommendationItem(
                    rank=len(items) + 1,
                    product_id=candidate.product_id,
                    title=candidate.title,
                    price=candidate.price,
                    reasons=[reason],
                    evidence_refs=evidence_refs,
                    limitations=[
                        "理由生成不可用：本条为模板化兜底输出，不得视为真实 DeepSeek "
                        "Generator 结果。"
                    ],
                )
            )
        return items

    def _fallback(
        self,
        *,
        request_id: str,
        candidates: Sequence[GeneratorCandidate],
        stop_reason: StopReason | None,
        quality_satisfied: bool | None,
        budget_exhausted: bool | None,
        started: float,
        attempts: int,
        api_retries: int,
        format_retries: int,
        error_kind: str,
        error: str,
        completion: JSONCompletion | None = None,
    ) -> RecommendationOutput:
        if not candidates and stop_reason is None:
            raise ValueError(
                "an empty Generator result requires a non-None stop_reason "
                "(RecommendationOutput forbids unexplained empty output)"
            )
        prompt_tokens = None
        completion_tokens = None
        if completion is not None:
            prompt_tokens = completion.completion.prompt_tokens
            completion_tokens = completion.completion.completion_tokens
        _LOGGER.warning(
            "generator fell back to template output",
            extra={
                "request_id": request_id,
                "error_kind": error_kind,
                "attempts": attempts,
                "candidate_count": len(candidates),
            },
        )
        return RecommendationOutput(
            request_id=request_id,
            recommendations=self._fallback_items(candidates),
            model=FALLBACK_MODEL_NAME,
            stop_reason=stop_reason,
            quality_satisfied=quality_satisfied,
            budget_exhausted=budget_exhausted,
            generation_metadata=GenerationMetadata(
                prompt_version=GENERATOR_PROMPT_VERSION,
                attempts=attempts,
                api_retries=api_retries,
                format_retries=format_retries,
                fallback_used=True,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                candidate_count=len(candidates),
                error_kind=error_kind,
                error=error[:800],
            ),
        )

    # -- main entry point ----------------------------------------------------
    def generate(
        self,
        *,
        request_id: str,
        user_request: str,
        candidates: Sequence[GeneratorCandidate],
        stop_reason: StopReason | None,
        quality_satisfied: bool | None,
        budget_exhausted: bool | None,
    ) -> RecommendationOutput:
        """Generate user-facing recommendations for Controller-selected candidates."""
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(user_request, str) or not user_request.strip():
            raise ValueError("user_request must be a non-empty string")
        normalised_stop = _normalise_stop_reason(stop_reason)
        prepared = _dedupe_candidates(candidates)
        if len(prepared) > self._config.top_k:
            prepared = prepared[: self._config.top_k]
        started = time.perf_counter()

        if not prepared:
            return self._fallback(
                request_id=request_id.strip(),
                candidates=[],
                stop_reason=normalised_stop,
                quality_satisfied=quality_satisfied,
                budget_exhausted=budget_exhausted,
                started=started,
                attempts=0,
                api_retries=0,
                format_retries=0,
                error_kind="no_candidates",
                error="no Controller-selected candidate was provided",
            )

        messages = build_generator_messages(
            request_id=request_id.strip(),
            user_request=user_request.strip(),
            candidates=prepared,
            stop_reason=normalised_stop,
            quality_satisfied=quality_satisfied,
            budget_exhausted=budget_exhausted,
        )
        attempts = 0
        api_retries = 0
        format_retries = 0
        last_completion: JSONCompletion | None = None
        while True:
            attempts += 1
            try:
                completion = self._client.chat_json(messages)
            except DeepSeekAPIError as exc:
                if api_retries < GENERATOR_MAX_API_RETRIES:
                    api_retries += 1
                    _LOGGER.warning(
                        "generator API call failed; retrying once",
                        extra={"request_id": request_id, "error_kind": exc.kind.value},
                    )
                    continue
                return self._fallback(
                    request_id=request_id.strip(),
                    candidates=prepared,
                    stop_reason=normalised_stop,
                    quality_satisfied=quality_satisfied,
                    budget_exhausted=budget_exhausted,
                    started=started,
                    attempts=attempts,
                    api_retries=api_retries,
                    format_retries=format_retries,
                    error_kind=f"api_error:{exc.kind.value}",
                    error=f"{type(exc).__name__}: {exc}",
                    completion=last_completion,
                )
            last_completion = completion
            try:
                items = _parse_llm_output(completion.data, prepared)
            except ValueError as exc:
                if format_retries < GENERATOR_MAX_FORMAT_RETRIES:
                    format_retries += 1
                    messages = build_generator_repair_messages(
                        messages,
                        invalid_output=completion.raw_content,
                        error=str(exc),
                    )
                    continue
                return self._fallback(
                    request_id=request_id.strip(),
                    candidates=prepared,
                    stop_reason=normalised_stop,
                    quality_satisfied=quality_satisfied,
                    budget_exhausted=budget_exhausted,
                    started=started,
                    attempts=attempts,
                    api_retries=api_retries,
                    format_retries=format_retries,
                    error_kind="invalid_output",
                    error=f"{type(exc).__name__}: {exc}",
                    completion=completion,
                )
            model_name = completion.completion.model or getattr(self._client, "model", None)
            return RecommendationOutput(
                request_id=request_id.strip(),
                recommendations=items,
                model=model_name or "deepseek",
                stop_reason=normalised_stop,
                quality_satisfied=quality_satisfied,
                budget_exhausted=budget_exhausted,
                generation_metadata=GenerationMetadata(
                    prompt_version=GENERATOR_PROMPT_VERSION,
                    attempts=attempts,
                    api_retries=api_retries,
                    format_retries=format_retries,
                    fallback_used=False,
                    latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
                    prompt_tokens=completion.completion.prompt_tokens,
                    completion_tokens=completion.completion.completion_tokens,
                    candidate_count=len(prepared),
                ),
            )
