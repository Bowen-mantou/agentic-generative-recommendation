"""Versioned prompts and typed inputs for the Initial Planner / Replanner.

Prompt text is code here (not free-floating config) so a prompt change is a reviewable
diff with an explicit version bump. The model authors only the *semantic* plan; the
``plan_id`` / ``parent_plan_id`` / ``plan_version`` / ``mode`` identity fields are
asserted by :class:`~recommender.planner.planner.Planner` because they are system
bookkeeping, not model judgement.

Initial and Replan prompts share the unified ``RecommendationPlan`` schema (module 2
§5--6); there are no parallel top-level ``retrieval_strategy`` / ``tiger`` / ``e5_faiss``
fields and the prompt forbids extra keys.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field, field_validator

from recommender.schemas.common import ContractModel, NonEmptyStr, RecallPath
from recommender.schemas.controller import BudgetState
from recommender.schemas.plan import (
    MAX_E5_QUERIES,
    MAX_QUERY_CHARS,
    RecommendationPlan,
)

# The literal prompt sentences are intentionally long Chinese lines; artificial wrapping
# inside the prompt would change the text sent to the model.
# ruff: noqa: E501

INITIAL_PROMPT_VERSION = "planner.initial.v1"
REPLAN_PROMPT_VERSION = "planner.replan.v1"

OUTPUT_SCHEMA_EXAMPLE = """{
  "schema_version": "1.0",
  "plan_id": "<输入中给定的 plan_id>",
  "parent_plan_id": null,
  "plan_version": 1,
  "mode": "initial",
  "intent": {
    "intent": "product_recommendation",
    "category": "<商品大类，可为 null>",
    "product_type": "<商品类型，可为 null>",
    "usage_scenario": "<使用场景，可为 null>"
  },
  "constraints": {
    "hard": [
      {"field": "price", "operator": "lte", "value": 30, "unit": "USD", "source": "current_request"}
    ],
    "soft": [
      {"attribute": "lightweight", "source": "current_request", "priority": "high"}
    ]
  },
  "preferences": {
    "current": ["<当前请求中的偏好>"],
    "historical": ["<记忆中的历史偏好>"],
    "negative": ["<负向偏好>"],
    "priority": ["<排序优先级>"]
  },
  "retrieval": {
    "strategy": "hybrid_balanced",
    "tiger": {"enabled": true, "candidate_budget": "standard"},
    "e5_faiss": {
      "enabled": true,
      "queries": ["<自然语言检索 query>"],
      "candidate_budget": "standard"
    }
  },
  "ranking": {
    "rerank_query": "<用于精排的非空查询>",
    "priority_attributes": ["price", "lightweight"]
  },
  "clarification": {"required": false, "question": null, "reason": null},
  "assumptions": [],
  "change_summary": []
}"""

_COMMON_RULES = """通用规则（违反任何一条都会被视为无效输出并触发重试）：
1. 只输出一个 JSON 对象，禁止 markdown 代码块、注释、前后解释文字。
2. 字段名必须与给定结构完全一致，禁止新增、拼写变体或顶层平行字段（例如禁止顶层 retrieval_strategy / tiger / e5_faiss）。
3. 你只负责制定计划：不得执行检索、不得生成或猜测商品 ID、不得编造商品属性和价格。
4. 硬约束只能是可用商品字段上的确定性比较，operator 取 eq/ne/lt/lte/gt/gte/in/not_in/contains，value 为 JSON 标量或标量数组，unit 仅用于价格等带单位字段。当前请求中明确提出的条件才可成为硬约束；来自记忆或历史行为的偏好只能是软偏好，不得自动升级为硬约束。
5. 软偏好 source 取 current_request/user_memory/historical_behavior/system_default，priority 取 high/medium/low。
6. retrieval.strategy 取 hybrid_balanced/behavior_priority/semantic_priority；至少启用一条召回路径；system_capabilities.retrieval_modules 之外的路径必须 enabled=false；启用 e5_faiss 时 queries 必须有 max_e5_queries 条以内、互不重复的自然语言 query，每条不超过 max_query_chars 个字符；candidate_budget 只能取 standard/expanded。
7. ranking.rerank_query 必须是非空字符串；priority_attributes 是字符串数组。
8. clarification.required=false 时 question 和 reason 必须为 null；required=true 时 question 必须非空且 reason 简短说明原因。
9. 不得输出剩余预算/次数等计数器；预算由 Controller 的 BudgetState 管理，规划中的 candidate_budget 只是档位选择。
10. 只使用 system_capabilities.available_product_fields 中列出的商品字段名。"""

INITIAL_SYSTEM_PROMPT = (
    """你是 Agentic Generative Recommendation 系统的 Initial Recommendation Planner。
Prompt 版本：planner.initial.v1。

你的唯一任务是：理解用户当前购物需求，融合用户记忆与会话上下文，输出一个符合统一
RecommendationPlan Schema 的初始推荐计划。

输出结构（严格遵循，null 字段也必须出现）：
"""
    + OUTPUT_SCHEMA_EXAMPLE
    + """

Initial 模式专属规则：
- plan_identity 中 mode 必须为 "initial"，parent_plan_id 必须为 null，plan_version 必须为 1，plan_id 使用输入给定的值。
- 初始计划的 change_summary 必须为空数组 []。
- assumptions 只在确有必要的规划假设时填写，不得用于罗列用户未说过的事实。

"""
    + _COMMON_RULES
)

REPLAN_SYSTEM_PROMPT = (
    """你是 Agentic Generative Recommendation 系统的 Recommendation Replanner。
Prompt 版本：planner.replan.v1。

你的任务不是重新开始整个推荐任务，而是根据上一轮执行反馈，在保留原始需求与硬约束的前提下
输出一个完整的、新的 RecommendationPlan。

输出结构与 initial 模式相同（示例中的 mode/plan_version 需按 replan 规则填写）：
"""
    + OUTPUT_SCHEMA_EXAMPLE
    + """

Replan 模式专属规则（必须全部满足）：
1. mode 必须为 "replan"。
2. plan_id 使用输入给定的新 plan_id；parent_plan_id 必须等于 previous_plan.plan_id；plan_version 必须等于 previous_plan.plan_version + 1。
3. 必须完整保留 previous_plan.constraints.hard 中的每一条硬约束，field/operator/value/unit/source 不得改动、删除或放宽；用户没有明确修改需求时不得改写硬约束。
4. 可以调整软偏好、召回策略、检索 query、candidate_budget 与精排目标，但必须针对 controller_feedback 指出的问题。
5. e5_faiss.queries 必须至少包含一条与上一轮不同的新 query，不得原样重复上一轮查询。
6. budget_state 只是只读参考：不得在计划中输出或修改剩余次数，candidate_budget 仍只能是 standard/expanded。
7. change_summary 必须至少有一条，简洁说明本轮相对上一版计划改了什么（中文）。
8. intent 中的用户原始需求不得被改写：如果用户没有改变品类/场景，保持与上一轮一致。

"""
    + _COMMON_RULES
)


class PlannerLimits(ContractModel):
    """Operational query limits (config-driven; stricter than protocol maxima)."""

    max_e5_queries: int = Field(default=5, ge=1, le=MAX_E5_QUERIES)
    max_query_chars: int = Field(default=300, ge=1, le=MAX_QUERY_CHARS)


class SystemCapabilities(ContractModel):
    """What the current deployment can actually execute (module 2 §3.4)."""

    retrieval_modules: list[RecallPath] = Field(
        default_factory=lambda: [RecallPath.TIGER, RecallPath.E5_FAISS]
    )
    available_product_fields: list[str] = Field(
        default_factory=lambda: [
            "price",
            "category",
            "main_category",
            "brand",
            "rating",
            "title",
            "description",
            "details",
        ]
    )
    reranker_available: bool = True
    judgment_available: bool = True

    @field_validator("retrieval_modules")
    @classmethod
    def _modules_are_unique(cls, value: list[RecallPath]) -> list[RecallPath]:
        if len(set(value)) != len(value):
            raise ValueError("retrieval_modules must not repeat a path")
        return value

    @field_validator("available_product_fields")
    @classmethod
    def _fields_are_unique(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("available_product_fields must not repeat a field")
        return cleaned


class PlannerInput(ContractModel):
    """Inputs of the Initial Planner (module 2 §5.1)."""

    user_request: NonEmptyStr
    plan_id: str | None = None
    user_memory: dict[str, Any] = Field(default_factory=dict)
    session_history: list[dict[str, Any]] = Field(default_factory=list)
    system_capabilities: SystemCapabilities = SystemCapabilities()


class ExecutionFeedback(ContractModel):
    """Structured summary of the previous round (module 2 §5.3).

    Only aggregate statistics and representative failures are passed; full candidate
    payloads never reach the Replanner prompt.
    """

    retrieved_candidates: int = Field(default=0, ge=0)
    after_deduplication: int = Field(default=0, ge=0)
    constraint_satisfied: int = Field(default=0, ge=0)
    judgment_passed: int = Field(default=0, ge=0)
    missing_attributes: list[str] = Field(default_factory=list)
    diversity_issue: bool = False
    notes: list[str] = Field(default_factory=list)


class ControllerFeedback(ContractModel):
    """Why the Controller asked for a replan, with an optional suggested direction."""

    reason: str = Field(min_length=1)
    suggested_direction: str | None = None
    reasons: list[str] = Field(default_factory=list)


class ReplanInput(ContractModel):
    """Inputs of the Replanner (module 2 §5.3)."""

    user_request: NonEmptyStr
    previous_plan: RecommendationPlan
    execution_feedback: ExecutionFeedback
    controller_feedback: ControllerFeedback
    budget_state: BudgetState
    plan_id: str | None = None
    user_memory: dict[str, Any] = Field(default_factory=dict)
    session_history: list[dict[str, Any]] = Field(default_factory=list)
    system_capabilities: SystemCapabilities = SystemCapabilities()


def _json_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _capabilities_payload(capabilities: SystemCapabilities) -> dict[str, Any]:
    return {
        "retrieval_modules": [module.value for module in capabilities.retrieval_modules],
        "available_product_fields": capabilities.available_product_fields,
        "reranker_available": capabilities.reranker_available,
        "judgment_available": capabilities.judgment_available,
    }


def _limits_payload(limits: PlannerLimits) -> dict[str, int]:
    return {
        "max_e5_queries": limits.max_e5_queries,
        "max_query_chars": limits.max_query_chars,
    }


def build_initial_messages(
    data: PlannerInput,
    *,
    limits: PlannerLimits,
    plan_id: str,
) -> list[dict[str, str]]:
    """Render the Initial Planner prompt for one request."""
    payload = {
        "task": "initial_plan",
        "plan_identity": {
            "schema_version": "1.0",
            "plan_id": plan_id,
            "parent_plan_id": None,
            "plan_version": 1,
            "mode": "initial",
        },
        "current_request": data.user_request,
        "user_memory": data.user_memory,
        "session_history": data.session_history,
        "system_capabilities": _capabilities_payload(data.system_capabilities),
        "retrieval_limits": _limits_payload(limits),
        "required_output": "严格按 system prompt 中的 RecommendationPlan JSON 结构输出。",
    }
    return [
        {"role": "system", "content": INITIAL_SYSTEM_PROMPT},
        {"role": "user", "content": _json_payload(payload)},
    ]


def build_replan_messages(
    data: ReplanInput,
    *,
    limits: PlannerLimits,
    plan_id: str,
) -> list[dict[str, str]]:
    """Render the Replanner prompt for one round."""
    previous = data.previous_plan
    payload = {
        "task": "replan",
        "plan_identity": {
            "schema_version": "1.0",
            "plan_id": plan_id,
            "parent_plan_id": previous.plan_id,
            "plan_version": previous.plan_version + 1,
            "mode": "replan",
        },
        "original_request": data.user_request,
        "user_memory": data.user_memory,
        "session_history": data.session_history,
        "previous_plan": previous.model_dump(mode="json", exclude_none=False),
        "execution_feedback": data.execution_feedback.model_dump(mode="json"),
        "controller_feedback": data.controller_feedback.model_dump(mode="json"),
        "budget_state": data.budget_state.model_dump(mode="json"),
        "system_capabilities": _capabilities_payload(data.system_capabilities),
        "retrieval_limits": _limits_payload(limits),
        "required_output": "严格按 system prompt 中的 RecommendationPlan JSON 结构输出完整新计划。",
    }
    return [
        {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
        {"role": "user", "content": _json_payload(payload)},
    ]


def build_repair_messages(
    messages: list[dict[str, str]],
    *,
    invalid_output: str,
    error: str,
) -> list[dict[str, str]]:
    """Append the invalid answer plus a bounded correction instruction.

    Only the model's own output and a truncated validator message are echoed; no extra
    system data is injected, so the repair round cannot silently change the request.
    """
    trimmed = invalid_output.strip()
    if len(trimmed) > 4000:
        trimmed = trimmed[:4000] + "...(truncated)"
    return [
        *messages,
        {"role": "assistant", "content": trimmed},
        {
            "role": "user",
            "content": (
                "上面的输出不合法，错误为："
                + error[:1200]
                + "\n请只返回修正后的完整 JSON 对象，不要输出任何解释文字。"
            ),
        },
    ]
