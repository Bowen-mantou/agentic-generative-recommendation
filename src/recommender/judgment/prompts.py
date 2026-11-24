"""Versioned Judgment prompt (DeepSeek) and message builder.

Judgment answers one question per product: how well does the catalog evidence support
this product for the current request? It never re-checks numeric hard constraints and
never decides REPLAN/STOP (final-stage doc §3.7). The prompt therefore *passes in* the
Python Constraint Evaluator verdict as read-only context and asks for one aligned result
per candidate.
"""

from __future__ import annotations

import json
from typing import Any

from recommender.schemas.judgment import JudgmentCandidate
from recommender.schemas.plan import RecommendationPlan

# The literal prompt sentences are intentionally long Chinese lines; artificial wrapping
# inside the prompt would change the text sent to the model.
# ruff: noqa: E501

JUDGMENT_PROMPT_VERSION = "judgment.v1"

OUTPUT_SCHEMA_EXAMPLE = """{
  "items": [
    {
      "product_id": "<原样复制输入商品的 product_id>",
      "preference_fit": {
        "level": "high",
        "matched_preferences": ["lightweight"],
        "unmatched_preferences": [],
        "unknown_preferences": ["fragrance_free"]
      },
      "attribute_evidence": {
        "lightweight": {"status": "supported", "source": "description"},
        "fragrance_free": {"status": "unknown", "source": null}
      },
      "match_assessment": "partial_match",
      "uncertainty": "medium",
      "warnings": ["商品未说明是否无香型"]
    }
  ]
}"""

JUDGMENT_SYSTEM_PROMPT = (
    """你是 Agentic Generative Recommendation 系统的 Judgment Model。
Prompt 版本：judgment.v1。

你的唯一任务：针对输入中的每一个候选商品，独立输出一条结构化判断，说明它与用户当前
需求、用户偏好和硬约束（已由 Python 校验）之间的匹配程度与证据充分性。

输出必须是一个 JSON 对象，且严格符合以下结构（字段名不得改动，禁止新增字段）：
"""
    + OUTPUT_SCHEMA_EXAMPLE
    + """

规则（违反任何一条都会被视为无效输出并触发重试）：
1. 每个输入商品必须恰好有一条判断结果，不得遗漏；不得新增输入中不存在的商品；product_id 必须原样复制。
2. 只依据输入商品中实际提供的字段（title、price、description、attributes）判断。缺失的字段就是未知，必须使用 unknown，不得推断、补造或假设其满足。
3. attribute_evidence.status 取 supported/unknown/contradicted：
   - supported/contradicted 时 source 必须引用该商品真实存在的字段：title、description、price，或 attributes.<key>（key 必须存在于该商品的 attributes 中）；
   - unknown 时 source 必须为 null。
4. attribute_evidence 的键应当是本次需要核实的偏好或属性（来自 recommendation_plan 的软偏好、偏好列表或品类关键属性）。
5. 不得输出、修改或推断 constraint_status。价格/品类等硬约束已由 Python Constraint Evaluator 判定，仅作为只读上下文；你不得重新比较数值，也不得把 unknown 当作 pass。
6. preference_fit.matched_preferences / unmatched_preferences / unknown_preferences 三个集合互不重叠，元素应来自 recommendation_plan 中出现的偏好字符串。
7. preference_fit.level 取 high/medium/low；match_assessment 取 direct_match/partial_match/no_match；uncertainty 取 low/medium/high（表示商品信息是否充分）。
8. warnings 用简短中文说明缺失证据或不确定点；没有则为空数组。
9. 只输出 JSON 对象：禁止 markdown 代码块、注释和任何解释文字。"""
)


def _json_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _plan_context(plan: RecommendationPlan) -> dict[str, Any]:
    return {
        "intent": plan.intent.model_dump(mode="json"),
        "constraints": plan.constraints.model_dump(mode="json"),
        "preferences": plan.preferences.model_dump(mode="json"),
        "ranking": plan.ranking.model_dump(mode="json"),
    }


def build_judgment_messages(
    *,
    request_id: str,
    user_request: str,
    plan: RecommendationPlan,
    candidates: list[JudgmentCandidate],
    batch_index: int,
    batch_total: int,
) -> list[dict[str, str]]:
    """Render one batched Judgment request covering ``candidates``."""
    payload = {
        "task": "judge_products",
        "request_id": request_id,
        "batch": {"index": batch_index, "total": batch_total, "size": len(candidates)},
        "current_request": user_request,
        "recommendation_plan": _plan_context(plan),
        "candidates": [
            candidate.model_dump(mode="json", exclude_none=False) for candidate in candidates
        ],
        "output_schema": OUTPUT_SCHEMA_EXAMPLE,
        "required_output": "严格按 output_schema 输出 items 数组，每个输入商品恰好一条。",
    }
    return [
        {"role": "system", "content": JUDGMENT_SYSTEM_PROMPT},
        {"role": "user", "content": _json_payload(payload)},
    ]


def build_judgment_repair_messages(
    messages: list[dict[str, str]],
    *,
    invalid_output: str,
    error: str,
) -> list[dict[str, str]]:
    """Bounded repair round: echo the invalid answer plus the validator message."""
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
                + "\n请只返回修正后的完整 JSON 对象（items 数组），不要输出任何解释文字。"
            ),
        },
    ]
