"""The single Planner protocol: :class:`RecommendationPlan`.

Initial Planner and Replanner share this schema (module 2 §6.2, AGENTS.md §2.3).
There is no parallel top-level ``retrieval_strategy`` / ``tiger`` / ``e5_faiss``
field; extra fields are rejected by construction.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from recommender.schemas.common import (
    SCHEMA_VERSION,
    BudgetTier,
    ConstraintOperator,
    ConstraintSource,
    ContractModel,
    NonEmptyStr,
    PlanMode,
    Priority,
    RecallPath,
    RetrievalStrategy,
    SchemaVersion,
)

# Protocol-level absolute limits. Operational (stricter) limits come from configs/.
MAX_E5_QUERIES = 10
MAX_QUERY_CHARS = 500
MAX_HARD_CONSTRAINTS = 20
MAX_SOFT_PREFERENCES = 50
MAX_PREFERENCE_ITEMS = 50
MAX_ASSUMPTIONS = 10
MAX_CHANGE_SUMMARY = 20

_ALLOWED_SCALARS = (str, int, float, bool)


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


class Intent(ContractModel):
    """What the user wants (module 2 §4.1)."""

    intent: NonEmptyStr = "product_recommendation"
    category: str | None = None
    product_type: str | None = None
    usage_scenario: str | None = None

    @field_validator("category", "product_type", "usage_scenario")
    @classmethod
    def _clean_optional_fields(cls, value: str | None) -> str | None:
        return _clean_optional(value)


class HardConstraint(ContractModel):
    """A deterministically checkable constraint (module 2 §4.2).

    The Planner extracts it; only the Python Constraint Evaluator decides whether a
    product satisfies it.
    """

    field: NonEmptyStr
    operator: ConstraintOperator
    value: Any
    unit: str | None = None
    source: ConstraintSource = ConstraintSource.CURRENT_REQUEST

    @field_validator("value")
    @classmethod
    def _value_is_json_scalar_or_list(cls, value: Any) -> Any:
        if value is None or isinstance(value, dict):
            raise ValueError("hard constraint value must be a JSON scalar or a list of scalars")
        if isinstance(value, list):
            if not value:
                raise ValueError("hard constraint list value must not be empty")
            if any(isinstance(item, (dict, list)) or item is None for item in value):
                raise ValueError("hard constraint list values must be scalars")
            return value
        if not isinstance(value, _ALLOWED_SCALARS):
            raise ValueError(f"unsupported hard constraint value type: {type(value).__name__}")
        return value

    @model_validator(mode="after")
    def _operator_value_shape(self) -> Self:
        list_operators = {ConstraintOperator.IN, ConstraintOperator.NOT_IN}
        if self.operator in list_operators and not isinstance(self.value, list):
            raise ValueError(f"operator '{self.operator.value}' requires a list value")
        if self.operator not in list_operators and isinstance(self.value, list):
            raise ValueError(f"operator '{self.operator.value}' requires a scalar value")
        return self

    def key(self) -> str:
        """Stable identity used to verify that a Replanner preserves constraints."""
        return json.dumps(
            {
                "field": self.field.lower(),
                "operator": self.operator.value,
                "value": self.value,
                "unit": self.unit,
            },
            sort_keys=True,
            ensure_ascii=False,
        )


class SoftPreference(ContractModel):
    """A preference that ranks but never filters (module 2 §4.2)."""

    attribute: NonEmptyStr
    source: ConstraintSource = ConstraintSource.CURRENT_REQUEST
    priority: Priority = Priority.MEDIUM


class Constraints(ContractModel):
    hard: list[HardConstraint] = Field(default_factory=list, max_length=MAX_HARD_CONSTRAINTS)
    soft: list[SoftPreference] = Field(default_factory=list, max_length=MAX_SOFT_PREFERENCES)


class Preferences(ContractModel):
    current: list[str] = Field(default_factory=list, max_length=MAX_PREFERENCE_ITEMS)
    historical: list[str] = Field(default_factory=list, max_length=MAX_PREFERENCE_ITEMS)
    negative: list[str] = Field(default_factory=list, max_length=MAX_PREFERENCE_ITEMS)
    priority: list[str] = Field(default_factory=list, max_length=MAX_PREFERENCE_ITEMS)


class TigerRetrievalSpec(ContractModel):
    enabled: bool = True
    candidate_budget: BudgetTier = BudgetTier.STANDARD


class E5FaissRetrievalSpec(ContractModel):
    enabled: bool = True
    queries: list[str] = Field(default_factory=list, max_length=MAX_E5_QUERIES)
    candidate_budget: BudgetTier = BudgetTier.STANDARD

    @field_validator("queries")
    @classmethod
    def _normalize_queries(cls, queries: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for query in queries:
            trimmed = " ".join(query.split())
            if not trimmed:
                continue
            if len(trimmed) > MAX_QUERY_CHARS:
                raise ValueError(
                    f"E5 query exceeds {MAX_QUERY_CHARS} characters: {trimmed[:60]}..."
                )
            key = trimmed.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(trimmed)
        return cleaned


class RetrievalPlan(ContractModel):
    """Which paths run and with which budget tier (module 2 §4.4)."""

    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_BALANCED
    tiger: TigerRetrievalSpec = TigerRetrievalSpec()
    e5_faiss: E5FaissRetrievalSpec = E5FaissRetrievalSpec()

    @model_validator(mode="after")
    def _at_least_one_path(self) -> Self:
        if not self.tiger.enabled and not self.e5_faiss.enabled:
            raise ValueError("at least one retrieval path must be enabled")
        if self.e5_faiss.enabled and not self.e5_faiss.queries:
            raise ValueError("e5_faiss.enabled requires at least one query")
        return self

    def enabled_paths(self) -> list[RecallPath]:
        paths: list[RecallPath] = []
        if self.tiger.enabled:
            paths.append(RecallPath.TIGER)
        if self.e5_faiss.enabled:
            paths.append(RecallPath.E5_FAISS)
        return paths


class RankingGuidance(ContractModel):
    """Reranker query and attributes to prioritise (module 2 §4.5)."""

    rerank_query: NonEmptyStr
    priority_attributes: list[str] = Field(default_factory=list, max_length=MAX_PREFERENCE_ITEMS)


class Clarification(ContractModel):
    required: bool = False
    question: str | None = None
    reason: str | None = None

    @field_validator("question", "reason")
    @classmethod
    def _clean_optional_fields(cls, value: str | None) -> str | None:
        return _clean_optional(value)

    @model_validator(mode="after")
    def _question_iff_required(self) -> Self:
        if self.required and not self.question:
            raise ValueError("clarification.required=true requires a non-empty question")
        if not self.required and (self.question or self.reason):
            raise ValueError("clarification.required=false must not carry question/reason")
        return self


class RecommendationPlan(ContractModel):
    """Unified plan used by Planner, Replanner, Retrieval, Reranker and Controller."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    plan_id: NonEmptyStr
    parent_plan_id: str | None = None
    plan_version: int = Field(default=1, ge=1)
    mode: PlanMode
    intent: Intent
    constraints: Constraints = Constraints()
    preferences: Preferences = Preferences()
    retrieval: RetrievalPlan
    ranking: RankingGuidance
    clarification: Clarification = Clarification()
    assumptions: list[str] = Field(default_factory=list, max_length=MAX_ASSUMPTIONS)
    change_summary: list[str] = Field(default_factory=list, max_length=MAX_CHANGE_SUMMARY)

    @field_validator("parent_plan_id")
    @classmethod
    def _clean_parent(cls, value: str | None) -> str | None:
        return _clean_optional(value)

    @model_validator(mode="after")
    def _mode_invariants(self) -> Self:
        if self.mode is PlanMode.INITIAL:
            if self.parent_plan_id is not None:
                raise ValueError("initial plan must not have parent_plan_id")
            if self.plan_version != 1:
                raise ValueError("initial plan must have plan_version=1")
            if self.change_summary:
                raise ValueError("initial plan must not carry change_summary")
        else:
            if self.parent_plan_id is None:
                raise ValueError("replan must reference parent_plan_id")
            if self.parent_plan_id == self.plan_id:
                raise ValueError("replan parent_plan_id must differ from plan_id")
            if self.plan_version < 2:
                raise ValueError("replan must have plan_version>=2")
        return self


def validate_replan_preserves_hard_constraints(
    parent: RecommendationPlan,
    replan: RecommendationPlan,
) -> None:
    """Fail if a Replanner dropped or rewrote an existing hard constraint.

    Removing a user constraint is a semantic change that only the user can authorise
    (module 2 §8, "Replanner 是否保留原始硬性约束").
    """
    if replan.mode is not PlanMode.REPLAN:
        raise ValueError("validate_replan_preserves_hard_constraints requires mode=replan")
    if replan.parent_plan_id != parent.plan_id:
        raise ValueError(
            f"replan.parent_plan_id={replan.parent_plan_id!r} does not match "
            f"parent.plan_id={parent.plan_id!r}"
        )
    parent_keys = {constraint.key() for constraint in parent.constraints.hard}
    child_keys = {constraint.key() for constraint in replan.constraints.hard}
    missing = parent_keys - child_keys
    if missing:
        raise ValueError(
            "replan dropped or rewrote hard constraints: " + ", ".join(sorted(missing))
        )
