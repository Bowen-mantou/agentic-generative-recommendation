"""Shared enums and base model for all cross-module contracts.

Every cross-module payload is a Pydantic model derived from :class:`ContractModel`.
Unknown fields are rejected: a silently renamed field is a contract violation
(AGENTS.md §2).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

SchemaVersion = Literal["1.0"]
SCHEMA_VERSION: SchemaVersion = "1.0"

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ContractModel(BaseModel):
    """Strict base class for interface payloads (no extra fields allowed)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PlanMode(str, Enum):
    INITIAL = "initial"
    REPLAN = "replan"


class RetrievalStrategy(str, Enum):
    HYBRID_BALANCED = "hybrid_balanced"
    BEHAVIOR_PRIORITY = "behavior_priority"
    SEMANTIC_PRIORITY = "semantic_priority"


class BudgetTier(str, Enum):
    STANDARD = "standard"
    EXPANDED = "expanded"


class RecallPath(str, Enum):
    TIGER = "tiger"
    E5_FAISS = "e5_faiss"


class RetrievalStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ConstraintStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class ConstraintOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


class ConstraintSource(str, Enum):
    CURRENT_REQUEST = "current_request"
    USER_MEMORY = "user_memory"
    HISTORICAL_BEHAVIOR = "historical_behavior"
    SYSTEM_DEFAULT = "system_default"


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PreferenceFitLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    UNKNOWN = "unknown"
    CONTRADICTED = "contradicted"


class MatchAssessment(str, Enum):
    DIRECT_MATCH = "direct_match"
    PARTIAL_MATCH = "partial_match"
    NO_MATCH = "no_match"


class UncertaintyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ControllerAction(str, Enum):
    REPLAN = "replan"
    STOP = "stop"
    CLARIFY = "clarify"


class StopReason(str, Enum):
    QUALITY_SATISFIED = "quality_satisfied"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    NO_VALID_CANDIDATES = "no_valid_candidates"
    RETRIEVAL_DEGRADED = "retrieval_degraded"
    # Decision D4 (2026-11-24): a Planner / Judgment / Generator call that failed after
    # its bounded retries is its own stop reason. It must never be reported as a
    # retrieval degradation, an empty candidate list or a quality pass.
    API_FAILED = "api_failed"


class FailedComponent(str, Enum):
    """Which model-backed component exhausted its retries (decision D4)."""

    PLANNER = "planner"
    JUDGMENT = "judgment"
    GENERATOR = "generator"


class MemoryType(str, Enum):
    LONG_TERM_PREFERENCE = "long_term_preference"
    SHORT_TERM_INTEREST = "short_term_interest"
    EPISODIC = "episodic"


class MemoryPolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    CONFLICTED = "conflicted"


class MemorySourceType(str, Enum):
    AMAZON_REVIEW = "amazon_review"
    SIMULATED_DIALOGUE = "simulated_dialogue"
    EXPLICIT_USER_FEEDBACK = "explicit_user_feedback"
    SYSTEM_INFERENCE = "system_inference"


class MemoryTargetStore(str, Enum):
    MEM0 = "mem0"
    SQLITE_INTERACTION_HISTORY = "sqlite_interaction_history"


class MemoryAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"


class EventType(str, Enum):
    """Interaction event types.

    Amazon Reviews 2023 only yields review/rating proxy signals. Click / favorite /
    purchase are reserved for future first-party telemetry and must not be inferred
    from reviews (AGENTS.md §2.2, module 3 §3.3).
    """

    REVIEW = "review"
    RATING = "rating"
    CLICK = "click"
    FAVORITE = "favorite"
    PURCHASE = "purchase"
    EXPLICIT_POSITIVE_FEEDBACK = "explicit_positive_feedback"
    EXPLICIT_NEGATIVE_FEEDBACK = "explicit_negative_feedback"


class EvalSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
