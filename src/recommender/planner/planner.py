"""Initial Planner / Replanner built on the shared DeepSeek client.

Validation policy (module 2 §8, final-stage doc §9.2):

* every answer is parsed as JSON and validated against :class:`RecommendationPlan`;
* a parse/schema/capability failure triggers a bounded repair round (the invalid answer
  plus the validator message are sent back once);
* when repair still fails the Planner raises :class:`PlanGenerationError` with a
  structured payload. It never fabricates a "default plan": a failed replan must not be
  mistaken for a successful one.

Decision D7 (2026-11-24) adds an **explicitly configurable** degradation:
``planner.fallback.enabled`` defaults to ``false`` (structured failure). When an
operator enables it, an exhausted Planner call returns a deterministic, balanced
system-default plan. That plan is marked ``degraded=True`` / ``plan_source=
system_default`` on :class:`PlannerResult`, in its ``assumptions`` and in the trace, so
it can never be mistaken for a Planner API success.

The Replanner reuses :func:`validate_replan_preserves_hard_constraints` so a model can
never drop or rewrite a user's hard constraint.
"""

from __future__ import annotations

import hashlib
import time
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from recommender.api.deepseek import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekJSONError,
    parse_json_content,
)
from recommender.config import PlannerFallbackConfig
from recommender.observability.context import get_request_context, new_request_context
from recommender.observability.logging import get_logger
from recommender.observability.trace import TraceWriter
from recommender.planner.prompts import (
    INITIAL_PROMPT_VERSION,
    REPLAN_PROMPT_VERSION,
    PlannerInput,
    PlannerLimits,
    ReplanInput,
    SystemCapabilities,
    build_initial_messages,
    build_repair_messages,
    build_replan_messages,
)
from recommender.schemas.common import (
    SCHEMA_VERSION,
    PlanMode,
    RecallPath,
    RetrievalStrategy,
)
from recommender.schemas.plan import (
    RecommendationPlan,
    validate_replan_preserves_hard_constraints,
)
from recommender.utils.errors import RecommenderError

_LOGGER = get_logger(__name__)

DEFAULT_MAX_ATTEMPTS = 2
_UNKNOWN_FIELD_PLACEHOLDER = object()

# Assumptions written into every deterministic fallback plan (decision D7). They are
# the plan-level audit marker: RecommendationPlan has no free-form ``degraded`` field
# (schemas/plan.py is outside this task's scope), so the degradation is recorded in
# ``assumptions`` plus ``PlannerResult.degraded``/``plan_source`` and trace events.
FALLBACK_ASSUMPTIONS: tuple[str, ...] = (
    "Planner API unavailable",
    "Using deterministic balanced retrieval",
    "degraded: plan_source=system_default",
)
_FALLBACK_SUFFIXES: tuple[str, ...] = (
    "alternatives",
    "alternative options",
    "other options",
    "different results",
)


class PlanGenerationError(RecommenderError):
    """Explicit, structured failure of Initial Planner / Replanner.

    ``to_dict()`` is the serializable degradation record for traces. There is no
    ``plan`` attribute: a failed generation has no plan, by construction.
    """

    code = "plan_generation_failed"

    def __init__(
        self,
        message: str,
        *,
        mode: PlanMode,
        attempts: int,
        prompt_version: str,
        last_error: str,
        request_id: str,
        trace_id: str,
        error_kind: str = "invalid_output",
        model: str | None = None,
        raw_output: str | None = None,
    ) -> None:
        super().__init__(message)
        self.mode = mode
        self.attempts = attempts
        self.prompt_version = prompt_version
        self.last_error = last_error
        self.request_id = request_id
        self.trace_id = trace_id
        self.error_kind = error_kind
        self.model = model
        self.raw_output = raw_output

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "degraded": True,
                "plan_generated": False,
                "mode": self.mode.value,
                "attempts": self.attempts,
                "prompt_version": self.prompt_version,
                "error_kind": self.error_kind,
                "last_error": self.last_error,
                "model": self.model,
                "request_id": self.request_id,
                "trace_id": self.trace_id,
            }
        )
        return payload


class PlanFallbackError(RecommenderError):
    """The explicit fallback was enabled but a deterministic plan could not be built.

    Raised (chained from the original :class:`PlanGenerationError`) when the system has
    no retrieval path to plan into. Never silently downgraded to an empty plan.
    """

    code = "plan_fallback_failed"


@dataclass(frozen=True)
class PlannerResult:
    """A validated plan plus the metadata needed for tracing and cost accounting."""

    plan: RecommendationPlan
    prompt_version: str
    model: str
    attempts: int
    latency_ms: float
    request_id: str
    trace_id: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    repaired: bool = False
    identity_overrides: list[str] = field(default_factory=list)
    # Decision D7: True only for an explicitly enabled system-default fallback plan.
    degraded: bool = False
    plan_source: str | None = None


def _new_plan_id() -> str:
    return f"plan_{uuid.uuid4().hex[:12]}"


def _fallback_plan_id(prefix: str, seed: str) -> str:
    digest = hashlib.sha256(f"{prefix}\x00{seed}".encode()).hexdigest()[:12]
    return f"plan_fallback_{digest}"


def _normalize_field_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_query(value: str) -> str:
    """Canonical query text (decision D5: NFKC + whitespace collapse + casefold)."""
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _truncate_query(value: str, max_chars: int) -> str:
    cleaned = " ".join(value.split())
    if not cleaned:
        cleaned = "product recommendation"
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].strip() or cleaned[:max_chars]


def _format_validation_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts: list[str] = []
        for error in exc.errors()[:6]:
            location = ".".join(str(item) for item in error.get("loc", ()))
            message = str(error.get("msg", "invalid value"))
            parts.append(f"{location}: {message}" if location else message)
        return "; ".join(parts)[:800] or "schema validation failed"
    return str(exc)[:800] or type(exc).__name__


class Planner:
    """DeepSeek-backed Initial Planner and Replanner (shared schema, two prompts)."""

    def __init__(
        self,
        client: DeepSeekClient,
        *,
        model: str | None = None,
        limits: PlannerLimits | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: TraceWriter | None = None,
        fallback: PlannerFallbackConfig | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._client = client
        self._model = model or client.model
        self._limits = limits or PlannerLimits()
        self._max_attempts = max_attempts
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._trace = trace
        # Decision D7 default: structured failure, never a fabricated plan.
        self._fallback = fallback or PlannerFallbackConfig()

    @property
    def model(self) -> str:
        return self._model

    @property
    def limits(self) -> PlannerLimits:
        return self._limits

    @property
    def fallback(self) -> PlannerFallbackConfig:
        return self._fallback

    @property
    def prompt_versions(self) -> dict[str, str]:
        return {
            PlanMode.INITIAL.value: INITIAL_PROMPT_VERSION,
            PlanMode.REPLAN.value: REPLAN_PROMPT_VERSION,
        }

    # -- public entry points -------------------------------------------------
    def generate(
        self,
        mode: PlanMode | str,
        input_data: PlannerInput | ReplanInput,
    ) -> PlannerResult:
        """Unified Planner interface (module 2 §6.1)."""
        resolved = mode if isinstance(mode, PlanMode) else PlanMode(mode)
        if resolved is PlanMode.INITIAL:
            if not isinstance(input_data, PlannerInput):
                raise ValueError("initial mode requires PlannerInput")
            return self.plan_initial(input_data)
        if not isinstance(input_data, ReplanInput):
            raise ValueError("replan mode requires ReplanInput")
        return self.plan_replan(input_data)

    def plan_initial(self, data: PlannerInput) -> PlannerResult:
        """Generate and validate the initial RecommendationPlan."""
        plan_id = data.plan_id or _new_plan_id()
        base_messages = build_initial_messages(data, limits=self._limits, plan_id=plan_id)

        def validate(payload: dict[str, Any]) -> tuple[RecommendationPlan, list[str]]:
            overrides = self._apply_identity(
                payload,
                mode=PlanMode.INITIAL,
                plan_id=plan_id,
                parent_plan_id=None,
                plan_version=1,
            )
            plan = RecommendationPlan.model_validate(payload)
            self._validate_capabilities(plan, data.system_capabilities)
            self._validate_limits(plan)
            return plan, overrides

        return self._run(
            mode=PlanMode.INITIAL,
            prompt_version=INITIAL_PROMPT_VERSION,
            base_messages=base_messages,
            validate=validate,
            fallback_builder=(
                lambda: self._build_fallback_initial(data, plan_id=data.plan_id)
            ),
        )

    def plan_replan(self, data: ReplanInput) -> PlannerResult:
        """Generate and validate a Replan that preserves the parent hard constraints."""
        plan_id = data.plan_id or _new_plan_id()
        parent = data.previous_plan
        base_messages = build_replan_messages(data, limits=self._limits, plan_id=plan_id)

        def validate(payload: dict[str, Any]) -> tuple[RecommendationPlan, list[str]]:
            overrides = self._apply_identity(
                payload,
                mode=PlanMode.REPLAN,
                plan_id=plan_id,
                parent_plan_id=parent.plan_id,
                plan_version=parent.plan_version + 1,
            )
            plan = RecommendationPlan.model_validate(payload)
            validate_replan_preserves_hard_constraints(parent, plan)
            self._validate_capabilities(plan, data.system_capabilities)
            self._validate_limits(plan)
            self._validate_replan_changes_queries(plan, parent)
            if not plan.change_summary:
                raise ValueError("replan.change_summary must contain at least one entry")
            return plan, overrides

        return self._run(
            mode=PlanMode.REPLAN,
            prompt_version=REPLAN_PROMPT_VERSION,
            base_messages=base_messages,
            validate=validate,
            fallback_builder=(
                lambda: self._build_fallback_replan(data, plan_id=data.plan_id)
            ),
        )

    # -- generation loop -----------------------------------------------------
    def _run(
        self,
        *,
        mode: PlanMode,
        prompt_version: str,
        base_messages: list[dict[str, str]],
        validate: Callable[[dict[str, Any]], tuple[RecommendationPlan, list[str]]],
        fallback_builder: Callable[[], RecommendationPlan] | None = None,
    ) -> PlannerResult:
        started = time.perf_counter()
        context = get_request_context() or new_request_context()
        last_error = ""
        last_raw = ""
        last_identity: tuple[str, str] = (context.request_id, context.trace_id)
        for attempt in range(1, self._max_attempts + 1):
            messages = (
                base_messages
                if attempt == 1
                else build_repair_messages(
                    base_messages,
                    invalid_output=last_raw,
                    error=last_error,
                )
            )
            try:
                completion = self._client.chat(
                    messages,
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            except DeepSeekAPIError as exc:
                error = PlanGenerationError(
                    f"DeepSeek API failed while generating the {mode.value} plan: {exc.message}",
                    mode=mode,
                    attempts=attempt,
                    prompt_version=prompt_version,
                    last_error=exc.message,
                    request_id=exc.request_id or context.request_id,
                    trace_id=exc.trace_id or context.trace_id,
                    error_kind=exc.kind.value,
                    model=self._model,
                )
                return self._handle_failure(
                    error,
                    cause=exc,
                    fallback_builder=fallback_builder,
                    prompt_version=prompt_version,
                    started=started,
                    attempts=attempt,
                )

            last_raw = completion.content
            last_identity = (completion.request_id, completion.trace_id)
            try:
                payload = parse_json_content(
                    completion.content,
                    request_id=completion.request_id,
                    trace_id=completion.trace_id,
                )
            except DeepSeekJSONError as exc:
                last_error = exc.message
                _LOGGER.warning(
                    "planner output is not valid JSON",
                    extra={
                        "mode": mode.value,
                        "attempt": attempt,
                        "max_attempts": self._max_attempts,
                        "prompt_version": prompt_version,
                        "error": last_error,
                    },
                )
                continue
            try:
                plan, identity_overrides = validate(payload)
            except (ValidationError, ValueError) as exc:
                last_error = _format_validation_error(exc)
                _LOGGER.warning(
                    "planner output failed validation",
                    extra={
                        "mode": mode.value,
                        "attempt": attempt,
                        "max_attempts": self._max_attempts,
                        "prompt_version": prompt_version,
                        "error": last_error,
                    },
                )
                continue

            latency_ms = (time.perf_counter() - started) * 1000.0
            result = PlannerResult(
                plan=plan,
                prompt_version=prompt_version,
                model=completion.model,
                attempts=attempt,
                latency_ms=latency_ms,
                request_id=completion.request_id,
                trace_id=completion.trace_id,
                input_tokens=completion.prompt_tokens,
                output_tokens=completion.completion_tokens,
                repaired=attempt > 1,
                identity_overrides=identity_overrides,
            )
            if identity_overrides:
                _LOGGER.warning(
                    "planner identity fields were enforced by the orchestrator",
                    extra={
                        "mode": mode.value,
                        "plan_id": plan.plan_id,
                        "overrides": identity_overrides,
                    },
                )
            if self._trace is not None:
                self._trace.emit(
                    "planner.plan_generated",
                    {
                        "mode": mode.value,
                        "plan_id": plan.plan_id,
                        "parent_plan_id": plan.parent_plan_id,
                        "plan_version": plan.plan_version,
                        "plan": plan.model_dump(mode="json"),
                        "prompt_version": prompt_version,
                        "model": completion.model,
                        "attempts": attempt,
                        "repaired": result.repaired,
                        "latency_ms": round(latency_ms, 1),
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "identity_overrides": identity_overrides,
                    },
                    stage="planner",
                )
            _LOGGER.info(
                "planner plan generated",
                extra={
                    "mode": mode.value,
                    "plan_id": plan.plan_id,
                    "plan_version": plan.plan_version,
                    "attempts": attempt,
                    "repaired": result.repaired,
                    "prompt_version": prompt_version,
                    "model": completion.model,
                    "latency_ms": round(latency_ms, 1),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "e5_queries": plan.retrieval.e5_faiss.queries,
                    "hard_constraints": len(plan.constraints.hard),
                },
            )
            return result

        request_id, trace_id = last_identity
        error = PlanGenerationError(
            (
                f"could not obtain a valid {mode.value} plan after {self._max_attempts} "
                "attempts; no plan was fabricated"
            ),
            mode=mode,
            attempts=self._max_attempts,
            prompt_version=prompt_version,
            last_error=last_error or "unknown validation failure",
            request_id=request_id,
            trace_id=trace_id,
            error_kind="invalid_output",
            model=self._model,
            raw_output=last_raw[:2000],
        )
        return self._handle_failure(
            error,
            fallback_builder=fallback_builder,
            prompt_version=prompt_version,
            started=started,
            attempts=self._max_attempts,
        )

    # -- explicit fallback (decision D7) ------------------------------------
    def _handle_failure(
        self,
        error: PlanGenerationError,
        *,
        fallback_builder: Callable[[], RecommendationPlan] | None,
        prompt_version: str,
        started: float,
        attempts: int,
        cause: BaseException | None = None,
    ) -> PlannerResult:
        """Return a marked fallback plan or re-raise the structured error.

        The default (``planner.fallback.enabled=false``) is a structured failure: the
        error is emitted and re-raised. Only an explicit opt-in produces a
        deterministic system-default plan, and that plan is always marked degraded.
        """
        if not self._fallback.enabled or fallback_builder is None:
            self._emit_failure(error)
            if cause is not None:
                raise error from cause
            raise error
        try:
            plan = fallback_builder()
        except PlanFallbackError as fallback_error:
            self._emit_failure(error)
            raise fallback_error from error
        result = PlannerResult(
            plan=plan,
            prompt_version=prompt_version,
            model=self._model,
            attempts=attempts,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            request_id=error.request_id,
            trace_id=error.trace_id,
            degraded=True,
            plan_source="system_default",
        )
        self._emit_degraded(error, result)
        return result

    def _build_fallback_initial(
        self,
        data: PlannerInput,
        *,
        plan_id: str | None,
    ) -> RecommendationPlan:
        modules = set(data.system_capabilities.retrieval_modules)
        tiger_enabled = RecallPath.TIGER in modules
        e5_enabled = RecallPath.E5_FAISS in modules
        if not tiger_enabled and not e5_enabled:
            raise PlanFallbackError(
                "cannot build a deterministic fallback plan: the system exposes no "
                "retrieval module"
            )
        query = _truncate_query(data.user_request, self._limits.max_query_chars)
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "plan_id": plan_id or _fallback_plan_id("initial", data.user_request),
            "parent_plan_id": None,
            "plan_version": 1,
            "mode": PlanMode.INITIAL.value,
            "intent": {"intent": "product_recommendation"},
            "constraints": {"hard": [], "soft": []},
            "preferences": {"current": [], "historical": [], "negative": [], "priority": []},
            "retrieval": {
                "strategy": RetrievalStrategy.HYBRID_BALANCED.value,
                "tiger": {"enabled": tiger_enabled, "candidate_budget": "standard"},
                "e5_faiss": {
                    "enabled": e5_enabled,
                    "queries": [query] if e5_enabled else [],
                    "candidate_budget": "standard",
                },
            },
            "ranking": {"rerank_query": query, "priority_attributes": []},
            "clarification": {"required": False},
            "assumptions": [
                *FALLBACK_ASSUMPTIONS,
                "Hard constraints were not extracted by the Planner",
            ],
            "change_summary": [],
        }
        return self._validated_fallback_plan(payload, data.system_capabilities)

    def _build_fallback_replan(
        self,
        data: ReplanInput,
        *,
        plan_id: str | None,
    ) -> RecommendationPlan:
        parent = data.previous_plan
        modules = set(data.system_capabilities.retrieval_modules)
        tiger_enabled = RecallPath.TIGER in modules
        e5_enabled = RecallPath.E5_FAISS in modules
        if not tiger_enabled and not e5_enabled:
            raise PlanFallbackError(
                "cannot build a deterministic fallback replan: the system exposes no "
                "retrieval module"
            )
        query = self._fallback_replan_query(data.user_request, parent)
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "plan_id": plan_id
            or _fallback_plan_id("replan", f"{parent.plan_id}\x00{data.user_request}"),
            "parent_plan_id": parent.plan_id,
            "plan_version": parent.plan_version + 1,
            "mode": PlanMode.REPLAN.value,
            # A deterministic fallback never edits the user's plan semantics: intent,
            # hard constraints and preferences are copied from the validated parent.
            "intent": parent.intent.model_dump(mode="json"),
            "constraints": parent.constraints.model_dump(mode="json"),
            "preferences": parent.preferences.model_dump(mode="json"),
            "retrieval": {
                "strategy": RetrievalStrategy.HYBRID_BALANCED.value,
                "tiger": {"enabled": tiger_enabled, "candidate_budget": "standard"},
                "e5_faiss": {
                    "enabled": e5_enabled,
                    "queries": [query] if e5_enabled else [],
                    "candidate_budget": "standard",
                },
            },
            "ranking": {
                "rerank_query": _truncate_query(
                    data.user_request, self._limits.max_query_chars
                ),
                "priority_attributes": list(parent.ranking.priority_attributes),
            },
            "clarification": {"required": False},
            "assumptions": [
                *FALLBACK_ASSUMPTIONS,
                "Previous plan's hard constraints and preferences are preserved unchanged",
            ],
            "change_summary": [
                "deterministic fallback: retry balanced retrieval with a new query"
            ],
        }
        plan = self._validated_fallback_plan(payload, data.system_capabilities)
        validate_replan_preserves_hard_constraints(parent, plan)
        self._validate_replan_changes_queries(plan, parent)
        return plan

    def _validated_fallback_plan(
        self,
        payload: dict[str, Any],
        capabilities: SystemCapabilities,
    ) -> RecommendationPlan:
        try:
            plan = RecommendationPlan.model_validate(payload)
            self._validate_capabilities(plan, capabilities)
            self._validate_limits(plan)
        except (ValidationError, ValueError) as exc:
            raise PlanFallbackError(
                f"deterministic fallback plan is invalid: {_format_validation_error(exc)}"
            ) from exc
        return plan

    def _fallback_replan_query(
        self,
        user_request: str,
        parent: RecommendationPlan,
    ) -> str:
        parent_queries = {
            _normalize_query(query) for query in parent.retrieval.e5_faiss.queries
        }
        base = _truncate_query(
            user_request, max(1, self._limits.max_query_chars - 24)
        )
        for suffix in _FALLBACK_SUFFIXES:
            candidate = _truncate_query(
                f"{base} {suffix}", self._limits.max_query_chars
            )
            if _normalize_query(candidate) not in parent_queries:
                return candidate
        raise PlanFallbackError(
            "cannot build a deterministic fallback replan: every fallback query repeats "
            "a query from the previous plan"
        )

    # -- validation helpers --------------------------------------------------
    @staticmethod
    def _apply_identity(
        payload: dict[str, Any],
        *,
        mode: PlanMode,
        plan_id: str,
        parent_plan_id: str | None,
        plan_version: int,
    ) -> list[str]:
        """Assert system-owned identity fields, returning what had to change.

        The model is instructed to copy these values from the prompt; enforcing them
        here prevents a model typo from producing a Replan that claims to be an Initial
        plan (or vice versa) while keeping the semantic content model-authored.
        """
        expected: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "mode": mode.value,
            "plan_id": plan_id,
            "parent_plan_id": parent_plan_id,
            "plan_version": plan_version,
        }
        overrides: list[str] = []
        for key, value in expected.items():
            current = payload.get(key, _UNKNOWN_FIELD_PLACEHOLDER)
            if current != value:
                overrides.append(f"{key}: model={current!r} -> enforced={value!r}")
            payload[key] = value
        return overrides

    def _validate_capabilities(
        self,
        plan: RecommendationPlan,
        capabilities: SystemCapabilities,
    ) -> None:
        available = set(capabilities.retrieval_modules)
        missing_paths = [path for path in plan.retrieval.enabled_paths() if path not in available]
        if missing_paths:
            names = ", ".join(path.value for path in missing_paths)
            raise ValueError(f"plan enables retrieval modules the system does not provide: {names}")
        allowed_fields = {
            _normalize_field_name(name) for name in capabilities.available_product_fields
        }
        unknown_fields = sorted(
            {
                constraint.field
                for constraint in plan.constraints.hard
                if _normalize_field_name(constraint.field) not in allowed_fields
            }
        )
        if unknown_fields:
            raise ValueError(
                "hard constraints reference fields outside system capabilities: "
                + ", ".join(unknown_fields)
            )

    def _validate_limits(self, plan: RecommendationPlan) -> None:
        queries = plan.retrieval.e5_faiss.queries
        if plan.retrieval.e5_faiss.enabled:
            if len(queries) > self._limits.max_e5_queries:
                raise ValueError(
                    f"plan has {len(queries)} E5 queries; operational maximum is "
                    f"{self._limits.max_e5_queries}"
                )
            too_long = [query for query in queries if len(query) > self._limits.max_query_chars]
            if too_long:
                raise ValueError(
                    f"E5 queries exceed {self._limits.max_query_chars} characters: "
                    + ", ".join(query[:40] for query in too_long)
                )

    @staticmethod
    def _validate_replan_changes_queries(
        plan: RecommendationPlan,
        parent: RecommendationPlan,
    ) -> None:
        new_queries = {_normalize_query(query) for query in plan.retrieval.e5_faiss.queries}
        old_queries = {_normalize_query(query) for query in parent.retrieval.e5_faiss.queries}
        if old_queries and new_queries.issubset(old_queries):
            raise ValueError(
                "replan repeats only the previous E5 queries; generate at least one new query"
            )

    # -- trace ---------------------------------------------------------------
    def _emit_failure(self, error: PlanGenerationError) -> None:
        _LOGGER.error(
            "planner plan generation failed",
            extra={
                "mode": error.mode.value,
                "attempts": error.attempts,
                "prompt_version": error.prompt_version,
                "error_kind": error.error_kind,
                "last_error": error.last_error,
            },
        )
        if self._trace is not None:
            self._trace.emit(
                "planner.plan_failed",
                error.to_dict(),
                stage="planner",
            )

    def _emit_degraded(
        self,
        error: PlanGenerationError,
        result: PlannerResult,
    ) -> None:
        payload: dict[str, Any] = {
            "degraded": True,
            "plan_source": result.plan_source,
            "plan_generated": True,
            "mode": result.plan.mode.value,
            "plan_id": result.plan.plan_id,
            "assumptions": result.plan.assumptions,
            "attempts": result.attempts,
            "prompt_version": result.prompt_version,
            "error": error.to_dict(),
        }
        _LOGGER.warning(
            "planner degraded to a deterministic system-default plan",
            extra={
                "mode": result.plan.mode.value,
                "plan_id": result.plan.plan_id,
                "plan_source": result.plan_source,
                "error_kind": error.error_kind,
                "attempts": result.attempts,
            },
        )
        if self._trace is not None:
            self._trace.emit("planner.plan_degraded", payload, stage="planner")


def planner_capabilities(
    modules: list[RecallPath],
    *,
    available_product_fields: list[str] | None = None,
) -> SystemCapabilities:
    """Small helper for CLI/Phase 5 integration (keeps capability wiring explicit)."""
    if available_product_fields is None:
        return SystemCapabilities(retrieval_modules=modules)
    return SystemCapabilities(
        retrieval_modules=modules,
        available_product_fields=available_product_fields,
    )
