"""DeepSeek per-product structured judgment (batch requests, independent results).

Contract enforcement beyond Pydantic (phase-5 task, module final-stage §3.3):

* exactly one result per input ``product_id`` — missing *or* extra products fail;
* no product outside the input set can be introduced;
* ``attribute_evidence`` may only cite catalog fields that actually exist on that
  product, so a missing attribute can never be upgraded to "supported";
* the Python Constraint Evaluator verdict is read-only context and is never part of the
  model output (``JudgmentLLMOutput`` forbids it as an extra field).

After the bounded repair rounds a batch failure raises :class:`JudgmentBatchError` with a
structured payload. Failed judgments are never reported as ``unknown``/``pass`` by this
module: the caller decides the degradation, and "no result" cannot masquerade as a
successful judgment.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from recommender.api.deepseek import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekJSONError,
    parse_json_content,
)
from recommender.judgment.evidence import (
    CatalogLookup,
    EvidenceBuildResult,
    JudgmentEvidenceBuilder,
)
from recommender.judgment.prompts import (
    JUDGMENT_PROMPT_VERSION,
    build_judgment_messages,
    build_judgment_repair_messages,
)
from recommender.observability.context import get_request_context
from recommender.observability.logging import get_logger
from recommender.observability.trace import TraceWriter
from recommender.schemas.common import ConstraintStatus, EvidenceStatus
from recommender.schemas.judgment import (
    JudgmentBatch,
    JudgmentCandidate,
    JudgmentLLMOutput,
    JudgmentRecord,
)
from recommender.schemas.plan import RecommendationPlan
from recommender.schemas.product import ProductRecord
from recommender.utils.errors import ContractError, RecommenderError

_LOGGER = get_logger(__name__)

DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_ATTEMPTS = 2


class JudgmentBatchError(RecommenderError):
    """Structured failure of one Judgment batch (no result fabricated)."""

    code = "judgment_batch_failed"

    def __init__(
        self,
        message: str,
        *,
        batch_index: int,
        product_ids: Sequence[str],
        attempts: int,
        last_error: str,
        request_id: str,
        trace_id: str,
        error_kind: str = "invalid_output",
        missing_products: Sequence[str] = (),
        extra_products: Sequence[str] = (),
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.batch_index = batch_index
        self.product_ids = list(product_ids)
        self.attempts = attempts
        self.last_error = last_error
        self.request_id = request_id
        self.trace_id = trace_id
        self.error_kind = error_kind
        self.missing_products = list(missing_products)
        self.extra_products = list(extra_products)
        self.model = model

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "degraded": True,
                "judgment_available": False,
                "batch_index": self.batch_index,
                "product_ids": self.product_ids,
                "missing_products": self.missing_products,
                "extra_products": self.extra_products,
                "attempts": self.attempts,
                "error_kind": self.error_kind,
                "last_error": self.last_error,
                "model": self.model,
                "request_id": self.request_id,
                "trace_id": self.trace_id,
            }
        )
        return payload


class _AlignmentError(ValueError):
    """Internal: model returned a different product set than requested."""

    def __init__(self, missing: list[str], extra: list[str]) -> None:
        parts: list[str] = []
        if missing:
            parts.append("missing product judgments: " + ", ".join(missing))
        if extra:
            parts.append("judgments for products not in the input: " + ", ".join(extra))
        super().__init__("; ".join(parts) or "judgment product alignment failed")
        self.missing = missing
        self.extra = extra


@dataclass(frozen=True)
class RankedJudgment:
    """One ``judge_ranked`` call: the evidence used and the resulting judgments."""

    evidence: EvidenceBuildResult
    batch: JudgmentBatch

    @property
    def product_ids(self) -> list[str]:
        return [record.result.product_id for record in self.batch.records]


def _normalize_source(value: str) -> str:
    return value.strip().casefold().replace(" ", "_").replace("-", "_")


def available_evidence_sources(candidate: JudgmentCandidate) -> set[str]:
    """Catalog fields a ``supported``/``contradicted`` evidence claim may cite."""
    available: set[str] = set()
    if candidate.title.strip():
        available.add("title")
    if candidate.description is not None and candidate.description.strip():
        available.add("description")
    if candidate.price is not None:
        available.add("price")
    for key, value in candidate.attributes.items():
        if value is None or not str(value).strip():
            continue
        available.add(_normalize_source(f"attributes.{key}"))
        available.add(_normalize_source(key))
    return available


def _validate_evidence_sources(
    candidate: JudgmentCandidate,
    output: JudgmentLLMOutput,
) -> None:
    available = available_evidence_sources(candidate)
    for attribute, evidence in output.attribute_evidence.items():
        if evidence.status is EvidenceStatus.UNKNOWN:
            continue
        source = evidence.source or ""
        if _normalize_source(source) not in available:
            raise ValueError(
                f"product {candidate.product_id}: attribute '{attribute}' claims "
                f"{evidence.status.value} but source {source!r} does not exist on the product"
            )


def _format_validation_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts: list[str] = []
        for error in exc.errors()[:6]:
            location = ".".join(str(item) for item in error.get("loc", ()))
            message = str(error.get("msg", "invalid value"))
            parts.append(f"{location}: {message}" if location else message)
        return "; ".join(parts)[:800] or "schema validation failed"
    return str(exc)[:800] or type(exc).__name__


def _chunks(items: Sequence[JudgmentCandidate], size: int) -> list[list[JudgmentCandidate]]:
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def _dedupe_ordered(values: list[str]) -> list[str]:
    """Stable de-duplicated ordering for alignment diagnostics."""
    return list(dict.fromkeys(values))


class Judge:
    """Batched DeepSeek judgment producing exactly one record per candidate."""

    def __init__(
        self,
        client: DeepSeekClient,
        *,
        model: str | None = None,
        prompt_version: str = JUDGMENT_PROMPT_VERSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._client = client
        self._model = model or client.model
        self._prompt_version = prompt_version
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._trace = trace

    @property
    def model(self) -> str:
        return self._model

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def batch_count(self, product_count: int) -> int:
        """Number of API calls (batches) needed for ``product_count`` products."""
        if product_count <= 0:
            return 0
        return (product_count + self._batch_size - 1) // self._batch_size

    def judge_ranked(
        self,
        *,
        request_id: str,
        user_request: str,
        plan: RecommendationPlan,
        ranked_product_ids: Sequence[str],
        catalog: Mapping[str, ProductRecord] | None = None,
        catalog_lookup: CatalogLookup | None = None,
        constraint_status_by_id: Mapping[str, ConstraintStatus | None] | None = None,
        evidence_builder: JudgmentEvidenceBuilder | None = None,
    ) -> RankedJudgment:
        """D6 convenience path: enrich Top-N ids from the Catalog, then judge them.

        This is additive: callers that already own a :class:`JudgmentCandidate` view
        keep using :meth:`judge`, whose behaviour is unchanged. Here the ranked
        ``product_id`` list is translated into candidates by
        :class:`JudgmentEvidenceBuilder` (truncated description, relevance-filtered
        details, read-only constraint status) before the same batch contract applies.
        """
        builder = evidence_builder or JudgmentEvidenceBuilder()
        evidence = builder.build(
            ranked_product_ids,
            plan=plan,
            catalog=catalog,
            catalog_lookup=catalog_lookup,
            constraint_status_by_id=constraint_status_by_id,
        )
        if not evidence.candidates:
            missing = ", ".join(evidence.stats.missing_products[:10])
            raise ContractError(
                "no catalog record found for the ranked products"
                + (f": {missing}" if missing else "")
            )
        if evidence.stats.missing_products:
            _LOGGER.warning(
                "judgment evidence missing catalog products",
                extra={
                    "request_id": request_id,
                    "missing_products": evidence.stats.missing_products,
                },
            )
        if self._trace is not None:
            self._trace.emit(
                "judgment.evidence_built",
                {"request_id": request_id, **evidence.stats.to_dict()},
                stage="judgment",
            )
        batch = self.judge(
            request_id=request_id,
            user_request=user_request,
            plan=plan,
            candidates=evidence.candidates,
        )
        return RankedJudgment(evidence=evidence, batch=batch)

    def judge(
        self,
        *,
        request_id: str,
        user_request: str,
        plan: RecommendationPlan,
        candidates: Sequence[JudgmentCandidate],
    ) -> JudgmentBatch:
        """Judge every candidate; raises :class:`JudgmentBatchError` on any batch failure."""
        materialized = list(candidates)
        if not materialized:
            raise ContractError("judgment requires at least one candidate")
        seen: set[str] = set()
        for candidate in materialized:
            if candidate.product_id in seen:
                raise ContractError(
                    f"judgment candidates must be unique by product_id: {candidate.product_id}"
                )
            seen.add(candidate.product_id)

        started = time.perf_counter()
        batches = _chunks(materialized, self._batch_size)
        records: list[JudgmentRecord] = []
        for index, batch_candidates in enumerate(batches, start=1):
            batch_records = self._judge_batch(
                request_id=request_id,
                user_request=user_request,
                plan=plan,
                batch=batch_candidates,
                batch_index=index,
                batch_total=len(batches),
            )
            records.extend(batch_records)
            if self._trace is not None:
                self._trace.emit(
                    "judgment.batch_completed",
                    {
                        "batch_index": index,
                        "batch_total": len(batches),
                        "product_ids": [
                            candidate.product_id for candidate in batch_candidates
                        ],
                        "prompt_version": self._prompt_version,
                        "model": self._model,
                        "results": [
                            record.result.model_dump(mode="json") for record in batch_records
                        ],
                        "latency_ms": round(batch_records[0].latency_ms or 0.0, 1),
                        "input_tokens": batch_records[0].input_tokens,
                        "output_tokens": batch_records[0].output_tokens,
                    },
                    stage="judgment",
                )

        batch = JudgmentBatch(request_id=request_id, records=records)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if self._trace is not None:
            self._trace.emit(
                "judgment.completed",
                {
                    "request_id": request_id,
                    "prompt_version": self._prompt_version,
                    "model": self._model,
                    "batch_count": len(batches),
                    "product_count": len(records),
                    "elapsed_ms": round(elapsed_ms, 1),
                },
                stage="judgment",
            )
        _LOGGER.info(
            "judgment completed",
            extra={
                "request_id": request_id,
                "prompt_version": self._prompt_version,
                "model": self._model,
                "batch_count": len(batches),
                "batch_size": self._batch_size,
                "product_count": len(records),
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )
        return batch

    # -- one batch -----------------------------------------------------------
    def _judge_batch(
        self,
        *,
        request_id: str,
        user_request: str,
        plan: RecommendationPlan,
        batch: list[JudgmentCandidate],
        batch_index: int,
        batch_total: int,
    ) -> list[JudgmentRecord]:
        product_ids = [candidate.product_id for candidate in batch]
        base_messages = build_judgment_messages(
            request_id=request_id,
            user_request=user_request,
            plan=plan,
            candidates=batch,
            batch_index=batch_index,
            batch_total=batch_total,
        )
        last_error = ""
        last_raw = ""
        last_missing: list[str] = []
        last_extra: list[str] = []
        context = get_request_context()
        fallback_trace_id = context.trace_id if context is not None else request_id
        last_identity = (request_id, fallback_trace_id)
        for attempt in range(1, self._max_attempts + 1):
            messages = (
                base_messages
                if attempt == 1
                else build_judgment_repair_messages(
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
                last_identity = (
                    exc.request_id or request_id,
                    exc.trace_id or fallback_trace_id,
                )
                error = JudgmentBatchError(
                    f"DeepSeek API failed while judging batch {batch_index}: {exc.message}",
                    batch_index=batch_index,
                    product_ids=product_ids,
                    attempts=attempt,
                    last_error=exc.message,
                    request_id=last_identity[0],
                    trace_id=last_identity[1],
                    error_kind=exc.kind.value,
                    model=self._model,
                )
                self._emit_failure(error)
                raise error from exc

            last_raw = completion.content
            last_identity = (completion.request_id, completion.trace_id)
            try:
                payload = parse_json_content(
                    completion.content,
                    request_id=completion.request_id,
                    trace_id=completion.trace_id,
                )
                ordered = self._validate_payload(payload, batch)
            except DeepSeekJSONError as exc:
                last_error = exc.message
                self._log_retry(batch_index, attempt, last_error)
                continue
            except _AlignmentError as exc:
                last_missing, last_extra = exc.missing, exc.extra
                last_error = str(exc)
                self._log_retry(batch_index, attempt, last_error)
                continue
            except ValueError as exc:
                last_error = _format_validation_error(exc)
                self._log_retry(batch_index, attempt, last_error)
                continue

            records: list[JudgmentRecord] = []
            for output in ordered:
                records.append(
                    JudgmentRecord(
                        result=output,
                        model=completion.model,
                        prompt_version=self._prompt_version,
                        # Batch-level execution metadata duplicated per record: one API
                        # call produced these judgments, so per-product latency/token
                        # counts do not exist. Sum only across batches.
                        latency_ms=completion.latency_ms,
                        input_tokens=completion.prompt_tokens,
                        output_tokens=completion.completion_tokens,
                        attempts=completion.attempts,
                    )
                )
            if attempt > 1:
                _LOGGER.info(
                    "judgment batch repaired",
                    extra={
                        "batch_index": batch_index,
                        "attempt": attempt,
                        "product_count": len(records),
                    },
                )
            return records

        request_id_value, trace_id_value = last_identity
        error = JudgmentBatchError(
            (
                f"could not obtain valid judgments for batch {batch_index} after "
                f"{self._max_attempts} attempts"
            ),
            batch_index=batch_index,
            product_ids=product_ids,
            attempts=self._max_attempts,
            last_error=last_error or "unknown validation failure",
            request_id=request_id_value,
            trace_id=trace_id_value,
            error_kind="invalid_output",
            missing_products=last_missing,
            extra_products=last_extra,
            model=self._model,
        )
        self._emit_failure(error)
        raise error

    def _validate_payload(
        self,
        payload: dict[str, Any],
        batch: list[JudgmentCandidate],
    ) -> list[JudgmentLLMOutput]:
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("judgment output must contain a non-empty 'items' array")
        outputs: list[JudgmentLLMOutput] = []
        item_errors: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                item_errors.append(f"item is {type(item).__name__}, expected object")
                continue
            try:
                outputs.append(JudgmentLLMOutput.model_validate(item))
            except ValidationError as exc:
                item_errors.append(_format_validation_error(exc))
        if item_errors:
            raise ValueError("invalid judgment items: " + " | ".join(item_errors[:4]))
        ordered = self._align(batch, outputs)
        for candidate, output in zip(batch, ordered, strict=True):
            _validate_evidence_sources(candidate, output)
        return ordered

    @staticmethod
    def _align(
        batch: list[JudgmentCandidate],
        outputs: list[JudgmentLLMOutput],
    ) -> list[JudgmentLLMOutput]:
        by_id: dict[str, JudgmentLLMOutput] = {}
        duplicates: list[str] = []
        for output in outputs:
            if output.product_id in by_id:
                duplicates.append(output.product_id)
                continue
            by_id[output.product_id] = output
        expected = [candidate.product_id for candidate in batch]
        expected_set = set(expected)
        missing = [product_id for product_id in expected if product_id not in by_id]
        extra = [product_id for product_id in by_id if product_id not in expected_set]
        if not duplicates and not missing and not extra:
            return [by_id[product_id] for product_id in expected]
        if duplicates:
            extra = [*extra, *duplicates]
        raise _AlignmentError(missing=missing, extra=_dedupe_ordered(extra))

    def _log_retry(self, batch_index: int, attempt: int, error: str) -> None:
        _LOGGER.warning(
            "judgment output rejected",
            extra={
                "batch_index": batch_index,
                "attempt": attempt,
                "max_attempts": self._max_attempts,
                "prompt_version": self._prompt_version,
                "error": error,
            },
        )

    def _emit_failure(self, error: JudgmentBatchError) -> None:
        _LOGGER.error(
            "judgment batch failed",
            extra={
                "batch_index": error.batch_index,
                "attempts": error.attempts,
                "error_kind": error.error_kind,
                "last_error": error.last_error,
                "missing_products": error.missing_products,
                "extra_products": error.extra_products,
            },
        )
        if self._trace is not None:
            self._trace.emit(
                "judgment.batch_failed",
                error.to_dict(),
                stage="judgment",
            )
