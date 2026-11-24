"""Independent DeepSeek PFlash query auditor (dataset D, protocol §6).

Different prompt and schema from the generator; the auditor sees only profile facts and
the candidate query (not the generator's used_fact_ids or any target label), so its
judgement is not anchored by the generator's explanation.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from recommender.config import EvaluationQueryConfig
from recommender.evaluation.client import ChatJSONClient
from recommender.evaluation.profile import profile_to_input
from recommender.evaluation.prompts import (
    AUDITOR_SYSTEM_PROMPT,
)
from recommender.observability.logging import get_logger
from recommender.schemas.evaluation_dataset import QueryAudit, SyntheticQuery, UserProfile

_LOGGER = get_logger(__name__)


def audit_id_for(query_id: str) -> str:
    return "audit_" + hashlib.sha256(query_id.encode()).hexdigest()[:12]


def _metadata(completion: Any, *, attempt: int, latency_ms: float) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "latency_ms": round(latency_ms, 1),
        "request_id": getattr(completion, "request_id", None),
        "prompt_tokens": getattr(completion, "prompt_tokens", None),
        "completion_tokens": getattr(completion, "completion_tokens", None),
        "finish_reason": getattr(completion, "finish_reason", None),
    }


def _normalise_status(payload: dict[str, Any]) -> str:
    hallucination = bool(payload.get("hallucination"))
    contradiction = bool(payload.get("contradiction"))
    groundedness = str(payload.get("groundedness", "fail"))
    quality = str(payload.get("query_quality", "review"))
    if hallucination or contradiction or quality == "bad":
        return "FAIL"
    if groundedness != "pass" or quality != "good":
        return "REVIEW"
    return "PASS"


class QueryAuditor:
    def __init__(
        self,
        client: ChatJSONClient,
        config: EvaluationQueryConfig,
        *,
        prompt_version: str | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._prompt_version = prompt_version or config.auditor_prompt_version

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    @property
    def model(self) -> str:
        return self._client.model

    def audit(self, profile: UserProfile, query: SyntheticQuery) -> QueryAudit:
        audit_id = audit_id_for(query.query_id)
        if query.status != "generated" or not query.query:
            return QueryAudit(
                audit_id=audit_id,
                query_id=query.query_id,
                sample_id=query.sample_id,
                groundedness="fail",
                hallucination=False,
                contradiction=False,
                query_quality="review",
                audit_status="ERROR",
                reason=f"query status is {query.status}",
                auditor_model=self.model,
                audit_prompt_version=self._prompt_version,
            )

        payload = {
            "sample_id": profile.sample_id,
            "cutoff_timestamp": profile.cutoff_timestamp,
            "profile": profile_to_input(profile),
            "candidate_query": query.query,
        }
        messages: list[dict[str, str]] = [
            {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]

        last_error = ""
        for attempt in range(1, 3):
            started = time.perf_counter()
            try:
                completion = self._client.chat_json(
                    messages,
                    temperature=self._config.temperature,
                    max_tokens=self._config.audit_max_tokens,
                    reasoning_effort="none",
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                _LOGGER.warning(
                    "query audit API failed",
                    extra={
                        "sample_id": profile.sample_id,
                        "attempt": attempt,
                        "error": last_error[:300],
                    },
                )
                continue
            latency_ms = (time.perf_counter() - started) * 1000
            data = completion.data
            try:
                groundedness = str(data.get("groundedness", "fail"))
                quality = str(data.get("query_quality", "review"))
                if groundedness not in {"pass", "fail"}:
                    raise ValueError("groundedness must be pass|fail")
                if quality not in {"good", "review", "bad"}:
                    raise ValueError("query_quality must be good|review|bad")
                unsupported = data.get("unsupported_claims") or []
                supported = data.get("supported_fact_ids") or []
                if not isinstance(unsupported, list) or not all(
                    isinstance(x, str) for x in unsupported
                ):
                    raise ValueError("unsupported_claims must be string list")
                if not isinstance(supported, list) or not all(
                    isinstance(x, str) for x in supported
                ):
                    raise ValueError("supported_fact_ids must be string list")
                raw_status = str(data.get("audit_status", ""))
                if raw_status not in {"PASS", "REVIEW", "FAIL"}:
                    raise ValueError("audit_status must be PASS|REVIEW|FAIL")
                normalised = _normalise_status(data)
                record = QueryAudit(
                    audit_id=audit_id,
                    query_id=query.query_id,
                    sample_id=query.sample_id,
                    groundedness=groundedness,  # type: ignore[arg-type]
                    hallucination=bool(data.get("hallucination")),
                    contradiction=bool(data.get("contradiction")),
                    query_quality=quality,  # type: ignore[arg-type]
                    unsupported_claims=unsupported,
                    supported_fact_ids=supported,
                    audit_status=normalised,  # type: ignore[arg-type]
                    reason=str(data.get("reason", ""))[:500],
                    auditor_model=self.model,
                    audit_prompt_version=self._prompt_version,
                    audit_metadata={
                        "raw_status": raw_status,
                        **_metadata(completion.completion, attempt=attempt, latency_ms=latency_ms),
                    },
                )
                _LOGGER.info(
                    "query audited",
                    extra={
                        "sample_id": profile.sample_id,
                        "status": record.audit_status,
                        "attempt": attempt,
                    },
                )
                return record
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                messages = [
                    *messages[:2],
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer was invalid: "
                            f"{last_error}. Return only corrected JSON with the required keys."
                        ),
                    },
                ]

        return QueryAudit(
            audit_id=audit_id,
            query_id=query.query_id,
            sample_id=query.sample_id,
            groundedness="fail",
            hallucination=False,
            contradiction=False,
            query_quality="review",
            audit_status="ERROR",
            reason=f"audit failed after retries: {last_error[:400]}",
            auditor_model=self.model,
            audit_prompt_version=self._prompt_version,
        )


class BatchQueryAuditor:
    def __init__(self, auditor: QueryAuditor) -> None:
        self._auditor = auditor

    def audit_all(
        self,
        profiles: dict[str, UserProfile],
        queries: list[SyntheticQuery],
        existing: dict[str, QueryAudit] | None = None,
    ) -> list[QueryAudit]:
        cache = dict(existing or {})
        results: list[QueryAudit] = []
        for index, query in enumerate(queries, start=1):
            if query.sample_id in cache:
                results.append(cache[query.sample_id])
                continue
            profile = profiles[query.sample_id]
            record = self._auditor.audit(profile, query)
            cache[query.sample_id] = record
            results.append(record)
            if index % 10 == 0:
                _LOGGER.info("audit batch progress", extra={"done": index, "total": len(queries)})
        return results
