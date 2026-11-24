"""DeepSeek PFlash query generator (dataset C, protocol §5).

The generator receives only sample metadata, cutoff and profile facts. Target labels are
not part of this module's inputs, and the DTO built here contains no target fields.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from typing import Any

from recommender.config import EvaluationQueryConfig
from recommender.evaluation.client import ChatJSONClient
from recommender.evaluation.profile import profile_to_input
from recommender.evaluation.prompts import (
    GENERATOR_PROMPT_HASH,
    GENERATOR_SYSTEM_PROMPT,
)
from recommender.observability.logging import get_logger
from recommender.schemas.evaluation_dataset import SyntheticQuery, UserProfile

_LOGGER = get_logger(__name__)


def query_id_for(sample_id: str) -> str:
    return "query_" + hashlib.sha256(sample_id.encode()).hexdigest()[:12]


def _completion_metadata(completion: Any, *, attempt: int, latency_ms: float) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "latency_ms": round(latency_ms, 1),
        "request_id": getattr(completion, "request_id", None),
        "api_request_id": getattr(completion, "api_request_id", None),
        "prompt_tokens": getattr(completion, "prompt_tokens", None),
        "completion_tokens": getattr(completion, "completion_tokens", None),
        "total_tokens": getattr(completion, "total_tokens", None),
        "finish_reason": getattr(completion, "finish_reason", None),
    }


class QueryGenerator:
    def __init__(
        self,
        client: ChatJSONClient,
        config: EvaluationQueryConfig,
        *,
        prompt_version: str | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._prompt_version = prompt_version or config.generator_prompt_version

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    @property
    def model(self) -> str:
        return self._client.model

    def generate(self, profile: UserProfile, *, repair_hint: str | None = None) -> SyntheticQuery:
        query_id = query_id_for(profile.sample_id)
        facts = profile.all_facts()
        if not any(
            (
                profile.category_interests,
                profile.brand_interests,
                profile.attribute_interests,
            )
        ):
            return SyntheticQuery(
                query_id=query_id,
                sample_id=profile.sample_id,
                query=None,
                used_fact_ids=[],
                status="insufficient_profile",
                generator_model=self.model,
                prompt_version=self._prompt_version,
                generation_metadata={
                    "reason": "no_positive_profile_facts",
                    "prompt_hash": GENERATOR_PROMPT_HASH,
                },
            )

        payload = {
            "sample_id": profile.sample_id,
            "cutoff_timestamp": profile.cutoff_timestamp,
            "profile": profile_to_input(profile),
        }
        messages: list[dict[str, str]] = [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]

        last_error = ""
        for attempt in range(1, 3):
            started = time.perf_counter()
            try:
                completion = self._client.chat_json(
                    messages,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    reasoning_effort="none",
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                _LOGGER.warning(
                    "query generation API failed",
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
                status = str(data.get("status", "generated"))
                query_text = data.get("query")
                used = data.get("used_fact_ids") or []
                if not isinstance(used, list) or any(not isinstance(item, str) for item in used):
                    raise ValueError("used_fact_ids must be a list of strings")
                if status not in {"generated", "insufficient_profile"}:
                    raise ValueError(f"invalid status: {status}")
                unknown = sorted(set(used) - set(facts))
                if unknown:
                    raise ValueError(f"unknown fact ids: {unknown[:5]}")
                query = SyntheticQuery(
                    query_id=query_id,
                    sample_id=profile.sample_id,
                    query=str(query_text) if query_text else None,
                    used_fact_ids=used,
                    status=status,  # type: ignore[arg-type]
                    generator_model=self.model,
                    prompt_version=self._prompt_version,
                    generation_metadata={
                        "prompt_hash": GENERATOR_PROMPT_HASH,
                        "temperature": self._config.temperature,
                        **_completion_metadata(
                            completion.completion, attempt=attempt, latency_ms=latency_ms
                        ),
                    },
                )
                _LOGGER.info(
                    "query generated",
                    extra={
                        "sample_id": profile.sample_id,
                        "status": query.status,
                        "attempt": attempt,
                    },
                )
                return query
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                messages = [
                    *messages[:2],
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer was invalid for this task: "
                            f"{last_error}. Return only a corrected JSON object with keys "
                            "query, used_fact_ids, status."
                        ),
                    },
                ]

        return SyntheticQuery(
            query_id=query_id,
            sample_id=profile.sample_id,
            query=None,
            used_fact_ids=[],
            status="failed",
            generator_model=self.model,
            prompt_version=self._prompt_version,
            generation_metadata={
                "prompt_hash": GENERATOR_PROMPT_HASH,
                "error": last_error[:500],
            },
        )


class BatchQueryGenerator:
    """Generate/cache queries for a sequence of profiles with resume support."""

    def __init__(self, generator: QueryGenerator) -> None:
        self._generator = generator

    def generate_all(
        self, profiles: Sequence[UserProfile], existing: dict[str, SyntheticQuery] | None = None
    ) -> list[SyntheticQuery]:
        cache = dict(existing or {})
        results: list[SyntheticQuery] = []
        for index, profile in enumerate(profiles, start=1):
            if profile.sample_id in cache:
                results.append(cache[profile.sample_id])
                continue
            result = self._generator.generate(profile)
            cache[result.sample_id] = result
            results.append(result)
            if index % 10 == 0:
                _LOGGER.info("query batch progress", extra={"done": index, "total": len(profiles)})
        return results
