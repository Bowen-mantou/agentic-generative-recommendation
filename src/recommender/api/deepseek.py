"""DeepSeek chat-completions client shared by Planner, Judgment and (later) Generator.

Requirements implemented here (AGENTS.md §5, final-stage doc §9.1):

* finite timeout, finite retries with exponential backoff and jitter;
* a concurrency ceiling (one semaphore per client instance);
* typed error classification (auth / rate limit / server / timeout / network /
  request / response / JSON) so callers can decide what is retryable;
* ``request_id`` / ``trace_id`` propagation into logs and call results;
* token / latency logging, and secret redaction (the API key is never logged);
* JSON output mode (``response_format={"type": "json_object"}``) with a bounded
  compatibility fallback for providers that reject the parameter.

The client is deliberately dumb about *what* is being generated: Planner and
Judgment own their prompts, schemas, retry/repair policy and budgets. The client
never turns a failed call into an empty or fake successful result.

No network call happens at import time and no client is created at module scope.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from recommender.config import Settings, load_settings
from recommender.observability.context import get_request_context
from recommender.observability.logging import get_logger
from recommender.utils.errors import ConfigError, ExternalAPIError

_LOGGER = get_logger(__name__)

DEFAULT_CHAT_PATH = "/chat/completions"
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4096
DEFAULT_RETRY_BASE_DELAY_S = 0.5
DEFAULT_RETRY_MAX_DELAY_S = 8.0
DEFAULT_JITTER_RATIO = 0.1

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429})
_MAX_ERROR_EXCERPT_CHARS = 300
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class DeepSeekErrorKind(str, Enum):
    """Machine-readable failure class (never collapsed into one opaque error)."""

    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    TIMEOUT = "timeout"
    NETWORK = "network"
    REQUEST = "request"
    RESPONSE = "response"
    TRUNCATED = "truncated"
    JSON = "json"


class DeepSeekAPIError(ExternalAPIError):
    """Base class for every DeepSeek failure; ``retryable`` is explicit."""

    kind: DeepSeekErrorKind = DeepSeekErrorKind.RESPONSE

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        request_id: str | None = None,
        trace_id: str | None = None,
        response_excerpt: str | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(
            message,
            provider="deepseek",
            status_code=status_code,
            retryable=retryable,
            code=f"deepseek_{self.kind.value}",
        )
        self.request_id = request_id
        self.trace_id = trace_id
        self.response_excerpt = response_excerpt
        self.attempts = attempts

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "kind": self.kind.value,
                "request_id": self.request_id,
                "trace_id": self.trace_id,
                "attempts": self.attempts,
            }
        )
        if self.response_excerpt:
            payload["response_excerpt"] = self.response_excerpt
        return payload


class DeepSeekAuthError(DeepSeekAPIError):
    kind = DeepSeekErrorKind.AUTH


class DeepSeekRateLimitError(DeepSeekAPIError):
    kind = DeepSeekErrorKind.RATE_LIMIT


class DeepSeekServerError(DeepSeekAPIError):
    kind = DeepSeekErrorKind.SERVER


class DeepSeekTimeoutError(DeepSeekAPIError):
    kind = DeepSeekErrorKind.TIMEOUT


class DeepSeekNetworkError(DeepSeekAPIError):
    kind = DeepSeekErrorKind.NETWORK


class DeepSeekRequestError(DeepSeekAPIError):
    kind = DeepSeekErrorKind.REQUEST


class DeepSeekResponseError(DeepSeekAPIError):
    kind = DeepSeekErrorKind.RESPONSE


class DeepSeekTruncatedError(DeepSeekAPIError):
    """The model spent the whole token budget on reasoning and emitted no content.

    Observed with the reasoning model ``deepseek-flash`` (2026-11-24): a multi-product
    Judgment prompt at ``max_tokens=4096`` produced ``finish_reason=length`` with an empty
    ``content`` and thousands of characters in ``reasoning_content``. This is a distinct
    failure because a bounded retry with ``reasoning_effort="none"`` fixes it, while
    treating it as "empty response" gives an unhelpful error.
    """

    kind = DeepSeekErrorKind.TRUNCATED

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = True,
        request_id: str | None = None,
        trace_id: str | None = None,
        max_tokens: int | None = None,
        reasoning_chars: int = 0,
        attempts: int = 0,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            retryable=retryable,
            request_id=request_id,
            trace_id=trace_id,
            attempts=attempts,
        )
        self.max_tokens = max_tokens
        self.reasoning_chars = reasoning_chars

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update({"max_tokens": self.max_tokens, "reasoning_chars": self.reasoning_chars})
        return payload


class DeepSeekJSONError(DeepSeekAPIError):
    kind = DeepSeekErrorKind.JSON


@dataclass(frozen=True)
class ChatCompletion:
    """One successful ``/chat/completions`` response plus local execution metadata."""

    content: str
    model: str
    request_id: str
    trace_id: str
    latency_ms: float
    attempts: int
    http_status: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    api_request_id: str | None = None


@dataclass(frozen=True)
class JSONCompletion:
    """A chat completion whose content was parsed into a JSON object."""

    data: dict[str, Any]
    completion: ChatCompletion

    @property
    def raw_content(self) -> str:
        return self.completion.content


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _excerpt(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:_MAX_ERROR_EXCERPT_CHARS]


def _extract_outermost_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_json_content(
    content: str,
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Parse a model answer into a JSON object.

    Models occasionally wrap JSON in a markdown fence or add a short prefix/suffix.
    Extraction is tried in that order; anything else is a hard :class:`DeepSeekJSONError`
    so the caller can run its own bounded repair loop instead of receiving a fake dict.
    """
    if not content or not content.strip():
        raise DeepSeekJSONError(
            "model returned empty content instead of a JSON object",
            request_id=request_id,
            trace_id=trace_id,
            response_excerpt=_excerpt(content or ""),
        )
    candidates = [content.strip()]
    fenced = _FENCE_RE.search(content)
    if fenced is not None and fenced.group(1).strip():
        candidates.append(fenced.group(1).strip())
    outermost = _extract_outermost_object(content)
    if outermost is not None:
        candidates.append(outermost)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise DeepSeekJSONError(
        "model output is not a valid JSON object",
        request_id=request_id,
        trace_id=trace_id,
        response_excerpt=_excerpt(content),
    )


def _ensure_settings(settings: Settings | None) -> Settings:
    return settings if settings is not None else load_settings()


class DeepSeekClient:
    """Synchronous, thread-safe-enough chat client with bounded retries.

    ``base_url`` / ``model`` / ``api_key`` default to :func:`load_settings` (key only
    from the environment / gitignored ``.env``). Every value is injectable so tests and
    alternate deployments can supply a mock ``httpx.Client`` and a fake sleeper.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        max_concurrency: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retry_base_delay_s: float = DEFAULT_RETRY_BASE_DELAY_S,
        retry_max_delay_s: float = DEFAULT_RETRY_MAX_DELAY_S,
        jitter_ratio: float = DEFAULT_JITTER_RATIO,
        use_json_mode: bool = True,
        reasoning_effort: str | None = None,
        allow_reasoning_fallback: bool = True,
        chat_path: str = DEFAULT_CHAT_PATH,
        client: httpx.Client | None = None,
        settings: Settings | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        needs_settings = base_url is None or model is None or api_key is None
        settings_obj = _ensure_settings(settings) if needs_settings else settings

        if base_url is None:
            if settings_obj is None:  # pragma: no cover - defensive
                raise ConfigError("base_url is required when settings cannot be loaded")
            base_url = settings_obj.config.deepseek.base_url
        if model is None:
            if settings_obj is None:  # pragma: no cover - defensive
                raise ConfigError("model is required when settings cannot be loaded")
            model = settings_obj.config.deepseek.planner_model
        if api_key is None:
            if settings_obj is None:  # pragma: no cover - defensive
                raise ConfigError("api_key is required when settings cannot be loaded")
            api_key = settings_obj.require_deepseek_api_key()
        if not api_key.strip():
            raise ConfigError("DeepSeek API key must not be blank")

        if timeout_s is None:
            timeout_s = (
                settings_obj.config.deepseek.timeout_s
                if settings_obj is not None
                else DEFAULT_TIMEOUT_S
            )
        if max_retries is None:
            max_retries = (
                settings_obj.config.deepseek.max_retries
                if settings_obj is not None
                else DEFAULT_MAX_RETRIES
            )
        if max_concurrency is None:
            max_concurrency = (
                settings_obj.config.deepseek.max_concurrency
                if settings_obj is not None
                else DEFAULT_MAX_CONCURRENCY
            )
        if temperature is None:
            temperature = (
                settings_obj.config.deepseek.temperature
                if settings_obj is not None
                else DEFAULT_TEMPERATURE
            )
        if max_tokens is None:
            max_tokens = (
                settings_obj.config.deepseek.max_tokens
                if settings_obj is not None
                else DEFAULT_MAX_TOKENS
            )
        if timeout_s <= 0:
            raise ConfigError("timeout_s must be positive")
        if max_retries < 0:
            raise ConfigError("max_retries must be non-negative")
        if max_concurrency < 1:
            raise ConfigError("max_concurrency must be >= 1")

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout_s = float(timeout_s)
        self._max_retries = int(max_retries)
        self._max_concurrency = int(max_concurrency)
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._retry_base_delay_s = max(0.0, float(retry_base_delay_s))
        self._retry_max_delay_s = max(0.0, float(retry_max_delay_s))
        self._jitter_ratio = max(0.0, float(jitter_ratio))
        self._use_json_mode = use_json_mode
        self._reasoning_effort = reasoning_effort
        self._allow_reasoning_fallback = allow_reasoning_fallback
        self._chat_path = chat_path if chat_path.startswith("/") else f"/{chat_path}"
        self._sleeper = sleeper
        self._random_source = random_source
        self._semaphore = threading.BoundedSemaphore(self._max_concurrency)
        self._owns_client = client is None
        self._http = client if client is not None else httpx.Client(timeout=self._timeout_s)

    # -- safe introspection (never exposes the key) --------------------------
    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def json_mode_enabled(self) -> bool:
        return self._use_json_mode

    @property
    def reasoning_effort(self) -> str | None:
        return self._reasoning_effort

    @property
    def reasoning_fallback_enabled(self) -> bool:
        return self._allow_reasoning_fallback

    def describe(self) -> dict[str, Any]:
        """Safe client description for traces/logs (contains no secret material)."""
        return {
            "provider": "deepseek",
            "base_url": self._base_url,
            "model": self._model,
            "timeout_s": self._timeout_s,
            "max_retries": self._max_retries,
            "max_concurrency": self._max_concurrency,
            "json_mode": self._use_json_mode,
            "reasoning_effort": self._reasoning_effort,
            "reasoning_fallback": self._allow_reasoning_fallback,
        }

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> DeepSeekClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- public API ----------------------------------------------------------
    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_json_mode: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> JSONCompletion:
        """Call chat completions and parse the answer as a JSON object."""
        completion = self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            use_json_mode=use_json_mode,
            reasoning_effort=reasoning_effort,
        )
        data = parse_json_content(
            completion.content,
            request_id=completion.request_id,
            trace_id=completion.trace_id,
        )
        return JSONCompletion(data=data, completion=completion)

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_json_mode: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatCompletion:
        """Call chat completions with bounded retries; return the raw answer.

        ``reasoning_effort`` is passed through to the API when set. If it is unset and
        the model burns the whole token budget on reasoning (empty ``content``,
        ``finish_reason=length``), the client retries once with ``reasoning_effort="none"``
        when ``allow_reasoning_fallback`` is enabled; this compatibility fallback is
        logged and never hides the failure from traces.
        """
        if not messages:
            raise ValueError("messages must not be empty")
        request_id, trace_id = self._identity()
        model_name = model or self._model
        temperature_value = self._temperature if temperature is None else float(temperature)
        max_tokens_value = self._max_tokens if max_tokens is None else int(max_tokens)
        json_mode = self._use_json_mode if use_json_mode is None else use_json_mode
        reasoning_value = self._reasoning_effort if reasoning_effort is None else reasoning_effort

        attempt = 0
        compatibility_retry_used = False
        reasoning_fallback_used = False
        while True:
            attempt += 1
            body = self._build_body(
                messages,
                model=model_name,
                temperature=temperature_value,
                max_tokens=max_tokens_value,
                json_mode=json_mode,
                reasoning_effort=reasoning_value,
            )
            started = time.perf_counter()
            try:
                with self._semaphore:
                    response = self._http.post(
                        self._url(),
                        json=body,
                        headers=self._headers(),
                    )
            except httpx.TimeoutException as exc:
                error: DeepSeekAPIError = DeepSeekTimeoutError(
                    f"DeepSeek request timed out after {self._timeout_s:.1f}s: {exc}",
                    retryable=True,
                    request_id=request_id,
                    trace_id=trace_id,
                    attempts=attempt,
                )
                self._log_failure(error, model_name, started, attempt)
                if attempt <= self._max_retries:
                    self._sleep_before_retry(attempt, None)
                    continue
                raise error from exc
            except httpx.TransportError as exc:
                error = DeepSeekNetworkError(
                    f"DeepSeek transport error: {type(exc).__name__}: {exc}",
                    retryable=True,
                    request_id=request_id,
                    trace_id=trace_id,
                    attempts=attempt,
                )
                self._log_failure(error, model_name, started, attempt)
                if attempt <= self._max_retries:
                    self._sleep_before_retry(attempt, None)
                    continue
                raise error from exc
            except httpx.HTTPError as exc:  # pragma: no cover - unusual client misuse
                raise DeepSeekNetworkError(
                    f"DeepSeek HTTP client error: {type(exc).__name__}: {exc}",
                    retryable=False,
                    request_id=request_id,
                    trace_id=trace_id,
                    attempts=attempt,
                ) from exc

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if response.status_code == 200:
                try:
                    completion = self._decode(
                        response,
                        request_id,
                        trace_id,
                        attempt,
                        elapsed_ms,
                        max_tokens_value,
                    )
                except DeepSeekTruncatedError as exc:
                    self._log_failure(exc, model_name, started, attempt)
                    if (
                        self._allow_reasoning_fallback
                        and not reasoning_fallback_used
                        and reasoning_value is None
                        and attempt <= self._max_retries
                    ):
                        reasoning_fallback_used = True
                        reasoning_value = "none"
                        _LOGGER.warning(
                            "deepseek.chat exhausted max_tokens on reasoning; retrying "
                            "with reasoning_effort='none'",
                            extra={
                                "provider": "deepseek",
                                "request_id": request_id,
                                "trace_id": trace_id,
                                "model": model_name,
                                "attempt": attempt,
                                "max_tokens": max_tokens_value,
                                "reasoning_chars": exc.reasoning_chars,
                            },
                        )
                        continue
                    raise
                _LOGGER.info(
                    "deepseek.chat completed",
                    extra={
                        "provider": "deepseek",
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "model": completion.model,
                        "attempt": attempt,
                        "http_status": completion.http_status,
                        "latency_ms": round(completion.latency_ms, 1),
                        "prompt_tokens": completion.prompt_tokens,
                        "completion_tokens": completion.completion_tokens,
                        "total_tokens": completion.total_tokens,
                        "finish_reason": completion.finish_reason,
                        "json_mode": json_mode and self._use_json_mode,
                        "reasoning_effort": reasoning_value,
                        "reasoning_fallback_used": reasoning_fallback_used,
                    },
                )
                return completion

            body_text = response.text or ""
            rejection = (
                response.status_code == 400
                and self._use_json_mode
                and "response_format" in body_text.lower()
            )
            if rejection and not compatibility_retry_used:
                compatibility_retry_used = True
                self._use_json_mode = False
                json_mode = False
                attempt -= 1  # a compatibility probe is not a user-visible attempt
                _LOGGER.warning(
                    "deepseek.chat JSON mode rejected; retrying without response_format",
                    extra={
                        "provider": "deepseek",
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "http_status": response.status_code,
                    },
                )
                continue

            error = self._http_error(response, request_id, trace_id, attempt)
            self._log_failure(error, model_name, started, attempt)
            if error.retryable and attempt <= self._max_retries:
                retry_after = self._retry_after_seconds(response)
                self._sleep_before_retry(attempt, retry_after)
                continue
            raise error

    # -- internals -----------------------------------------------------------
    def _identity(self) -> tuple[str, str]:
        context = get_request_context()
        if context is not None:
            return context.request_id, context.trace_id
        return _new_id("req"), _new_id("tr")

    def _url(self) -> str:
        return f"{self._base_url}{self._chat_path}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _build_body(
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        return body

    def _decode(
        self,
        response: httpx.Response,
        request_id: str,
        trace_id: str,
        attempt: int,
        elapsed_ms: float,
        max_tokens: int,
    ) -> ChatCompletion:
        try:
            payload = response.json()
        except ValueError as exc:
            raise DeepSeekResponseError(
                "DeepSeek response is not valid JSON",
                status_code=response.status_code,
                retryable=False,
                request_id=request_id,
                trace_id=trace_id,
                response_excerpt=_excerpt(response.text or ""),
                attempts=attempt,
            ) from exc
        if not isinstance(payload, dict):
            raise DeepSeekResponseError(
                "DeepSeek response envelope is not a JSON object",
                status_code=response.status_code,
                request_id=request_id,
                trace_id=trace_id,
                response_excerpt=_excerpt(response.text or ""),
                attempts=attempt,
            )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DeepSeekResponseError(
                "DeepSeek response has no choices",
                status_code=response.status_code,
                request_id=request_id,
                trace_id=trace_id,
                response_excerpt=_excerpt(response.text or ""),
                attempts=attempt,
            )
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        raw_finish = first.get("finish_reason") if isinstance(first, dict) else None
        raw_reasoning = (
            message.get("reasoning_content") if isinstance(message, dict) else None
        )
        reasoning_chars = len(raw_reasoning) if isinstance(raw_reasoning, str) else 0
        if not isinstance(content, str) or not content.strip():
            if raw_finish == "length":
                raise DeepSeekTruncatedError(
                    (
                        "DeepSeek returned no content because the "
                        f"{max_tokens}-token budget was exhausted before the answer "
                        f"(reasoning_content chars={reasoning_chars})"
                    ),
                    status_code=response.status_code,
                    retryable=True,
                    request_id=request_id,
                    trace_id=trace_id,
                    max_tokens=max_tokens,
                    reasoning_chars=reasoning_chars,
                    attempts=attempt,
                )
            raise DeepSeekResponseError(
                "DeepSeek response has no textual message content",
                status_code=response.status_code,
                request_id=request_id,
                trace_id=trace_id,
                response_excerpt=_excerpt(response.text or ""),
                attempts=attempt,
            )
        usage = payload.get("usage")
        usage_dict = usage if isinstance(usage, dict) else {}
        api_request_id = payload.get("id")
        return ChatCompletion(
            content=content,
            model=str(payload.get("model") or self._model),
            request_id=request_id,
            trace_id=trace_id,
            latency_ms=elapsed_ms,
            attempts=attempt,
            http_status=response.status_code,
            prompt_tokens=_int_or_none(usage_dict.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage_dict.get("completion_tokens")),
            total_tokens=_int_or_none(usage_dict.get("total_tokens")),
            finish_reason=raw_finish if isinstance(raw_finish, str) else None,
            api_request_id=api_request_id if isinstance(api_request_id, str) else None,
        )

    @staticmethod
    def _http_error(
        response: httpx.Response,
        request_id: str,
        trace_id: str,
        attempt: int,
    ) -> DeepSeekAPIError:
        status = response.status_code
        excerpt = _excerpt(response.text or "")
        message = f"DeepSeek API returned HTTP {status}"
        if excerpt:
            message += f": {excerpt}"
        common: dict[str, Any] = {
            "status_code": status,
            "request_id": request_id,
            "trace_id": trace_id,
            "response_excerpt": excerpt,
            "attempts": attempt,
        }
        if status in (401, 403):
            return DeepSeekAuthError(message, retryable=False, **common)
        if status == 429:
            return DeepSeekRateLimitError(message, retryable=True, **common)
        if status >= 500:
            return DeepSeekServerError(message, retryable=True, **common)
        if status in _RETRYABLE_STATUS or status == 408:
            return DeepSeekServerError(message, retryable=True, **common)
        if 400 <= status < 500:
            return DeepSeekRequestError(message, retryable=False, **common)
        return DeepSeekResponseError(message, retryable=False, **common)

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        raw = response.headers.get("retry-after")
        if raw is None:
            return None
        try:
            value = float(raw.strip())
        except ValueError:
            return None
        return value if value >= 0 else None

    def _backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self._retry_max_delay_s)
        base = min(
            self._retry_max_delay_s,
            self._retry_base_delay_s * (2.0 ** (attempt - 1)),
        )
        return base + base * self._jitter_ratio * max(0.0, min(1.0, self._random_source()))

    def _sleep_before_retry(self, attempt: int, retry_after: float | None) -> None:
        delay = self._backoff_delay(attempt, retry_after)
        _LOGGER.warning(
            "deepseek.chat retrying after failure",
            extra={
                "provider": "deepseek",
                "model": self._model,
                "attempt": attempt,
                "max_retries": self._max_retries,
                "retry_delay_s": round(delay, 3),
            },
        )
        if delay > 0:
            self._sleeper(delay)

    @staticmethod
    def _log_failure(
        error: DeepSeekAPIError,
        model: str,
        started: float,
        attempt: int,
    ) -> None:
        _LOGGER.warning(
            "deepseek.chat failed",
            extra={
                "provider": "deepseek",
                "request_id": error.request_id,
                "trace_id": error.trace_id,
                "model": model,
                "attempt": attempt,
                "error_kind": error.kind.value,
                "status_code": error.status_code,
                "retryable": error.retryable,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "response_excerpt": error.response_excerpt,
            },
        )
