"""Request / trace identity propagation via context variables.

Every recommendation request owns a ``request_id`` and a ``trace_id``. Both are
attached to log records and written into the JSONL trace so that one request can be
reconstructed end to end (AGENTS.md §4, §5).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass

_REQUEST_CONTEXT: ContextVar[RequestContext | None] = ContextVar(
    "recommender_request_context", default=None
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class RequestContext:
    """Immutable identity of one in-flight recommendation request."""

    request_id: str
    trace_id: str
    user_id: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload = {"request_id": self.request_id, "trace_id": self.trace_id}
        if self.user_id is not None:
            payload["user_id"] = self.user_id
        return payload


def new_request_context(
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    user_id: str | None = None,
) -> RequestContext:
    """Create a request context, generating ids when they are not supplied."""
    return RequestContext(
        request_id=request_id or _new_id("req"),
        trace_id=trace_id or _new_id("tr"),
        user_id=user_id,
    )


def get_request_context() -> RequestContext | None:
    """Return the context bound to the current task/thread, if any."""
    return _REQUEST_CONTEXT.get()


def set_request_context(context: RequestContext) -> Token[RequestContext | None]:
    """Bind a context; callers must reset the returned token."""
    return _REQUEST_CONTEXT.set(context)


def reset_request_context(token: Token[RequestContext | None]) -> None:
    _REQUEST_CONTEXT.reset(token)


@contextmanager
def request_context(
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[RequestContext]:
    """Bind a fresh request context for the duration of the block."""
    context = new_request_context(request_id=request_id, trace_id=trace_id, user_id=user_id)
    token = set_request_context(context)
    try:
        yield context
    finally:
        reset_request_context(token)


def context_from_headers(headers: Mapping[str, str]) -> RequestContext:
    """Build a context from inbound headers (missing ids are generated)."""
    lowered = {key.lower(): value for key, value in headers.items()}
    return new_request_context(
        request_id=lowered.get("x-request-id"),
        trace_id=lowered.get("x-trace-id"),
        user_id=lowered.get("x-user-id"),
    )
