"""Structured logging, trace identity propagation and JSONL tracing."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from recommender.observability.context import (
    get_request_context,
    new_request_context,
    request_context,
)
from recommender.observability.logging import configure_logging, get_logger
from recommender.observability.trace import TraceWriter, read_trace

pytestmark = pytest.mark.unit


def test_request_context_is_generated_and_reset() -> None:
    assert get_request_context() is None
    with request_context() as context:
        assert context.request_id.startswith("req_")
        assert context.trace_id.startswith("tr_")
        assert get_request_context() == context
    assert get_request_context() is None


def test_new_request_context_respects_supplied_ids() -> None:
    context = new_request_context(request_id="req_1", trace_id="tr_1", user_id="u1")
    assert context.as_dict() == {"request_id": "req_1", "trace_id": "tr_1", "user_id": "u1"}


def test_json_logging_injects_context_and_redacts_secrets() -> None:
    stream = io.StringIO()
    configure_logging("INFO", json_output=True, stream=stream)
    with request_context(request_id="req_log", trace_id="tr_log"):
        get_logger("test.logger").info(
            "candidate fused",
            extra={"api_key": "sk-should-not-leak", "count": 3},
        )
    raw = stream.getvalue().strip().splitlines()[-1]
    payload = json.loads(raw)
    assert payload["message"] == "candidate fused"
    assert payload["request_id"] == "req_log"
    assert payload["trace_id"] == "tr_log"
    assert payload["api_key"] == "***REDACTED***"
    assert payload["count"] == 3


def test_text_logging_is_human_readable() -> None:
    stream = io.StringIO()
    configure_logging("INFO", json_output=False, stream=stream)
    get_logger("test.logger").info("hello")
    assert "hello" in stream.getvalue()


def test_trace_writer_appends_sequenced_events(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path, request_id="req_t", trace_id="tr_t") as writer:
        writer.emit("planner.plan", {"plan_id": "plan_1", "api_key": "sk-secret"})
        writer.emit("retrieval.e5", {"hits": 10}, stage="retrieval")
    events = read_trace(path)
    assert [event["seq"] for event in events] == [1, 2]
    assert events[0]["request_id"] == "req_t"
    assert events[0]["trace_id"] == "tr_t"
    assert events[0]["payload"]["api_key"] == "***REDACTED***"
    assert events[1]["stage"] == "retrieval"
