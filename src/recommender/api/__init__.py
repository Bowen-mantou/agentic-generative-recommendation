"""DeepSeek API client shared by Planner, Judgment and (later) Generator."""

from __future__ import annotations

from recommender.api.deepseek import (
    ChatCompletion,
    DeepSeekAPIError,
    DeepSeekAuthError,
    DeepSeekClient,
    DeepSeekErrorKind,
    DeepSeekJSONError,
    DeepSeekNetworkError,
    DeepSeekRateLimitError,
    DeepSeekRequestError,
    DeepSeekResponseError,
    DeepSeekServerError,
    DeepSeekTimeoutError,
    DeepSeekTruncatedError,
    JSONCompletion,
    parse_json_content,
)

__all__ = [
    "ChatCompletion",
    "DeepSeekAPIError",
    "DeepSeekAuthError",
    "DeepSeekClient",
    "DeepSeekErrorKind",
    "DeepSeekJSONError",
    "DeepSeekNetworkError",
    "DeepSeekRateLimitError",
    "DeepSeekRequestError",
    "DeepSeekResponseError",
    "DeepSeekServerError",
    "DeepSeekTimeoutError",
    "DeepSeekTruncatedError",
    "JSONCompletion",
    "parse_json_content",
]
