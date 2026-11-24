"""Exception hierarchy shared by every module.

Errors are typed so callers can distinguish configuration problems, contract
violations, missing artifacts and external API failures instead of collapsing them
into one opaque failure (AGENTS.md §5). External API failures must never be reported
as empty candidate lists or successful results.
"""

from __future__ import annotations

from typing import Any


class RecommenderError(Exception):
    """Base class for all project errors."""

    code: str = "recommender_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code

    def to_dict(self) -> dict[str, Any]:
        """Structured representation safe for logs and traces."""
        return {"error_code": self.code, "error_type": type(self).__name__, "message": self.message}


class ConfigError(RecommenderError):
    """Invalid, missing or contradictory configuration."""

    code = "config_error"


class ContractError(RecommenderError):
    """A cross-module data contract was violated (schema, ID mapping, invariants)."""

    code = "contract_error"


class ArtifactError(RecommenderError):
    """A model artifact, index or mapping is missing, stale or inconsistent."""

    code = "artifact_error"


class RetrievalError(RecommenderError):
    """A retrieval path failed to produce a valid result."""

    code = "retrieval_error"


class ExternalAPIError(RecommenderError):
    """A remote API call failed.

    ``retryable`` tells the caller whether a bounded retry may help; it never means the
    failure can be downgraded to an empty successful result.
    """

    code = "external_api_error"

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retryable: bool = False,
        code: str | None = None,
    ) -> None:
        super().__init__(message, code=code)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "provider": self.provider,
                "status_code": self.status_code,
                "retryable": self.retryable,
            }
        )
        return payload
