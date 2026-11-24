"""Hard-constraint evaluation contracts.

There is exactly one Constraint Evaluator in the system (Python), executed after RRF
fusion. It returns ``pass`` / ``fail`` / ``unknown`` per product; ``unknown`` never
counts as satisfied (AGENTS.md §2.4, module 3 §5.5).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import Field, model_validator
from typing_extensions import Self

from recommender.schemas.common import (
    ConstraintOperator,
    ConstraintStatus,
    ContractModel,
    NonEmptyStr,
)


def aggregate_constraint_status(statuses: Iterable[ConstraintStatus]) -> ConstraintStatus | None:
    """Aggregate per-constraint outcomes into a product-level status.

    ``fail`` dominates, then ``unknown``. An empty set means "no hard constraint was
    configured" and yields ``None`` — never ``pass`` (decision Q3).
    """
    materialized = list(statuses)
    if not materialized:
        return None
    if any(status is ConstraintStatus.FAIL for status in materialized):
        return ConstraintStatus.FAIL
    if any(status is ConstraintStatus.UNKNOWN for status in materialized):
        return ConstraintStatus.UNKNOWN
    return ConstraintStatus.PASS


class ConstraintCheck(ContractModel):
    """Outcome of checking one hard constraint against one product."""

    field: NonEmptyStr
    operator: ConstraintOperator
    expected: Any
    observed: Any = None
    status: ConstraintStatus
    reason: str | None = None

    @model_validator(mode="after")
    def _non_pass_requires_reason(self) -> Self:
        if self.status is ConstraintStatus.FAIL and not self.reason:
            raise ValueError("a failed constraint must explain the mismatch")
        if self.status is ConstraintStatus.UNKNOWN and not self.reason:
            raise ValueError("an unknown constraint must explain what is missing")
        return self


class ConstraintEvaluation(ContractModel):
    """All hard-constraint checks for one product."""

    product_id: NonEmptyStr
    # None means "no hard constraint configured"; only the three explicit states may be
    # used when checks exist (decision Q3).
    status: ConstraintStatus | None = None
    checks: list[ConstraintCheck] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_matches_checks(self) -> Self:
        expected = aggregate_constraint_status(check.status for check in self.checks)
        if self.status is not expected:
            got = self.status.value if self.status is not None else None
            want = expected.value if expected is not None else None
            raise ValueError(f"overall status {got!r} contradicts checks ({want!r})")
        return self

    @property
    def failed_fields(self) -> list[str]:
        return [check.field for check in self.checks if check.status is ConstraintStatus.FAIL]

    @property
    def unknown_fields(self) -> list[str]:
        return [check.field for check in self.checks if check.status is ConstraintStatus.UNKNOWN]


class ConstraintSummary(ContractModel):
    """Aggregate constraint statistics used by the Controller."""

    evaluated_products: int = Field(ge=0)
    pass_count: int = Field(ge=0)
    fail_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    not_configured_count: int = Field(default=0, ge=0)
    valid_product_count: int = Field(ge=0)
    constraint_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> Self:
        total = self.pass_count + self.fail_count + self.unknown_count + self.not_configured_count
        if total != self.evaluated_products:
            raise ValueError("pass+fail+unknown+not_configured must equal evaluated_products")
        # Passing OR "no constraint configured" products are eligible; unknown/fail are
        # not (AGENTS.md §2.4, decision Q3).
        if self.valid_product_count != self.pass_count + self.not_configured_count:
            raise ValueError("valid_product_count must equal pass_count + not_configured_count")
        if self.not_configured_count and total != self.not_configured_count:
            raise ValueError(
                "not_configured_count>0 means no constraints were configured for any product"
            )
        return self
