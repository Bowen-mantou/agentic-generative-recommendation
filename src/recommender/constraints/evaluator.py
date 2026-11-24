"""The single Python hard-constraint evaluator (AGENTS.md §2.4, Q3).

Deterministic checks only: price, category, brand, rating and catalog attributes.
Missing or non-comparable fields yield ``unknown`` — never ``pass``. The evaluator is
the only place in the system that performs numeric comparisons for hard constraints;
Judgment and the Reranker must consume its output instead of re-checking values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from recommender.schemas.common import ConstraintOperator, ConstraintStatus
from recommender.schemas.constraints import (
    ConstraintCheck,
    ConstraintEvaluation,
    aggregate_constraint_status,
)
from recommender.schemas.plan import HardConstraint
from recommender.schemas.product import ProductRecord

_NUMERIC_OPERATORS = {
    ConstraintOperator.LT,
    ConstraintOperator.LTE,
    ConstraintOperator.GT,
    ConstraintOperator.GTE,
}
_LIST_OPERATORS = {ConstraintOperator.IN, ConstraintOperator.NOT_IN}

# Constraint field -> ProductRecord attribute. Unknown fields fall back to `details`.
_FIELD_MAP: dict[str, str] = {
    "price": "price",
    "price_usd": "price",
    "main_category": "main_category",
    "category": "main_category",
    "brand": "brand",
    "store": "store",
    "title": "title",
    "rating": "average_rating",
    "average_rating": "average_rating",
    "rating_number": "rating_number",
    "currency": "currency",
}


class ResolvedValue(NamedTuple):
    value: Any
    source: str
    found: bool


def _normalize_field(field: str) -> str:
    return field.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_attribute(product: ProductRecord | None, field: str) -> ResolvedValue:
    """Resolve a constraint field against the catalog record (never infers values)."""
    if product is None:
        return ResolvedValue(value=None, source="product_missing", found=False)
    normalized = _normalize_field(field)
    attribute = _FIELD_MAP.get(normalized)
    if attribute is not None:
        value = getattr(product, attribute)
        return ResolvedValue(value=value, source=attribute, found=value is not None)
    for key, raw_value in product.details.items():
        if _normalize_field(key) == normalized:
            return ResolvedValue(value=raw_value, source=f"details.{key}", found=True)
    return ResolvedValue(value=None, source="unknown_field", found=False)


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace("$", "").replace(",", ""))
        except ValueError:
            return None
    return None


def _as_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip().casefold() or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value).casefold()
    return None


def _unknown(
    field: str,
    operator: ConstraintOperator,
    expected: Any,
    reason: str,
) -> ConstraintCheck:
    return ConstraintCheck(
        field=field,
        operator=operator,
        expected=expected,
        observed=None,
        status=ConstraintStatus.UNKNOWN,
        reason=reason,
    )


def evaluate_constraint(
    constraint: HardConstraint,
    product: ProductRecord | None,
) -> ConstraintCheck:
    """Evaluate one hard constraint against one product."""
    resolved = resolve_attribute(product, constraint.field)
    observed = resolved.value
    if not resolved.found or observed is None:
        return _unknown(
            constraint.field,
            constraint.operator,
            constraint.value,
            f"field '{constraint.field}' is missing in the catalog",
        )

    if (
        constraint.field.strip().lower() in {"price", "price_usd"}
        and constraint.unit is not None
        and product is not None
        and product.currency.upper() != constraint.unit.upper()
    ):
        return _unknown(
            constraint.field,
            constraint.operator,
            constraint.value,
            (
                "currency mismatch: product currency is "
                f"{product.currency}, constraint unit is {constraint.unit}"
            ),
        )

    operator = constraint.operator
    expected = constraint.value

    if operator in _NUMERIC_OPERATORS:
        left = _as_number(observed)
        right = _as_number(expected)
        if left is None or right is None:
            return _unknown(
                constraint.field,
                operator,
                expected,
                f"non-numeric comparison for operator '{operator.value}'",
            )
        comparisons = {
            ConstraintOperator.LT: left < right,
            ConstraintOperator.LTE: left <= right,
            ConstraintOperator.GT: left > right,
            ConstraintOperator.GTE: left >= right,
        }
        passed = comparisons[operator]
        return ConstraintCheck(
            field=constraint.field,
            operator=operator,
            expected=expected,
            observed=observed,
            status=ConstraintStatus.PASS if passed else ConstraintStatus.FAIL,
            reason=None
            if passed
            else f"observed {observed!r} does not satisfy {operator.value} {expected!r}",
        )

    if operator in _LIST_OPERATORS:
        options = expected if isinstance(expected, list) else [expected]
        observed_text = _as_text(observed)
        option_texts = {_as_text(option) for option in options}
        matched = observed_text is not None and observed_text in option_texts
        passed = matched if operator is ConstraintOperator.IN else not matched
        return ConstraintCheck(
            field=constraint.field,
            operator=operator,
            expected=expected,
            observed=observed,
            status=ConstraintStatus.PASS if passed else ConstraintStatus.FAIL,
            reason=None
            if passed
            else f"observed {observed!r} does not satisfy {operator.value} {expected!r}",
        )

    if operator is ConstraintOperator.CONTAINS:
        expected_text = _as_text(expected)
        if observed_text := _as_text(observed):
            passed = expected_text is not None and expected_text in observed_text
        elif isinstance(observed, list):
            passed = expected_text is not None and expected_text in {
                _as_text(item) for item in observed
            }
        else:
            return _unknown(
                constraint.field,
                operator,
                expected,
                "contains requires a string or list value",
            )
        return ConstraintCheck(
            field=constraint.field,
            operator=operator,
            expected=expected,
            observed=observed,
            status=ConstraintStatus.PASS if passed else ConstraintStatus.FAIL,
            reason=None if passed else f"observed {observed!r} does not contain {expected!r}",
        )

    # eq / ne
    left_text, right_text = _as_text(observed), _as_text(expected)
    if left_text is not None and right_text is not None:
        equal = left_text == right_text
    else:
        equal = observed == expected
    passed = equal if operator is ConstraintOperator.EQ else not equal
    return ConstraintCheck(
        field=constraint.field,
        operator=operator,
        expected=expected,
        observed=observed,
        status=ConstraintStatus.PASS if passed else ConstraintStatus.FAIL,
        reason=None
        if passed
        else f"observed {observed!r} does not satisfy {operator.value} {expected!r}",
    )


class ConstraintEvaluator:
    """Evaluates a fixed constraint set against products and candidate lists."""

    def __init__(self, constraints: Sequence[HardConstraint]) -> None:
        self.constraints = list(constraints)

    def evaluate(self, product_id: str, product: ProductRecord | None) -> ConstraintEvaluation:
        checks = [evaluate_constraint(constraint, product) for constraint in self.constraints]
        return ConstraintEvaluation(
            product_id=product_id,
            status=aggregate_constraint_status(check.status for check in checks),
            checks=checks,
        )

    def evaluate_products(
        self, products: Mapping[str, ProductRecord]
    ) -> dict[str, ConstraintEvaluation]:
        return {
            product_id: self.evaluate(product_id, product)
            for product_id, product in products.items()
        }
