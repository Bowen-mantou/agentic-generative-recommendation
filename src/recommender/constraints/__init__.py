"""The single Python hard-constraint evaluator (Phase 2, AGENTS.md §2.4)."""

from __future__ import annotations

from recommender.constraints.evaluator import (
    ConstraintEvaluator,
    evaluate_constraint,
    resolve_attribute,
)

__all__ = ["ConstraintEvaluator", "evaluate_constraint", "resolve_attribute"]
