"""Candidate Fusion contracts: merged, de-duplicated, constraint-annotated pool.

Fusion keeps every source, source rank and raw score, then applies two-level RRF
(E5-internal first, then cross-path) and finally attaches the single Constraint
Evaluator's decision. Raw scores from different models are never summed
(AGENTS.md §2.5, module 3 §5.4).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from recommender.schemas.common import (
    ConstraintStatus,
    ContractModel,
    NonEmptyStr,
    RecallPath,
)
from recommender.schemas.constraints import ConstraintCheck, ConstraintSummary


class FusedCandidate(ContractModel):
    """One product after ID mapping, de-duplication and RRF."""

    product_id: NonEmptyStr
    fusion_score: float = Field(allow_inf_nan=False)
    retrieval_sources: list[RecallPath] = Field(min_length=1)
    source_ranks: dict[str, int] = Field(default_factory=dict)
    raw_scores: dict[str, float] = Field(default_factory=dict)
    constraint_status: ConstraintStatus | None = None
    constraint_checks: list[ConstraintCheck] = Field(default_factory=list)
    product_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sources_are_consistent(self) -> Self:
        sources = [source.value for source in self.retrieval_sources]
        if len(set(sources)) != len(sources):
            raise ValueError("retrieval_sources must be unique")
        if set(self.source_ranks) != set(sources):
            raise ValueError("source_ranks keys must exactly match retrieval_sources")
        if any(rank < 1 for rank in self.source_ranks.values()):
            raise ValueError("source ranks start at 1")
        if not set(self.raw_scores).issubset(set(sources)):
            raise ValueError("raw_scores references a source not present in retrieval_sources")
        return self

    @model_validator(mode="after")
    def _constraint_status_matches_checks(self) -> Self:
        if self.constraint_status is None:
            if self.constraint_checks:
                raise ValueError("constraint_checks require a constraint_status")
            return self
        statuses = {check.status for check in self.constraint_checks}
        if not statuses:
            if self.constraint_status is not ConstraintStatus.PASS:
                raise ValueError("a candidate without checks can only be pass")
            return self
        if self.constraint_status is ConstraintStatus.PASS and statuses != {ConstraintStatus.PASS}:
            raise ValueError("pass status requires every check to pass")
        if (
            self.constraint_status is ConstraintStatus.FAIL
            and ConstraintStatus.FAIL not in statuses
        ):
            raise ValueError("fail status requires at least one failed check")
        if self.constraint_status is ConstraintStatus.UNKNOWN and (
            ConstraintStatus.FAIL in statuses or ConstraintStatus.UNKNOWN not in statuses
        ):
            raise ValueError("unknown status requires unknown checks and no failed check")
        return self


class CandidatePool(ContractModel):
    """Post-constraint candidate pool handed to the Reranker.

    ``candidates`` are eligible (all hard constraints pass). Failed products are kept in
    ``rejected_candidates`` for diagnostics, and ``unverified_candidates`` (unknown)
    stay available for Replan but are never counted as eligible.
    """

    request_id: NonEmptyStr
    candidates: list[FusedCandidate] = Field(default_factory=list)
    unverified_candidates: list[FusedCandidate] = Field(default_factory=list)
    rejected_candidates: list[FusedCandidate] = Field(default_factory=list)
    constraint_summary: ConstraintSummary
    fusion_method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, ge=1)

    @model_validator(mode="after")
    def _partition_is_consistent(self) -> Self:
        eligible = [candidate.product_id for candidate in self.candidates]
        unverified = [candidate.product_id for candidate in self.unverified_candidates]
        rejected = [candidate.product_id for candidate in self.rejected_candidates]
        all_ids = eligible + unverified + rejected
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("a product may appear in only one pool partition")
        for candidate in self.candidates:
            if candidate.constraint_status not in (ConstraintStatus.PASS, None):
                raise ValueError(
                    "eligible candidates must have constraint_status=pass or None "
                    "(None only when no hard constraint was configured)"
                )
        for candidate in self.unverified_candidates:
            if candidate.constraint_status is not ConstraintStatus.UNKNOWN:
                raise ValueError("unverified candidates must have constraint_status=unknown")
        for candidate in self.rejected_candidates:
            if candidate.constraint_status is not ConstraintStatus.FAIL:
                raise ValueError("rejected candidates must have constraint_status=fail")
        summary = self.constraint_summary
        if (
            summary.pass_count + summary.not_configured_count != len(eligible)
            or summary.unknown_count != len(unverified)
            or summary.fail_count != len(rejected)
            or summary.evaluated_products != len(all_ids)
        ):
            raise ValueError("constraint_summary counts do not match the pool partitions")
        if summary.not_configured_count:
            if any(candidate.constraint_status is not None for candidate in self.candidates):
                raise ValueError(
                    "not_configured_count>0 requires eligible candidates with status=None"
                )
        elif any(
            candidate.constraint_status is not ConstraintStatus.PASS
            for candidate in self.candidates
        ):
            raise ValueError("with configured constraints, eligible candidates must be pass")
        scores = [candidate.fusion_score for candidate in self.candidates]
        if scores != sorted(scores, reverse=True):
            raise ValueError("eligible candidates must be ordered by fusion_score desc")
        return self

    @property
    def product_ids(self) -> list[str]:
        return [candidate.product_id for candidate in self.candidates]
