"""Evaluation contracts: versioned samples and reproducible metrics.

Formal experiments must never consume mock or synthetic data; the schemas therefore
carry explicit provenance and a ``uses_mock`` flag (AGENTS.md §6, §7.11).
"""

from __future__ import annotations

from pydantic import Field, model_validator
from typing_extensions import Self

from recommender.schemas.common import ContractModel, EvalSplit, NonEmptyStr

FORMAL_K_VALUES = (10, 50, 100)


class RetrievalEvalSample(ContractModel):
    """One time-split evaluation query for the retrieval paths."""

    sample_id: NonEmptyStr
    user_id: NonEmptyStr
    split: EvalSplit
    query: NonEmptyStr
    history_product_ids: list[str] = Field(default_factory=list)
    ground_truth_product_ids: list[str] = Field(min_length=1)
    cutoff_timestamp: int | None = Field(default=None, ge=0)
    is_synthetic: bool = False

    @model_validator(mode="after")
    def _no_target_leakage_in_history(self) -> Self:
        overlap = set(self.history_product_ids) & set(self.ground_truth_product_ids)
        if overlap:
            raise ValueError(
                "ground-truth targets must not appear in the pre-cutoff history "
                f"(leakage: {sorted(overlap)})"
            )
        return self

    @model_validator(mode="after")
    def _formal_split_is_real(self) -> Self:
        if self.split in (EvalSplit.VALIDATION, EvalSplit.TEST) and self.is_synthetic:
            raise ValueError("synthetic samples cannot enter validation/test evaluation")
        return self


class RetrievalMetrics(ContractModel):
    system: NonEmptyStr
    evaluated_samples: int = Field(ge=1)
    k_values: list[int] = Field(default_factory=lambda: list(FORMAL_K_VALUES))
    recall_at_k: dict[str, float] = Field(default_factory=dict)
    mrr: float | None = Field(default=None, ge=0.0, le=1.0)
    ndcg_at_10: float | None = Field(default=None, ge=0.0)
    catalog_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms_mean: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _recall_keys_match_k_values(self) -> Self:
        expected = {str(k) for k in self.k_values}
        if self.recall_at_k and set(self.recall_at_k) != expected:
            raise ValueError("recall_at_k keys must match k_values")
        return self


class RetrievalEvaluationReport(ContractModel):
    dataset_version: NonEmptyStr
    catalog_version: NonEmptyStr
    index_version: NonEmptyStr
    code_commit: str | None = None
    created_at: NonEmptyStr
    uses_mock: bool = False
    metrics: list[RetrievalMetrics] = Field(min_length=1)

    @model_validator(mode="after")
    def _mock_is_flagged(self) -> Self:
        if self.uses_mock:
            raise ValueError(
                "mock results cannot be published as formal evaluation reports; "
                "mark them in a separate diagnostic artifact"
            )
        return self
