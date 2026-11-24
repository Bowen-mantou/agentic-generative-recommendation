"""Schema / leakage validation for synthetic queries (protocol §7)."""

from __future__ import annotations

import re

from recommender.schemas.evaluation_dataset import (
    QueryValidation,
    SyntheticQuery,
    TargetLabel,
    TemporalSample,
    UserProfile,
)

_WHITESPACE = re.compile(r"\s+")


def _normalise_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value.casefold()).strip()


def validate_temporal(sample: TemporalSample, label: TargetLabel) -> list[str]:
    errors: list[str] = []
    if sample.sample_id != label.sample_id:
        errors.append("sample_id_mismatch")
    if label.target_timestamp <= sample.cutoff_timestamp:
        errors.append("target_not_after_cutoff")
    if len(sample.history_product_ids) != len(sample.history_event_ids):
        errors.append("history_alignment")
    if sample.is_synthetic:
        errors.append("historical_sample_marked_synthetic")
    return errors


def validate_query(
    query: SyntheticQuery,
    profile: UserProfile,
    label: TargetLabel,
) -> QueryValidation:
    """Check association, fact references and exact target leakage."""
    errors: list[str] = []
    checks: list[str] = ["schema"]
    if query.sample_id != profile.sample_id:
        errors.append("sample_id_mismatch")
    if not query.is_synthetic:
        errors.append("not_marked_synthetic")
    known_facts = profile.all_facts()
    unknown = sorted(set(query.used_fact_ids) - set(known_facts))
    if unknown:
        errors.append(f"unknown_fact_ids:{','.join(unknown[:5])}")

    target_id_scanned = False
    target_title_scanned = False
    if query.query:
        normalised_query = _normalise_text(query.query)
        if _normalise_text(label.target_product_id) in normalised_query:
            errors.append("target_product_id_in_query")
        target_id_scanned = True
        if label.target_title:
            title = _normalise_text(label.target_title)
            if len(title) >= 15 and title in normalised_query:
                errors.append("target_title_in_query")
            target_title_scanned = True
        if len(query.query.strip()) < 5:
            errors.append("query_too_short")
        checks.append("leakage_scan")
    else:
        errors.append("missing_query_text")

    return QueryValidation(
        query_id=query.query_id,
        sample_id=query.sample_id,
        passed=not errors,
        checks=checks,
        errors=errors,
        target_id_scanned=target_id_scanned,
        target_title_scanned=target_title_scanned,
    )
