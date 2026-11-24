"""Build the post-constraint candidate pool handed to the Reranker.

Fusion candidates are annotated with the single evaluator's verdict and partitioned into
eligible / unverified / rejected. A pool is only produced after the constraint pass so
downstream modules never see an unvalidated status.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from recommender.constraints.evaluator import ConstraintEvaluator
from recommender.fusion.fuse import FusionOutcome
from recommender.observability.logging import get_logger
from recommender.schemas.common import ConstraintStatus
from recommender.schemas.constraints import ConstraintSummary
from recommender.schemas.fusion import CandidatePool, FusedCandidate
from recommender.schemas.plan import HardConstraint
from recommender.schemas.product import ProductRecord

_LOGGER = get_logger(__name__)


def _metadata(product: ProductRecord | None) -> dict[str, object]:
    if product is None:
        return {}
    return {
        "title": product.title,
        "price": product.price,
        "main_category": product.main_category,
        "brand": product.brand,
    }


def build_candidate_pool(
    *,
    request_id: str,
    outcome: FusionOutcome,
    constraints: Sequence[HardConstraint],
    products: Mapping[str, ProductRecord] | None = None,
    rrf_k: int = 60,
) -> CandidatePool:
    """Evaluate constraints, partition candidates and return a validated pool."""
    catalog = products or {}
    evaluator = ConstraintEvaluator(constraints)
    annotated: list[FusedCandidate] = []
    for candidate in outcome.candidates:
        product = catalog.get(candidate.product_id)
        evaluation = evaluator.evaluate(candidate.product_id, product)
        annotated.append(
            candidate.model_copy(
                update={
                    "constraint_status": evaluation.status,
                    "constraint_checks": evaluation.checks,
                    "product_metadata": _metadata(product),
                }
            )
        )

    eligible = [
        candidate
        for candidate in annotated
        if candidate.constraint_status in (ConstraintStatus.PASS, None)
    ]
    unverified = [
        candidate
        for candidate in annotated
        if candidate.constraint_status is ConstraintStatus.UNKNOWN
    ]
    rejected = [
        candidate for candidate in annotated if candidate.constraint_status is ConstraintStatus.FAIL
    ]
    evaluated = len(annotated)
    not_configured = evaluated if not constraints else 0
    summary = ConstraintSummary(
        evaluated_products=evaluated,
        pass_count=len([c for c in annotated if c.constraint_status is ConstraintStatus.PASS]),
        fail_count=len(rejected),
        unknown_count=len(unverified),
        not_configured_count=not_configured,
        valid_product_count=len(eligible),
        constraint_fields=list(dict.fromkeys(c.field for c in constraints)),
    )
    pool = CandidatePool(
        request_id=request_id,
        candidates=eligible,
        unverified_candidates=unverified,
        rejected_candidates=rejected,
        constraint_summary=summary,
        rrf_k=rrf_k,
    )
    _LOGGER.info(
        "candidate pool built",
        extra={
            "request_id": request_id,
            "evaluated": evaluated,
            "eligible": len(eligible),
            "unverified": len(unverified),
            "rejected": len(rejected),
            "constraints": summary.constraint_fields,
            "not_configured": not_configured,
            "degraded_paths": outcome.degraded_paths,
        },
    )
    return pool
