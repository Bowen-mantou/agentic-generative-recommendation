"""D6 Judgment evidence builder: catalog facts on demand for Top-N products.

Decision D6=B keeps the fused candidate-pool contract unchanged
(``product_id/title/price/main_category/brand``) and enriches only the products that
actually reach Judgment::

    candidate pool -> Reranker -> Top-N product_id
        -> JudgmentEvidenceBuilder (batch Catalog lookup)
        -> truncated description + relevance-filtered details
        -> DeepSeek Judgment

Hard rules:

* only real catalog facts enter :class:`JudgmentCandidate` — a missing product or
  missing field is never inferred, defaulted or fabricated;
* ``constraint_status`` is upstream read-only context (decision Q3) and is never derived
  here (no price/category comparison happens in this module);
* ``product_id`` alignment follows the Reranker order; products absent from the catalog
  are reported in ``missing_products`` and skipped, never replaced by an empty
  candidate;
* the character/token budget drops optional text in a documented priority order —
  core identity fields (``product_id``/title/price/brand/main_category) are never
  dropped, then relevance-filtered details, then the description — and every drop is
  recorded in :class:`EvidenceStats`.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from recommender.observability.logging import get_logger
from recommender.schemas.common import ConstraintStatus, Priority
from recommender.schemas.judgment import JudgmentCandidate
from recommender.schemas.plan import RecommendationPlan
from recommender.schemas.product import ProductRecord
from recommender.utils.errors import ContractError

_LOGGER = get_logger(__name__)

DEFAULT_DESCRIPTION_MAX_CHARS = 800
DEFAULT_MAX_DETAIL_KEYS = 8
DEFAULT_MAX_TOTAL_CHARS = 24_000
DEFAULT_CHARS_PER_TOKEN = 4
TRUNCATION_MARKER = "…"

# Reason why one catalog detail did not enter the Judgment payload.
DropReason = Literal[
    "irrelevant",
    "max_detail_keys",
    "reserved_key",
    "missing_value",
    "duplicate_key",
    "budget",
]

# ``catalog_lookup`` is called once per build with the de-duplicated ranked ids.
CatalogLookup = Callable[[Sequence[str]], Mapping[str, ProductRecord]]

# Fields that are core identity, never dropped by the budget.
_RESERVED_ATTRIBUTE_KEYS = frozenset({"brand", "main_category"})
# Conservative JSON overhead ("description": "..." plus separator) reserved per product
# so the measured payload cannot exceed the configured character budget.
_DESCRIPTION_OVERHEAD_CHARS = 20
# A description slice shorter than this is useless as evidence: drop it entirely.
_MIN_BUDGET_DESCRIPTION_CHARS = 24

_TOKEN_SPLIT = re.compile(r"[^\w]+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


@dataclass(frozen=True)
class DroppedDetail:
    """One catalog detail excluded from the Judgment payload (with the reason)."""

    product_id: str
    key: str
    reason: DropReason

    def to_dict(self) -> dict[str, str]:
        return {"product_id": self.product_id, "key": self.key, "reason": self.reason}


@dataclass(frozen=True)
class EvidenceStats:
    """Auditable summary of one evidence build (no silent truncation)."""

    requested_products: int
    returned_products: int
    missing_products: list[str]
    duplicate_products: list[str]
    truncated_fields: dict[str, list[str]]
    dropped_fields: dict[str, list[str]]
    dropped_details: list[DroppedDetail]
    estimated_chars: int
    char_limit: int
    token_limit: int | None
    chars_per_token: int
    budget_exhausted: bool

    @property
    def estimated_tokens(self) -> int:
        """Approximate token count from the character estimate (not a tokenizer count)."""
        return math.ceil(self.estimated_chars / self.chars_per_token)

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_products": self.requested_products,
            "returned_products": self.returned_products,
            "missing_products": list(self.missing_products),
            "duplicate_products": list(self.duplicate_products),
            "truncated_fields": {
                product_id: list(fields)
                for product_id, fields in self.truncated_fields.items()
            },
            "dropped_fields": {
                product_id: list(fields)
                for product_id, fields in self.dropped_fields.items()
            },
            "dropped_details": [detail.to_dict() for detail in self.dropped_details],
            "estimated_chars": self.estimated_chars,
            "estimated_tokens": self.estimated_tokens,
            "char_limit": self.char_limit,
            "token_limit": self.token_limit,
            "chars_per_token": self.chars_per_token,
            "budget_exhausted": self.budget_exhausted,
        }


@dataclass(frozen=True)
class EvidenceBuildResult:
    """Judgment-ready candidates plus the build statistics."""

    candidates: list[JudgmentCandidate]
    stats: EvidenceStats

    @property
    def product_ids(self) -> list[str]:
        return [candidate.product_id for candidate in self.candidates]


@dataclass(frozen=True)
class _RelevanceTerm:
    text: str
    weight: int
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class _BudgetOutcome:
    candidates: list[JudgmentCandidate]
    truncated_fields: dict[str, list[str]]
    dropped_fields: dict[str, list[str]]
    dropped_details: list[DroppedDetail]
    estimated_chars: int
    exhausted: bool


def _tokens(value: str) -> tuple[str, ...]:
    folded = unicodedata.normalize("NFKC", value).casefold()
    folded = folded.replace("_", " ").replace("-", " ").replace("/", " ")
    tokens: list[str] = []
    for token in _TOKEN_SPLIT.split(folded):
        token = token.strip()
        if len(token) < 2 or token in _STOPWORDS or token in tokens:
            continue
        tokens.append(token)
    return tuple(tokens)


def _tokens_overlap(left: Sequence[str], right: Sequence[str]) -> bool:
    left_set = set(left)
    right_set = set(right)
    if left_set & right_set:
        return True
    # Long-token containment: "scent" vs "unscented" is a real relevance signal.
    for token in left_set:
        if len(token) < 4:
            continue
        for other in right_set:
            if len(other) < 4:
                continue
            if token in other or other in token:
                return True
    return False


def _detail_score(key: str, value: str, terms: Sequence[_RelevanceTerm]) -> int:
    """Relevance score of one catalog detail (key match > value match)."""
    key_tokens = _tokens(key)
    value_tokens = _tokens(value)
    if not key_tokens and not value_tokens:
        return 0
    score = 0
    for term in terms:
        if _tokens_overlap(key_tokens, term.tokens):
            score += term.weight
        elif _tokens_overlap(value_tokens, term.tokens):
            score += max(1, term.weight // 2)
    return score


def _collect_relevance_terms(
    plan: RecommendationPlan,
    extra_terms: Sequence[str],
) -> list[_RelevanceTerm]:
    """Plan-derived relevance terms, ordered by descending signal weight.

    Hard-constraint fields, high-priority soft preferences, preference priority and
    ranking priority attributes weigh most; historical preferences weigh least. A term
    is a token set, so matching is case-insensitive and separator-insensitive.
    """
    weighted: list[tuple[str, int]] = []
    weighted.extend((constraint.field, 3) for constraint in plan.constraints.hard)
    for preference in plan.constraints.soft:
        weight = 3 if preference.priority is Priority.HIGH else 2
        weighted.append((preference.attribute, weight))
    weighted.extend((attribute, 3) for attribute in plan.preferences.priority)
    weighted.extend((attribute, 2) for attribute in plan.preferences.current)
    weighted.extend((attribute, 1) for attribute in plan.preferences.historical)
    weighted.extend((attribute, 2) for attribute in plan.preferences.negative)
    weighted.extend((attribute, 3) for attribute in plan.ranking.priority_attributes)
    for value in (
        plan.intent.category,
        plan.intent.product_type,
        plan.intent.usage_scenario,
    ):
        if value is not None:
            weighted.append((value, 2))
    weighted.extend((term, 2) for term in extra_terms)

    by_key: dict[str, tuple[str, int]] = {}
    for text, weight in weighted:
        tokens = _tokens(text)
        if not tokens:
            continue
        key = " ".join(tokens)
        current = by_key.get(key)
        if current is None or weight > current[1]:
            by_key[key] = (text.strip(), weight)
    return [
        _RelevanceTerm(text=text, weight=weight, tokens=tuple(key.split(" ")))
        for key, (text, weight) in by_key.items()
    ]


def _select_attributes(
    record: ProductRecord,
    terms: Sequence[_RelevanceTerm],
    max_keys: int,
) -> tuple[dict[str, str], list[DroppedDetail]]:
    """Keep brand/category plus at most ``max_keys`` relevant, real detail attributes."""
    attributes: dict[str, str] = {}
    if record.brand is not None:
        attributes["brand"] = record.brand
    if record.main_category is not None:
        attributes["main_category"] = record.main_category

    dropped: list[DroppedDetail] = []
    scored: list[tuple[int, str, str]] = []
    for raw_key, raw_value in record.details.items():
        key = str(raw_key).strip()
        value = str(raw_value).strip()
        if not key or not value:
            dropped.append(DroppedDetail(record.product_id, key or str(raw_key), "missing_value"))
            continue
        if key.casefold().replace("-", "_").replace(" ", "_") in _RESERVED_ATTRIBUTE_KEYS:
            dropped.append(DroppedDetail(record.product_id, key, "reserved_key"))
            continue
        score = _detail_score(key, value, terms)
        if score <= 0:
            dropped.append(DroppedDetail(record.product_id, key, "irrelevant"))
            continue
        scored.append((score, key, value))

    scored.sort(key=lambda item: (-item[0], item[1].casefold()))
    kept_keys: set[str] = set()
    kept_count = 0
    for _, key, value in scored:
        folded = key.casefold()
        if folded in kept_keys:
            dropped.append(DroppedDetail(record.product_id, key, "duplicate_key"))
            continue
        if kept_count >= max_keys:
            dropped.append(DroppedDetail(record.product_id, key, "max_detail_keys"))
            continue
        kept_keys.add(folded)
        attributes[key] = value
        kept_count += 1
    return attributes, dropped


def _clean_description(description: str | None) -> str | None:
    if description is None:
        return None
    trimmed = description.strip()
    return trimmed or None


def _truncate_text(text: str, limit: int) -> str:
    """Truncate to at most ``limit`` characters, keeping a visible truncation marker."""
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    if limit == 1:
        return TRUNCATION_MARKER
    trimmed = text[: limit - 1].rstrip()
    return f"{trimmed}{TRUNCATION_MARKER}" if trimmed else TRUNCATION_MARKER


def _payload_chars(candidate: JudgmentCandidate) -> int:
    """Character size of the candidate as serialized into the Judgment prompt."""
    return len(candidate.model_dump_json(exclude_none=True))


def _attribute_chars(key: str, value: str) -> int:
    return len(json.dumps({key: value}, ensure_ascii=False))


class JudgmentEvidenceBuilder:
    """Build :class:`JudgmentCandidate` payloads from the Catalog on demand (D6).

    ``catalog`` (an already-loaded mapping) and ``catalog_lookup`` (a batch query
    function, matching D6 "按需查询") are alternatives; the lookup is called exactly once
    per :meth:`build` with the de-duplicated ranked ids in Reranker order.

    Only real catalog fields are copied. Missing products are reported, missing
    description/details stay missing, and ``constraint_status`` is copied from the
    caller's upstream mapping (``None`` when not provided — never inferred).
    """

    def __init__(
        self,
        *,
        catalog: Mapping[str, ProductRecord] | None = None,
        catalog_lookup: CatalogLookup | None = None,
        description_max_chars: int = DEFAULT_DESCRIPTION_MAX_CHARS,
        max_detail_keys: int = DEFAULT_MAX_DETAIL_KEYS,
        max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
        max_total_tokens: int | None = None,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
    ) -> None:
        if catalog is not None and catalog_lookup is not None:
            raise ValueError("provide either catalog or catalog_lookup, not both")
        if description_max_chars < 1:
            raise ValueError("description_max_chars must be >= 1")
        if max_detail_keys < 0:
            raise ValueError("max_detail_keys must be >= 0")
        if max_total_chars < 1:
            raise ValueError("max_total_chars must be >= 1")
        if max_total_tokens is not None and max_total_tokens < 1:
            raise ValueError("max_total_tokens must be >= 1 when provided")
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be >= 1")
        self._catalog = catalog
        self._catalog_lookup = catalog_lookup
        self._description_max_chars = description_max_chars
        self._max_detail_keys = max_detail_keys
        self._max_total_chars = max_total_chars
        self._chars_per_token = chars_per_token
        self._token_limit = max_total_tokens
        self._char_limit = (
            max_total_chars
            if max_total_tokens is None
            else min(max_total_chars, max_total_tokens * chars_per_token)
        )

    @property
    def char_limit(self) -> int:
        """Effective character budget (the stricter of the char and token caps)."""
        return self._char_limit

    @property
    def token_limit(self) -> int | None:
        return self._token_limit

    @property
    def description_max_chars(self) -> int:
        return self._description_max_chars

    @property
    def max_detail_keys(self) -> int:
        return self._max_detail_keys

    def build(
        self,
        product_ids: Sequence[str],
        *,
        plan: RecommendationPlan,
        catalog: Mapping[str, ProductRecord] | None = None,
        catalog_lookup: CatalogLookup | None = None,
        constraint_status_by_id: Mapping[str, ConstraintStatus | None] | None = None,
        extra_terms: Sequence[str] = (),
    ) -> EvidenceBuildResult:
        """Build Judgment candidates for ``product_ids`` in Reranker order."""
        ordered_ids, duplicate_ids = self._ordered_ids(product_ids)
        mapping = self._resolve_catalog(ordered_ids, catalog, catalog_lookup)
        terms = _collect_relevance_terms(plan, extra_terms)

        candidates: list[JudgmentCandidate] = []
        truncated_fields: dict[str, list[str]] = {}
        dropped_details: list[DroppedDetail] = []
        for product_id in ordered_ids:
            record = mapping.get(product_id)
            if record is None or record.product_id != product_id:
                continue
            status = (
                None
                if constraint_status_by_id is None
                else constraint_status_by_id.get(product_id)
            )
            description = _clean_description(record.description)
            if description is not None and len(description) > self._description_max_chars:
                description = _truncate_text(description, self._description_max_chars)
                truncated_fields.setdefault(product_id, []).append("description")
            attributes, dropped = _select_attributes(record, terms, self._max_detail_keys)
            dropped_details.extend(dropped)
            candidates.append(
                JudgmentCandidate(
                    product_id=product_id,
                    title=record.title,
                    price=record.price,
                    description=description,
                    attributes=attributes,
                    constraint_status=status,
                )
            )

        outcome = self._apply_budget(candidates)
        for product_id, fields in outcome.truncated_fields.items():
            existing = truncated_fields.setdefault(product_id, [])
            for field_name in fields:
                if field_name not in existing:
                    existing.append(field_name)
        dropped_details.extend(outcome.dropped_details)

        missing_products = [
            product_id
            for product_id in ordered_ids
            if product_id not in mapping or mapping[product_id].product_id != product_id
        ]
        stats = EvidenceStats(
            requested_products=len(ordered_ids),
            returned_products=len(outcome.candidates),
            missing_products=missing_products,
            duplicate_products=duplicate_ids,
            truncated_fields=truncated_fields,
            dropped_fields=outcome.dropped_fields,
            dropped_details=dropped_details,
            estimated_chars=outcome.estimated_chars,
            char_limit=self._char_limit,
            token_limit=self._token_limit,
            chars_per_token=self._chars_per_token,
            budget_exhausted=outcome.exhausted,
        )
        _LOGGER.info(
            "judgment evidence built",
            extra={
                "requested_products": stats.requested_products,
                "returned_products": stats.returned_products,
                "missing_products": stats.missing_products,
                "duplicate_products": stats.duplicate_products,
                "truncated_fields": stats.truncated_fields,
                "dropped_detail_count": len(stats.dropped_details),
                "estimated_chars": stats.estimated_chars,
                "char_limit": stats.char_limit,
                "token_limit": stats.token_limit,
                "budget_exhausted": stats.budget_exhausted,
            },
        )
        return EvidenceBuildResult(candidates=outcome.candidates, stats=stats)

    # -- input resolution ----------------------------------------------------
    @staticmethod
    def _ordered_ids(product_ids: Sequence[str]) -> tuple[list[str], list[str]]:
        requested = [product_id.strip() for product_id in product_ids]
        if any(not product_id for product_id in requested):
            raise ContractError("product_id must be a non-empty string")
        ordered = list(dict.fromkeys(requested))
        if not ordered:
            raise ContractError("evidence builder requires at least one product_id")
        duplicates: list[str] = []
        seen: set[str] = set()
        for product_id in requested:
            if product_id in seen and product_id not in duplicates:
                duplicates.append(product_id)
            seen.add(product_id)
        return ordered, duplicates

    def _resolve_catalog(
        self,
        product_ids: Sequence[str],
        catalog: Mapping[str, ProductRecord] | None,
        catalog_lookup: CatalogLookup | None,
    ) -> Mapping[str, ProductRecord]:
        if catalog is not None and catalog_lookup is not None:
            raise ValueError("provide either catalog or catalog_lookup, not both")
        if catalog is not None:
            return catalog
        if catalog_lookup is not None:
            return self._call_lookup(catalog_lookup, product_ids)
        if self._catalog is not None:
            return self._catalog
        if self._catalog_lookup is not None:
            return self._call_lookup(self._catalog_lookup, product_ids)
        raise ContractError("catalog mapping or catalog_lookup is required")

    @staticmethod
    def _call_lookup(
        lookup: CatalogLookup,
        product_ids: Sequence[str],
    ) -> Mapping[str, ProductRecord]:
        resolved = lookup(list(product_ids))
        if not isinstance(resolved, Mapping):
            raise ContractError("catalog_lookup must return a Mapping[str, ProductRecord]")
        return resolved

    # -- budget --------------------------------------------------------------
    def _apply_budget(self, candidates: list[JudgmentCandidate]) -> _BudgetOutcome:
        """Enforce the global character budget; record every truncation/drop.

        Priority: core identity fields > relevance-filtered details > description.
        Core fields are never dropped because that would break the one-result-per-
        product alignment. If the core fields alone exceed the budget the overflow is
        reported (``budget_exhausted=True``) instead of silently shrinking the
        candidate set.
        """
        total = sum(_payload_chars(candidate) for candidate in candidates)
        if total <= self._char_limit:
            return _BudgetOutcome(
                candidates=candidates,
                truncated_fields={},
                dropped_fields={},
                dropped_details=[],
                estimated_chars=total,
                exhausted=False,
            )

        core_candidates = [
            candidate.model_copy(
                update={
                    "description": None,
                    "attributes": {
                        key: value
                        for key, value in candidate.attributes.items()
                        if key.casefold() in _RESERVED_ATTRIBUTE_KEYS
                    },
                }
            )
            for candidate in candidates
        ]
        core_total = sum(_payload_chars(candidate) for candidate in core_candidates)
        truncated: dict[str, list[str]] = {}
        dropped_fields: dict[str, list[str]] = {}
        dropped_details: list[DroppedDetail] = []
        if core_total > self._char_limit:
            for candidate in candidates:
                if candidate.description is not None:
                    dropped_fields.setdefault(candidate.product_id, []).append("description")
                for key in candidate.attributes:
                    if key.casefold() not in _RESERVED_ATTRIBUTE_KEYS:
                        dropped_details.append(
                            DroppedDetail(candidate.product_id, key, "budget")
                        )
            return _BudgetOutcome(
                candidates=core_candidates,
                truncated_fields=truncated,
                dropped_fields=dropped_fields,
                dropped_details=dropped_details,
                estimated_chars=core_total,
                exhausted=True,
            )

        remaining = self._char_limit - core_total
        kept_attributes: list[dict[str, str]] = []
        for candidate, core in zip(candidates, core_candidates, strict=True):
            attributes = dict(core.attributes)
            for key, value in candidate.attributes.items():
                if key.casefold() in _RESERVED_ATTRIBUTE_KEYS:
                    continue
                size = _attribute_chars(key, value)
                if size <= remaining:
                    attributes[key] = value
                    remaining -= size
                else:
                    dropped_details.append(
                        DroppedDetail(candidate.product_id, key, "budget")
                    )
            kept_attributes.append(attributes)

        description_count = sum(1 for candidate in candidates if candidate.description)
        available_for_description = (
            remaining - _DESCRIPTION_OVERHEAD_CHARS * description_count
        )
        per_description = (
            available_for_description // description_count
            if description_count and available_for_description > 0
            else 0
        )
        final: list[JudgmentCandidate] = []
        for candidate, core, attributes in zip(
            candidates, core_candidates, kept_attributes, strict=True
        ):
            description = candidate.description
            if description is not None and len(description) > per_description:
                if per_description >= _MIN_BUDGET_DESCRIPTION_CHARS:
                    description = _truncate_text(description, per_description)
                    truncated.setdefault(candidate.product_id, []).append("description")
                else:
                    description = None
                    dropped_fields.setdefault(candidate.product_id, []).append("description")
            final.append(
                core.model_copy(
                    update={"description": description, "attributes": attributes}
                )
            )
        estimated = sum(_payload_chars(candidate) for candidate in final)
        return _BudgetOutcome(
            candidates=final,
            truncated_fields=truncated,
            dropped_fields=dropped_fields,
            dropped_details=dropped_details,
            estimated_chars=estimated,
            exhausted=True,
        )
