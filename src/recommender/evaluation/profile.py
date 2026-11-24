"""Deterministic Python profile extraction for dataset B (protocol §4).

Profile V1 uses only pre-cutoff product metadata, stored ratings and timestamps. It never
reads review bodies, target labels or future events. Every fact carries source event IDs,
support counts and a confidence assigned by fixed rules in ``EvaluationProfileConfig``.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from recommender.config import EvaluationProfileConfig
from recommender.interaction_history.store import InteractionStore
from recommender.observability.logging import get_logger
from recommender.schemas.evaluation_dataset import (
    Confidence,
    FactKind,
    ProfileFact,
    TemporalSample,
    UserProfile,
)
from recommender.schemas.product import ProductRecord

_LOGGER = get_logger(__name__)

# Small deterministic beauty-type taxonomy (v1). A product can match multiple types.
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "moisturizer": (
        "moisturizer",
        "moisturiser",
        "face cream",
        "body cream",
        "lotion",
        "emollient",
    ),
    "cleanser": ("cleanser", "face wash", "facial wash", "body wash", "soap", "shampoo"),
    "serum": ("serum", "essence", "ampoule"),
    "sunscreen": ("sunscreen", "sunblock", "spf", "sun protection"),
    "shampoo": ("shampoo",),
    "conditioner": ("conditioner", "hair mask", "hair treatment"),
    "mask": ("face mask", "clay mask", "sheet mask", "peel", "exfoliant", "scrub"),
    "makeup": (
        "makeup",
        "lipstick",
        "lip gloss",
        "foundation",
        "mascara",
        "eyeliner",
        "blush",
        "concealer",
        "eyeshadow",
    ),
    "fragrance": ("perfume", "fragrance", "cologne", "body mist", "eau de"),
    "haircare": ("hair oil", "hair serum", "hair spray", "styling gel", "leave-in"),
    "tools": ("brush", "comb", "sponge", "roller", "applicator"),
    "toner": ("toner", "astringent"),
    "lipcare": ("lip balm", "lip care", "chapstick"),
    "nailcare": ("nail polish", "nail care", "cuticle"),
}

_ATTRIBUTE_KEYS = {
    "scent",
    "skin type",
    "hair type",
    "item form",
    "color",
    "material",
    "flavor",
    "target audience",
    "formulation",
    "active ingredients",
    "ingredients",
}
_ATTRIBUTE_BLOCKLIST = {
    "package dimensions",
    "upc",
    "asin",
    "isbn",
    "item weight",
    "number of items",
    "unit count",
    "manufacturer",
    "part number",
    "best sellers rank",
    "date first available",
}

_WORD_BOUNDARY = re.compile(r"[^a-z0-9]+")


def _normalise(value: str) -> str:
    return " ".join(value.lower().split())


def _detect_categories(title: str, details: dict[str, str]) -> set[str]:
    text = _normalise(title)
    for key, value in details.items():
        if key.lower() in {"item form", "formulation"}:
            text += " " + _normalise(value)
    tokens = set(_WORD_BOUNDARY.split(text))
    matched: set[str] = set()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if " " in keyword:
                if keyword in text:
                    matched.add(category)
                    break
            elif keyword in tokens:
                matched.add(category)
                break
    return matched


def _confidence(
    positive: int,
    negative: int,
    *,
    high_support: int,
    medium_support: int,
) -> Confidence:
    if positive >= high_support and negative == 0:
        return "high"
    if positive >= medium_support and negative <= 1:
        return "medium"
    return "low"


@dataclass
class _Counter:
    positive: int = 0
    negative: int = 0
    source_event_ids: list[str] = field(default_factory=list)

    def add(self, event_id: str, *, positive: bool) -> None:
        if positive:
            self.positive += 1
        else:
            self.negative += 1
        self.source_event_ids.append(event_id)


def _fact_id(profile_id: str, kind: str, value: str) -> str:
    digest = hashlib.sha256(f"{profile_id}|{kind}|{value}".encode()).hexdigest()[:12]
    return f"fact_{digest}"


def _profile_id(sample_id: str) -> str:
    return "profile_" + hashlib.sha256(sample_id.encode()).hexdigest()[:12]


def _make_facts(
    profile_id: str,
    kind: FactKind,
    counters: dict[str, _Counter],
    *,
    config: EvaluationProfileConfig,
    limit: int,
) -> list[ProfileFact]:
    facts: list[ProfileFact] = []
    ranked = sorted(
        counters.items(),
        key=lambda item: (-(item[1].positive + item[1].negative), item[0]),
    )
    for value, counter in ranked[:limit]:
        support = counter.positive + counter.negative
        if support <= 0:
            continue
        facts.append(
            ProfileFact(
                fact_id=_fact_id(profile_id, kind, value),
                kind=kind,
                value=value,
                support_count=support,
                positive_count=counter.positive,
                negative_count=counter.negative,
                source_event_ids=sorted(set(counter.source_event_ids)),
                confidence=_confidence(
                    counter.positive if kind != "negative" else counter.negative,
                    counter.negative if kind != "negative" else 0,
                    high_support=config.high_support,
                    medium_support=config.medium_support,
                ),
            )
        )
    return facts


class ProfileExtractor:
    """Build one deterministic profile from a sample's pre-cutoff events."""

    def __init__(self, catalog: dict[str, ProductRecord], config: EvaluationProfileConfig) -> None:
        self._catalog = catalog
        self._config = config

    def build(self, sample: TemporalSample, store: InteractionStore) -> UserProfile:
        profile_id = _profile_id(sample.sample_id)
        events = store.events_for_user(sample.user_id, cutoff_ts=sample.cutoff_timestamp)
        by_id = {event.event_id: event for event in events}
        history_events = [
            by_id[event_id] for event_id in sample.history_event_ids if event_id in by_id
        ]
        if len(history_events) != len(sample.history_event_ids):
            missing = sorted(set(sample.history_event_ids) - set(by_id))
            _LOGGER.warning(
                "profile history events missing",
                extra={"sample_id": sample.sample_id, "missing": missing[:5]},
            )

        category: dict[str, _Counter] = defaultdict(_Counter)
        brand: dict[str, _Counter] = defaultdict(_Counter)
        attributes: dict[str, _Counter] = defaultdict(_Counter)
        negative: dict[str, _Counter] = defaultdict(_Counter)
        positive_prices: int = 0

        for event in history_events:
            product = self._catalog.get(event.product_id)
            if product is None:
                continue
            rating = event.rating
            positive = rating is not None and rating >= 4
            negative_rating = rating is not None and rating <= 2
            # Neutral / missing ratings are not interpreted as preference.
            if not positive and not negative_rating:
                continue

            if positive and product.price is not None:
                positive_prices += 1
            categories = _detect_categories(product.title, product.details)
            for category_name in categories:
                counter = category if positive else negative
                counter[category_name].add(event.event_id, positive=positive)
            if product.brand:
                counter = brand if positive else negative
                counter[product.brand].add(event.event_id, positive=positive)
            for key, value in product.details.items():
                normalised_key = key.strip().lower()
                if normalised_key in _ATTRIBUTE_BLOCKLIST or normalised_key not in _ATTRIBUTE_KEYS:
                    continue
                if not value or len(value) > 80:
                    continue
                attribute_value = f"{normalised_key}: {_normalise(value)}"
                counter = attributes if positive else negative
                counter[attribute_value].add(event.event_id, positive=positive)

        unknown: list[str] = []
        detail_keys = {
            key.strip().lower()
            for event in history_events
            for product in [self._catalog.get(event.product_id)]
            if product is not None
            for key in product.details
        }
        if positive_prices == 0:
            unknown.append("budget")
        if "scent" not in detail_keys:
            unknown.append("scent")
        if "skin type" not in detail_keys:
            unknown.append("skin type")
        if "hair type" not in detail_keys:
            unknown.append("hair type")

        return UserProfile(
            profile_id=profile_id,
            sample_id=sample.sample_id,
            user_id=sample.user_id,
            cutoff_timestamp=sample.cutoff_timestamp,
            profile_version=self._config.version,
            history_length=len(history_events),
            category_interests=_make_facts(
                profile_id, "category", category, config=self._config, limit=self._config.max_facts
            ),
            brand_interests=_make_facts(
                profile_id, "brand", brand, config=self._config, limit=self._config.max_facts
            ),
            attribute_interests=_make_facts(
                profile_id,
                "attribute",
                attributes,
                config=self._config,
                limit=self._config.max_facts,
            ),
            negative_signals=_make_facts(
                profile_id, "negative", negative, config=self._config, limit=self._config.max_facts
            ),
            unknown_preferences=unknown[: self._config.max_unknown],
        )


def profile_to_input(profile: UserProfile) -> dict[str, object]:
    """The only profile payload ever sent to generators/auditors (no target fields)."""

    def facts(items: Iterable[ProfileFact]) -> list[dict[str, object]]:
        return [
            {
                "fact_id": fact.fact_id,
                "kind": fact.kind,
                "value": fact.value,
                "support_count": fact.support_count,
                "positive_count": fact.positive_count,
                "negative_count": fact.negative_count,
                "confidence": fact.confidence,
            }
            for fact in items
        ]

    return {
        "profile_id": profile.profile_id,
        "sample_id": profile.sample_id,
        "cutoff_timestamp": profile.cutoff_timestamp,
        "profile_version": profile.profile_version,
        "history_length": profile.history_length,
        "category_interests": facts(profile.category_interests),
        "brand_interests": facts(profile.brand_interests),
        "attribute_interests": facts(profile.attribute_interests),
        "negative_signals": facts(profile.negative_signals),
        "unknown_preferences": list(profile.unknown_preferences),
    }
