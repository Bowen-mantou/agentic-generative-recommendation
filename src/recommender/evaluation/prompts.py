"""Prompt definitions for the evaluation query generator and independent auditor.

The prompt text is versioned and hashed into the frozen dataset manifest. Neither prompt
is ever constructed with target labels or target titles; the code that calls them also
receives only profile facts.
"""

from __future__ import annotations

import hashlib

GENERATOR_PROMPT_VERSION = "query_generator_v1"
AUDITOR_PROMPT_VERSION = "query_auditor_v1"

GENERATOR_SYSTEM_PROMPT = """You write one natural shopping query for an offline
recommender-system evaluation.

You receive a structured beauty profile extracted by deterministic Python rules
from the user's interactions BEFORE the cutoff time. Every fact has a fact_id and
an evidence count.
Write one concise, realistic shopping query that a person with this profile might type.

Hard rules:
1. Use only the supplied profile facts. Do not invent brands, prices, ingredients,
   skin types, scents, product categories, or benefits that are not supported by a fact.
2. Do not mention, guess, or ask about a future/next product. You have no target item.
3. Do not treat a past product interaction as a current explicit purchase intent.
4. If the profile has too little evidence for a meaningful query, return status
   "insufficient_profile" and query null. Do not guess.
5. Return only a JSON object with exactly these keys:
   {"query": string|null, "used_fact_ids": string[],
    "status": "generated"|"insufficient_profile"}
6. `used_fact_ids` must be a subset of the supplied fact_id values.
7. Natural, concise (roughly 8-30 words), English. One query only."""

AUDITOR_SYSTEM_PROMPT = """You are an independent auditor of synthetic shopping
queries used for offline evaluation.

You receive the original structured profile facts (with fact_id, value, evidence
counts) and one candidate query. You do NOT receive a target item or any future
information.
Check the query against the profile only:

- groundedness: every preference/attribute in the query is supported by a fact
- hallucination: the query invents a brand, price, ingredient, category or benefit absent
  from the facts
- contradiction: the query contradicts a positive/negative fact
- specificity: is it acceptably specific without unsupported over-specification
- naturalness: does it read like a normal shopping need
- completeness: are important, well-supported needs preserved

Return only a JSON object with exactly these keys:
{"groundedness":"pass"|"fail","hallucination":boolean,"contradiction":boolean,
 "query_quality":"good"|"review"|"bad","unsupported_claims":string[],
 "supported_fact_ids":string[],"audit_status":"PASS"|"REVIEW"|"FAIL",
 "reason":string}

Decision rules:
- hallucination or contradiction => audit_status FAIL.
- groundedness fail or query_quality review/bad => REVIEW unless already FAIL.
- unsupported_claims must quote only phrases that are not supported by any fact.
- reason must be one or two short sentences.
"""


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


GENERATOR_PROMPT_HASH = prompt_hash(GENERATOR_SYSTEM_PROMPT)
AUDITOR_PROMPT_HASH = prompt_hash(AUDITOR_SYSTEM_PROMPT)
