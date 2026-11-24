"""DeepSeek recommendation Generator (Phase 6).

Importing this package does not create a DeepSeek client and performs no network call.
"""

from __future__ import annotations

from recommender.generation.deepseek_generator import (
    FALLBACK_MODEL_NAME,
    GENERATOR_PROMPT_VERSION,
    GeneratorCandidate,
    GeneratorOutputError,
    RecommendationGenerator,
    build_generator_messages,
    candidate_evidence_paths,
)

__all__ = [
    "FALLBACK_MODEL_NAME",
    "GENERATOR_PROMPT_VERSION",
    "GeneratorCandidate",
    "GeneratorOutputError",
    "RecommendationGenerator",
    "build_generator_messages",
    "candidate_evidence_paths",
]
