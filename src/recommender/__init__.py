"""Agentic Generative Recommendation.

Memory-aware generative product recommendation: Planner -> TIGER / E5+FAISS ->
Candidate Fusion -> Constraint Evaluator -> Reranker -> Judgment -> Controller ->
Generator -> Memory.

Public modules are imported lazily by callers; importing this package must not load
models, open network connections or create files (AGENTS.md §4).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
