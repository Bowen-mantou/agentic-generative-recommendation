"""End-to-end recommendation service (Phase 5/6 integration).

Wires the real E5/FAISS retriever, Fusion + constraints, Qwen3-Reranker, D6 evidence
builder, DeepSeek Judgment, LangGraph Controller, DeepSeek Generator and Mem0 read path
into one command. Target labels and evaluation data are never part of this path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recommender.config import AppConfig, Settings
from recommender.constraints import ConstraintEvaluator  # noqa: F401  (import keeps parity)
from recommender.controller.controller import Controller
from recommender.controller.workflow import (
    RecommendationWorkflow,
    WorkflowDependencies,
    WorkflowOptions,
)
from recommender.fusion import build_candidate_pool, fuse_paths
from recommender.judgment.judge import Judge
from recommender.observability.context import new_request_context, request_context
from recommender.observability.logging import get_logger
from recommender.observability.trace import TraceWriter, trace_path
from recommender.planner.planner import Planner
from recommender.planner.prompts import SystemCapabilities
from recommender.reranking.qwen import Qwen3Reranker
from recommender.reranking.service import RerankService
from recommender.retrieval.e5.artifacts import load_retriever
from recommender.retrieval.e5.encoder import E5Encoder
from recommender.schemas.common import RecallPath, StopReason
from recommender.schemas.controller import RecommendationState
from recommender.schemas.generation import RecommendationItem, RecommendationOutput
from recommender.schemas.plan import RecommendationPlan
from recommender.schemas.product import ProductRecord
from recommender.utils.io import ensure_dir, write_json

_LOGGER = get_logger(__name__)
_DEFAULT_QUERY = "fragrance-free moisturizer for sensitive skin"


@dataclass(frozen=True)
class E2EOptions:
    catalog_dir: Path
    bundle_dir: Path
    trace_dir: Path
    top_k: int = 5
    judgment_top_n: int = 10
    rerank_top_n: int = 50
    memory_enabled: bool = True
    device: str | None = None


@dataclass
class E2EResult:
    request_id: str
    trace_id: str
    trace_path: Path
    summary: dict[str, Any]
    recommendation: RecommendationOutput
    state: RecommendationState


class _NullMemory:
    """Explicit unavailable memory backend; never pretends a write succeeded."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def health(self) -> dict[str, Any]:
        return {"available": False, "backend": "unavailable", "reason": self.reason}

    def read(self, *_: Any, **__: Any) -> list[Any]:
        return []

    def add(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError(f"memory backend unavailable: {self.reason}")


class _FallbackGenerator:
    """Template used only when the real Generator is unavailable or invalid; Gate marks it."""

    def generate(
        self,
        *,
        request_id: str,
        candidates: Sequence[Any],
        stop_reason: StopReason | None,
        quality_satisfied: bool,
        budget_exhausted: bool,
        **_: Any,
    ) -> RecommendationOutput:
        items: list[RecommendationItem] = []
        for rank, candidate in enumerate(candidates[:5], start=1):
            if isinstance(candidate, Mapping):
                product_id = str(candidate["product_id"])
                title = str(candidate.get("title", ""))
                price = candidate.get("price")
            else:
                product_id = str(candidate.product_id)
                title = str(getattr(candidate, "title", ""))
                price = getattr(candidate, "price", None)
            items.append(
                RecommendationItem(
                    rank=rank,
                    product_id=product_id,
                    title=title,
                    price=float(price) if isinstance(price, (int, float)) else None,
                    reasons=["Catalog candidate selected by the deterministic pipeline."],
                    limitations=["Generated without the real DeepSeek Generator."],
                )
            )
        return RecommendationOutput(
            request_id=request_id,
            recommendations=items,
            model="fallback_template",
            stop_reason=stop_reason,
            quality_satisfied=quality_satisfied,
            budget_exhausted=budget_exhausted,
        )


def _encoder(config: AppConfig, device: str | None = None) -> E5Encoder:
    e5 = config.e5
    return E5Encoder(
        model_id=e5.model_id,
        revision=e5.revision,
        device=device or e5.device,
        batch_size=e5.batch_size,
        max_length=e5.max_length,
        query_prefix=e5.query_prefix,
        passage_prefix=e5.passage_prefix,
        normalize=e5.normalize,
        expected_dim=e5.embedding_dim,
    )


def _catalog_products(catalog_dir: Path) -> dict[str, ProductRecord]:
    from recommender.catalog.build import load_products

    return {
        product.product_id: product for product in load_products(catalog_dir / "products.jsonl")
    }


def _catalog_texts(catalog_dir: Path) -> dict[str, str]:
    from recommender.catalog.build import load_text_pairs

    return {
        pair.product_id: pair.rerank_text for pair in load_text_pairs(catalog_dir / "texts.jsonl")
    }


def _memory_adapter(config: AppConfig, enabled: bool) -> tuple[Any, dict[str, Any]]:
    if not enabled:
        return _NullMemory("memory disabled by request"), {
            "available": False,
            "backend": "disabled",
        }
    try:
        from recommender.memory.adapter import create_memory_adapter
    except ImportError as exc:  # pragma: no cover - optional dependency / Phase 6
        return _NullMemory(f"memory adapter not installed: {exc}"), {
            "available": False,
            "backend": "unavailable",
        }
    try:
        memory_config = config.memory.model_copy(update={"enabled": True})
        adapter = create_memory_adapter(memory_config)
        health = adapter.health()
        return adapter, health
    except Exception as exc:
        return _NullMemory(f"{type(exc).__name__}: {exc}"), {
            "available": False,
            "backend": "unavailable",
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }


def _fallback_judgment(state: RecommendationState, product_id: str) -> dict[str, Any] | None:
    for record in state.judgment_results:
        if record.result.product_id == product_id:
            payload = record.result.model_dump(mode="json")
            payload["_model"] = record.model
            payload["_prompt_version"] = record.prompt_version
            return payload
    return None


def _generation_inputs(
    state: RecommendationState, catalog: Mapping[str, ProductRecord]
) -> list[Any]:
    try:
        from recommender.generation.deepseek_generator import GeneratorCandidate

        candidate_cls: Any = GeneratorCandidate
    except ImportError:
        candidate_cls = None

    candidates: list[Any] = []
    for product_id in state.selected_products:
        product = catalog.get(product_id)
        if product is None:
            continue
        judgment = _fallback_judgment(state, product_id)
        evidence: dict[str, Any] = {
            "title": product.title,
            "brand": product.brand,
            "price": product.price,
            "main_category": product.main_category,
            "details": dict(list(product.details.items())[:8]),
        }
        item = {
            "product_id": product.product_id,
            "title": product.title,
            "price": product.price,
            "evidence": evidence,
            "judgment": judgment or {},
        }
        candidates.append(candidate_cls(**item) if candidate_cls else item)
    return candidates


def run_e2e(
    *,
    query: str,
    user_id: str,
    options: E2EOptions,
    settings: Settings,
) -> E2EResult:
    """Run one real end-to-end request. Failures remain explicit; no fabricated output."""
    config = settings.config
    context = new_request_context(user_id=user_id)
    request_id, trace_id = context.request_id, context.trace_id
    directory = ensure_dir(options.trace_dir / request_id)
    trace_file = trace_path(options.trace_dir, request_id)
    with (
        request_context(request_id=request_id, trace_id=trace_id, user_id=user_id),
        TraceWriter(trace_file, request_id=request_id, trace_id=trace_id) as trace,
    ):
        trace.emit(
            "e2e.started", {"query": query, "user_id_hash": _hash_user(user_id)}, stage="e2e"
        )
        started = time.perf_counter()
        memory, memory_health = _memory_adapter(config, options.memory_enabled)
        memory_hits: list[Any] = []
        memory_started = time.perf_counter()
        if memory_health.get("available"):
            try:
                memory_hits = memory.read(user_id, query, limit=5)
            except Exception as exc:
                memory_health = {
                    **memory_health,
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                trace.emit("e2e.memory_read_failed", {"error": memory_health["error"]})
        memory_ms = (time.perf_counter() - memory_started) * 1000
        user_memory = {
            "memories": [
                {
                    "text": str(getattr(hit, "text", "")),
                    "metadata": getattr(hit, "metadata", {}),
                    "score": getattr(hit, "score", None),
                }
                for hit in memory_hits
            ],
            "backend": memory_health.get("backend"),
        }
        trace.emit(
            "e2e.memory_read",
            {
                "health": memory_health,
                "hits": len(memory_hits),
                "latency_ms": round(memory_ms, 1),
            },
            stage="memory",
        )

        from recommender.api.deepseek import DeepSeekClient

        client = DeepSeekClient(settings=settings, reasoning_effort="none", max_concurrency=4)
        try:
            planner = Planner(client, trace=trace)
            judge = Judge(client, trace=trace)
            capabilities = SystemCapabilities(retrieval_modules=[RecallPath.E5_FAISS])

            encoder = _encoder(config, options.device)
            retriever = load_retriever(
                options.bundle_dir,
                encoder,
                top_k_per_query=config.retrieval.e5.top_k_per_query,
                rrf_k=config.retrieval.e5.rrf_k,
                max_queries=config.retrieval.e5.max_queries,
            )
            products = _catalog_products(options.catalog_dir)
            texts = _catalog_texts(options.catalog_dir)
            reranker = Qwen3Reranker(
                model_id=config.reranker.primary_model_id,
                device=options.device or config.reranker.device,
                max_length=config.reranker.max_length,
                batch_size=config.reranker.batch_size,
                instruction=config.reranker.instruction,
            )
            rerank_service = RerankService(reranker, texts)
            from recommender.judgment.evidence import JudgmentEvidenceBuilder

            evidence_builder = JudgmentEvidenceBuilder()
            controller_config = config.controller.model_copy(deep=True)
            controller_config.budget.remaining_replan_rounds = 1
            controller = Controller(controller_config)

            def retrieve(plan: RecommendationPlan) -> Any:
                queries = plan.retrieval.e5_faiss.queries
                trace.emit("e2e.retrieval_started", {"queries": queries}, stage="retrieval")
                output = retriever.search(queries)
                outcome = fuse_paths(
                    [output.fused],
                    rrf_k=config.retrieval.fusion.rrf_k,
                    max_candidates=config.retrieval.fusion.max_candidates,
                )
                pool = build_candidate_pool(
                    request_id=request_id,
                    outcome=outcome,
                    constraints=plan.constraints.hard,
                    products=products,
                    rrf_k=config.retrieval.fusion.rrf_k,
                )
                trace.emit(
                    "e2e.retrieval_completed",
                    {
                        "eligible": len(pool.candidates),
                        "unverified": len(pool.unverified_candidates),
                        "rejected": len(pool.rejected_candidates),
                        "summary": pool.constraint_summary.model_dump(mode="json"),
                    },
                    stage="retrieval",
                )
                return pool

            def rerank(plan: RecommendationPlan, pool: Any) -> Sequence[Any]:
                candidates = list(pool.candidates)[: options.rerank_top_n]
                if not candidates:
                    return []
                result = rerank_service.rerank(
                    plan.ranking.rerank_query,
                    candidates,
                    top_k=options.rerank_top_n,
                )
                trace.emit(
                    "e2e.rerank_completed",
                    {
                        "scored": len(result.ranked),
                        "top": [
                            {
                                "product_id": item.product_id,
                                "score": round(item.rerank_score, 6),
                            }
                            for item in result.ranked[:10]
                        ],
                        "latency_ms": round(result.latency_ms, 1),
                    },
                    stage="rerank",
                )
                return result.ranked

            def candidates_from_pool(
                pool: Any, ranked: Sequence[Any], plan: RecommendationPlan
            ) -> list[Any]:
                top_ids = [item.product_id for item in ranked[: options.judgment_top_n]]
                status = {
                    candidate.product_id: candidate.constraint_status
                    for candidate in [
                        *pool.candidates,
                        *pool.unverified_candidates,
                        *pool.rejected_candidates,
                    ]
                }
                built = evidence_builder.build(
                    top_ids,
                    plan=plan,
                    catalog=products,
                    constraint_status_by_id=status,
                )
                trace.emit(
                    "e2e.evidence_built",
                    {
                        "candidates": len(built.candidates),
                        "missing": built.stats.missing_products[:10],
                        "truncated": built.stats.truncated_fields,
                    },
                    stage="evidence",
                )
                return built.candidates

            state = RecommendationState(
                request_id=request_id,
                trace_id=trace_id,
                user_id=user_id,
                user_request=query,
                user_memory=user_memory,
                budget_state=controller.initial_budget(),
                max_replan_rounds=1,
            )
            workflow = RecommendationWorkflow(
                WorkflowDependencies(
                    planner=planner,
                    judge=judge,
                    retrieve=retrieve,
                    rerank=rerank,
                    candidates_from_pool=candidates_from_pool,
                ),
                options=WorkflowOptions(
                    controller=controller,
                    capabilities=capabilities,
                    select_top_k=options.top_k,
                    use_langgraph=True,
                ),
                trace=trace,
            )
            workflow_started = time.perf_counter()
            state = workflow.run(state)
            workflow_ms = (time.perf_counter() - workflow_started) * 1000

            generator_started = time.perf_counter()
            generator_model = "deepseek-flash"
            try:
                from recommender.generation.deepseek_generator import (
                    RecommendationGenerator,
                )

                generator = RecommendationGenerator(client, config.generation)
                generated = generator.generate(
                    request_id=request_id,
                    user_request=query,
                    candidates=_generation_inputs(state, products),
                    stop_reason=state.stop_reason,
                    quality_satisfied=state.quality_satisfied,
                    budget_exhausted=state.budget_exhausted,
                )
                generator_model = generated.model or generator_model
            except Exception as exc:
                trace.emit(
                    "e2e.generator_failed",
                    {"error": f"{type(exc).__name__}: {exc}"},
                    stage="generation",
                )
                generated = _FallbackGenerator().generate(
                    request_id=request_id,
                    candidates=_generation_inputs(state, products),
                    stop_reason=state.stop_reason,
                    quality_satisfied=state.quality_satisfied,
                    budget_exhausted=state.budget_exhausted,
                )
                generator_model = generated.model or "fallback_template"
            generator_ms = (time.perf_counter() - generator_started) * 1000
            state.recommendation = generated
            trace.emit(
                "e2e.generator_completed",
                {
                    "model": generator_model,
                    "product_ids": generated.product_ids,
                    "latency_ms": round(generator_ms, 1),
                },
                stage="generation",
            )

            total_ms = (time.perf_counter() - started) * 1000
            summary: dict[str, Any] = {
                "request_id": request_id,
                "trace_id": trace_id,
                "user_id_hash": _hash_user(user_id),
                "query": query,
                "memory": {
                    "health": memory_health,
                    "hits": len(memory_hits),
                    "timing_ms": round(memory_ms, 1),
                    "texts": [str(getattr(hit, "text", "")) for hit in memory_hits],
                },
                "plan": (
                    state.current_plan.model_dump(mode="json")
                    if state.current_plan is not None
                    else None
                ),
                "retrieval": {
                    "eligible": len(state.candidate_pool.candidates) if state.candidate_pool else 0,
                    "unverified": len(state.candidate_pool.unverified_candidates)
                    if state.candidate_pool
                    else 0,
                    "rejected": len(state.candidate_pool.rejected_candidates)
                    if state.candidate_pool
                    else 0,
                },
                "ranked_top": [
                    {"product_id": item.product_id, "score": item.rerank_score}
                    for item in state.ranked_products[:10]
                ],
                "judgments": [
                    {
                        "product_id": record.result.product_id,
                        "match": record.result.match_assessment.value,
                        "uncertainty": record.result.uncertainty.value,
                    }
                    for record in state.judgment_results
                ],
                "controller": {
                    "action": state.action.value if state.action else None,
                    "stop_reason": state.stop_reason.value if state.stop_reason else None,
                    "quality_satisfied": state.quality_satisfied,
                    "budget_exhausted": state.budget_exhausted,
                    "selected_products": state.selected_products,
                    "reasons": state.decision.reasons if state.decision else [],
                    "rounds": state.current_round,
                },
                "recommendation": generated.model_dump(mode="json"),
                "generator_model": generator_model,
                "generator_fallback": generator_model == "fallback_template",
                "memory_backend": memory_health.get("backend"),
                "degraded": bool(
                    not memory_health.get("available", False)
                    or generator_model == "fallback_template"
                ),
                "degraded_reason": [
                    reason
                    for reason in (
                        None
                        if memory_health.get("available", False)
                        else f"memory:{memory_health.get('backend', 'unavailable')}",
                        None
                        if generator_model != "fallback_template"
                        else "generator:fallback_template",
                    )
                    if reason is not None
                ],
                "forced_branch_test": False,
                "runtime": "langgraph"
                if importlib.util.find_spec("langgraph") is not None
                else "python_fallback",
                "max_replans_reached": state.current_round >= 2,
                "timing_ms": {
                    "memory": round(memory_ms, 1),
                    "workflow": round(workflow_ms, 1),
                    "generator": round(generator_ms, 1),
                    "total": round(total_ms, 1),
                },
                "trace_path": str(trace_file),
            }
            write_json(directory / "summary.json", summary)
            trace.emit("e2e.completed", summary, stage="e2e")
            _LOGGER.info(
                "e2e completed",
                extra={
                    "request_id": request_id,
                    "stop_reason": summary["controller"]["stop_reason"],
                    "selected": len(state.selected_products),
                    "total_ms": summary["timing_ms"]["total"],
                },
            )
            return E2EResult(
                request_id=request_id,
                trace_id=trace_id,
                trace_path=trace_file,
                summary=summary,
                recommendation=generated,
                state=state,
            )
        finally:
            client.close()


def _hash_user(user_id: str) -> str:
    return hashlib.sha256(user_id.encode()).hexdigest()[:12]
