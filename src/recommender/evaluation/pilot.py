"""End-to-end evaluation-dataset pilot (protocol stages A–D).

This orchestrator builds a temporal test sample set, deterministic profiles, synthetic
queries, independent audits and a human-review queue. It never writes target labels into
profiles/queries/audits and never treats an unapproved REVIEW sample as PASS.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from recommender.api.deepseek import DeepSeekClient
from recommender.config import AppConfig, Settings, load_settings
from recommender.evaluation.profile import ProfileExtractor
from recommender.evaluation.prompts import (
    AUDITOR_PROMPT_HASH,
    GENERATOR_PROMPT_HASH,
)
from recommender.evaluation.query_auditor import QueryAuditor
from recommender.evaluation.query_generator import QueryGenerator
from recommender.evaluation.temporal import (
    CatalogIndex,
    RecoveryStats,
    load_tiger_samples,
    recover_samples,
    save_temporal_audit,
    select_pilot_samples,
)
from recommender.evaluation.validation import validate_query
from recommender.interaction_history.store import InteractionStore
from recommender.observability.logging import get_logger
from recommender.retrieval.tiger.semantic_ids import SemanticIdTable
from recommender.schemas.evaluation_dataset import (
    QueryAudit,
    SyntheticQuery,
    TargetLabel,
    TemporalSample,
    UserProfile,
)
from recommender.utils.io import ensure_dir, file_sha256, utc_now_iso, write_json

_LOGGER = get_logger(__name__)


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = row.model_dump() if hasattr(row, "model_dump") else row
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path, model: Any) -> dict[str, Any]:
    if not path.is_file():
        return {}
    rows: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = model.model_validate_json(line)
            rows[record.sample_id] = record
    return rows


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _catalog_products_path(catalog: Path) -> Path:
    return catalog / "products.jsonl" if catalog.is_dir() else catalog


def build_temporal_dataset(
    *,
    samples_path: Path,
    db_path: Path,
    semantic_ids_dir: Path,
    catalog_path: Path,
    out_dir: Path,
    dataset_version: str,
) -> tuple[list[TemporalSample], list[TargetLabel], RecoveryStats]:
    table = SemanticIdTable.load(semantic_ids_dir)
    catalog = CatalogIndex.load(_catalog_products_path(catalog_path))
    raw = load_tiger_samples(samples_path)
    with InteractionStore(db_path) as store:
        samples, labels, stats = recover_samples(
            raw, store, table, catalog, dataset_version=dataset_version
        )
    save_temporal_audit(
        out_dir,
        samples,
        labels,
        stats,
        source_samples=samples_path,
        source_db=db_path,
    )
    _write_jsonl(out_dir / "samples" / "test_samples.jsonl", list(samples))
    _write_jsonl(out_dir / "labels" / "target_labels.jsonl", list(labels))
    _LOGGER.info(
        "temporal dataset built",
        extra={"out_dir": str(out_dir), "recovered": stats.recovered, "raw": stats.raw_samples},
    )
    return samples, labels, stats


def _load_catalog_index(catalog_path: Path) -> CatalogIndex:
    return CatalogIndex.load(_catalog_products_path(catalog_path))


def _profile_rows(
    samples: list[TemporalSample], db_path: Path, catalog_path: Path, profile_config: Any
) -> list[UserProfile]:
    catalog = _load_catalog_index(catalog_path)
    extractor = ProfileExtractor(catalog.products, profile_config)
    profiles: list[UserProfile] = []
    with InteractionStore(db_path) as store:
        for sample in samples:
            profiles.append(extractor.build(sample, store))
    return profiles


def _token_totals(queries: list[SyntheticQuery], audits: list[QueryAudit]) -> dict[str, int]:
    prompt = completion = 0
    for query in queries:
        prompt += int(query.generation_metadata.get("prompt_tokens") or 0)
        completion += int(query.generation_metadata.get("completion_tokens") or 0)
    for audit in audits:
        prompt += int(audit.audit_metadata.get("prompt_tokens") or 0)
        completion += int(audit.audit_metadata.get("completion_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion}


def _human_queue(
    samples: list[TemporalSample],
    queries: list[SyntheticQuery],
    audits: list[QueryAudit],
    validation: dict[str, Any],
    *,
    seed: int,
    pass_sample_rate: float,
) -> list[dict[str, Any]]:
    by_sample = {sample.sample_id: sample for sample in samples}
    queue: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    for query in queries:
        sample = by_sample[query.sample_id]
        audit = next((item for item in audits if item.sample_id == query.sample_id), None)
        validation_result = validation.get(query.sample_id)
        needs_review = (
            query.status != "generated"
            or audit is None
            or audit.audit_status in {"REVIEW", "FAIL", "ERROR"}
            or (validation_result is not None and not validation_result.get("passed", False))
        )
        row = {
            "sample_id": sample.sample_id,
            "query_id": query.query_id,
            "query": query.query,
            "query_status": query.status,
            "audit_status": audit.audit_status if audit else "MISSING",
            "audit_reason": audit.reason if audit else "",
            "validation_errors": (validation_result or {}).get("errors", []),
            "human_decision": "",
            "human_notes": "",
            "reviewer": "",
            "reviewed_at": "",
        }
        if needs_review:
            queue.append(row)
        elif query.status == "generated" and audit and audit.audit_status == "PASS":
            pass_rows.append(row)
    rng = random.Random(seed)
    sample_size = max(1, round(len(pass_rows) * pass_sample_rate)) if pass_rows else 0
    queue.extend(rng.sample(pass_rows, min(sample_size, len(pass_rows))))
    return queue


def _write_human_review(out_dir: Path, queue: list[dict[str, Any]]) -> None:
    directory = ensure_dir(out_dir / "human_review")
    _write_jsonl(directory / "review_queue.jsonl", queue)
    fieldnames = (
        list(queue[0].keys())
        if queue
        else [
            "sample_id",
            "query_id",
            "query",
            "query_status",
            "audit_status",
            "audit_reason",
            "validation_errors",
            "human_decision",
            "human_notes",
            "reviewer",
            "reviewed_at",
        ]
    )
    with (directory / "human_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in queue:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_pilot(
    *,
    samples_path: Path,
    db_path: Path,
    semantic_ids_dir: Path,
    catalog_path: Path,
    out_dir: Path,
    limit: int | None = None,
    seed: int | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    config: AppConfig = settings.config
    dataset_version = config.evaluation.query.dataset_version
    limit = limit or config.evaluation.query.pilot_size
    seed = config.evaluation.query.seed if seed is None else seed
    directory = ensure_dir(out_dir)

    samples, labels, stats = build_temporal_dataset(
        samples_path=samples_path,
        db_path=db_path,
        semantic_ids_dir=semantic_ids_dir,
        catalog_path=catalog_path,
        out_dir=directory,
        dataset_version=dataset_version,
    )
    pilot_samples = select_pilot_samples(samples, limit, seed=seed)
    profiles = _profile_rows(pilot_samples, db_path, catalog_path, config.evaluation.profile)
    _write_jsonl(directory / "profiles" / "test_profiles.jsonl", profiles)

    client = DeepSeekClient(settings=settings, reasoning_effort="none")
    try:
        generator = QueryGenerator(client, config.evaluation.query)
        auditor = QueryAuditor(client, config.evaluation.query)
        existing_queries = _read_jsonl(directory / "queries" / "test_queries.jsonl", SyntheticQuery)
        queries: list[SyntheticQuery] = []
        for index, profile in enumerate(profiles, start=1):
            cached = existing_queries.get(profile.sample_id)
            query = cached or generator.generate(profile)
            queries.append(query)
            if index % 10 == 0:
                _write_jsonl(directory / "queries" / "test_queries.jsonl", queries)
                _LOGGER.info("query progress", extra={"done": index, "total": len(profiles)})
        _write_jsonl(directory / "queries" / "test_queries.jsonl", queries)

        existing_audits = _read_jsonl(directory / "audits" / "test_audits.jsonl", QueryAudit)
        profile_by_sample = {profile.sample_id: profile for profile in profiles}
        audits: list[QueryAudit] = []
        final_queries: list[SyntheticQuery] = []
        for index, query in enumerate(queries, start=1):
            profile = profile_by_sample[query.sample_id]
            cached = existing_audits.get(query.sample_id)
            audit = cached or auditor.audit(profile, query)
            # Protocol §6.4/§9: regenerate once after a FAIL, then audit the replacement.
            if audit.audit_status == "FAIL" and cached is None:
                replacement = generator.generate(
                    profile, repair_hint=audit.reason or "unsupported claims"
                )
                if replacement.status == "generated":
                    query = replacement
                    audit = auditor.audit(profile, query)
            audits.append(audit)
            final_queries.append(query)
            if index % 10 == 0:
                _write_jsonl(directory / "audits" / "test_audits.jsonl", audits)
                _LOGGER.info("audit progress", extra={"done": index, "total": len(queries)})
        queries = final_queries
        _write_jsonl(directory / "queries" / "test_queries.jsonl", queries)
        _write_jsonl(directory / "audits" / "test_audits.jsonl", audits)
    finally:
        client.close()

    labels_by_sample = {label.sample_id: label for label in labels}
    validation: dict[str, Any] = {}
    audit_by_sample = {audit.sample_id: audit for audit in audits}
    for query in queries:
        label = labels_by_sample[query.sample_id]
        result = validate_query(query, profile_by_sample[query.sample_id], label)
        validation[query.sample_id] = result.model_dump()
    _write_jsonl(directory / "validation" / "query_validation.jsonl", list(validation.values()))

    audit_counts = Counter(audit.audit_status for audit in audits)
    query_counts = Counter(query.status for query in queries)
    pass_valid = [
        query.sample_id
        for query in queries
        if query.status == "generated"
        and audit_by_sample.get(query.sample_id) is not None
        and audit_by_sample[query.sample_id].audit_status == "PASS"
        and validation[query.sample_id]["passed"]
    ]
    queue = _human_queue(
        pilot_samples,
        queries,
        audits,
        validation,
        seed=seed,
        pass_sample_rate=config.evaluation.query.pass_sample_rate,
    )
    _write_human_review(directory, queue)

    leakage_violations = [
        row
        for row in validation.values()
        if row["errors"] and any("target" in error for error in row["errors"])
    ]
    coverage = {
        "raw_test_samples": stats.raw_samples,
        "recovered_samples": stats.recovered,
        "recovery_failures": dict(stats.failures),
        "pilot_selected": len(pilot_samples),
        "profiles_built": len(profiles),
        "query_status": dict(query_counts),
        "audit_status": dict(audit_counts),
        "validation_passed": sum(1 for row in validation.values() if row["passed"]),
        "final_pass_unreviewed": len(pass_valid),
        "human_review_pending": len(queue),
        "final_frozen_eligible": 0,
        "note": (
            "REVIEW/FAIL/ERROR and sampled PASS rows await human review; "
            "they are not frozen as PASS."
        ),
    }
    lengths = [len(query.query.split()) for query in queries if query.query]
    quality = {
        "query_length_words": {
            "min": min(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "used_fact_ids_total": sum(len(query.used_fact_ids) for query in queries),
        "unsupported_claims_total": sum(len(audit.unsupported_claims) for audit in audits),
        "audit_status": dict(audit_counts),
        "query_status": dict(query_counts),
        "tokens": _token_totals(queries, audits),
        "human_review_rows": len(queue),
    }
    leakage = {
        "checked_queries": sum(
            1
            for row in validation.values()
            if row["target_id_scanned"] or row["target_title_scanned"]
        ),
        "target_id_scanned": sum(1 for row in validation.values() if row["target_id_scanned"]),
        "target_title_scanned": sum(
            1 for row in validation.values() if row["target_title_scanned"]
        ),
        "violations": leakage_violations,
        "violations_count": len(leakage_violations),
    }
    write_json(directory / "reports" / "coverage_report.json", coverage)
    write_json(directory / "reports" / "quality_report.json", quality)
    write_json(directory / "reports" / "leakage_check.json", leakage)

    def sha(relative: str) -> str | None:
        path = directory / relative
        return file_sha256(path) if path.is_file() else None

    manifest = {
        "dataset_version": dataset_version,
        "created_at": utc_now_iso(),
        "source_dataset": "Amazon Reviews 2023",
        "category": "All_Beauty",
        "split_method": "temporal",
        "profile_version": config.evaluation.profile.version,
        "query_generator": {
            "model": generator.model,
            "prompt_version": generator.prompt_version,
            "prompt_hash": GENERATOR_PROMPT_HASH,
            "temperature": config.evaluation.query.temperature,
        },
        "query_auditor": {
            "model": auditor.model,
            "prompt_version": auditor.prompt_version,
            "prompt_hash": AUDITOR_PROMPT_HASH,
        },
        "embedding_model": config.e5.model_id,
        "catalog_version": _load_catalog_index(catalog_path).version,
        "random_seed": seed,
        "target_isolated": True,
        "is_synthetic_query": True,
        "counts": coverage,
        "files": {
            "test_samples": sha("samples/test_samples.jsonl"),
            "target_labels": sha("labels/target_labels.jsonl"),
            "test_profiles": sha("profiles/test_profiles.jsonl"),
            "test_queries": sha("queries/test_queries.jsonl"),
            "test_audits": sha("audits/test_audits.jsonl"),
            "coverage_report": sha("reports/coverage_report.json"),
            "leakage_check": sha("reports/leakage_check.json"),
            "quality_report": sha("reports/quality_report.json"),
            "human_review_queue": sha("human_review/review_queue.jsonl"),
        },
        "code_commit": _git_commit(config.project_root),
        "human_review_completed": False,
    }
    write_json(directory / "manifests" / "evaluation_manifest.json", manifest)

    report_lines = [
        "# Evaluation Dataset Pilot Report",
        "",
        f"- dataset_version: `{dataset_version}`",
        f"- raw test samples: {stats.raw_samples}",
        f"- recovered: {stats.recovered}",
        f"- pilot samples: {len(pilot_samples)}",
        f"- query status: {dict(query_counts)}",
        f"- audit status: {dict(audit_counts)}",
        f"- validation passed: {coverage['validation_passed']}",
        f"- PASS & validation-pass (unreviewed): {coverage['final_pass_unreviewed']}",
        f"- human review pending: {len(queue)}",
        f"- leakage violations: {len(leakage_violations)}",
        "",
        "## Honest status",
        "",
        "This is a pilot. REVIEW/FAIL/ERROR and the sampled PASS rows await human review; "
        "the dataset is **not frozen as a final test set** until that review is completed.",
        "",
        "## Target isolation",
        "",
        "Profiles/queries were built without reading target labels; the validator scanned "
        "for exact target product IDs and full target titles.",
    ]
    (directory / "PILOT_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8", newline="\n"
    )
    _LOGGER.info("pilot finished", extra={"final_pass": len(pass_valid), "review": len(queue)})
    return {
        "out_dir": str(directory),
        "coverage": coverage,
        "quality": quality,
        "leakage": leakage,
        "manifest": manifest,
    }
