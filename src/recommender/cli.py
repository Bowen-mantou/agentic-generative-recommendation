"""``agr`` command-line entry point.

Phase 0/1 commands:

* ``version`` / ``config show|validate``
* ``catalog download|build`` — Amazon Reviews 2023 metadata -> versioned catalog
* ``index build|info`` — catalog -> E5 + FAISS bundle
* ``search`` — real retrieval over a built bundle
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from recommender import __version__
from recommender.catalog.amazon import download_file, metadata_url, reviews_url
from recommender.catalog.build import build_catalog, load_products, load_text_pairs
from recommender.config import AppConfig, Settings, load_settings
from recommender.fusion import build_candidate_pool, fuse_paths
from recommender.interaction_history import (
    InteractionStore,
    build_interaction_history,
    sequence_length_summary,
)
from recommender.observability.context import new_request_context
from recommender.observability.logging import configure_logging, get_logger
from recommender.reranking import Qwen3Reranker, RerankService
from recommender.retrieval.e5.artifacts import build_bundle, load_retriever, verify_bundle
from recommender.retrieval.e5.encoder import E5Encoder
from recommender.schemas.common import ConstraintOperator
from recommender.schemas.plan import HardConstraint

_LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agr", description="Agentic Generative Recommendation")
    parser.add_argument("--config-dir", type=Path, default=None, help="override configs/ directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="print the package version")

    config_parser = subparsers.add_parser("config", help="inspect the loaded configuration")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("show", help="print the resolved configuration as JSON")
    config_sub.add_parser("validate", help="validate configs and report resolved paths")

    catalog_parser = subparsers.add_parser("catalog", help="Amazon catalog ingestion")
    catalog_sub = catalog_parser.add_subparsers(dest="catalog_command", required=True)
    download = catalog_sub.add_parser("download", help="download raw category metadata")
    download.add_argument("--category", required=True)
    download.add_argument("--dest", type=Path, default=None)
    download.add_argument("--endpoint", default=None, help="HF endpoint (hf-mirror.com)")
    download.add_argument("--force", action="store_true")

    build = catalog_sub.add_parser("build", help="build a versioned catalog from metadata")
    build.add_argument("--meta", type=Path, required=True)
    build.add_argument("--category", required=True)
    build.add_argument("--version", default=None)
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--out", type=Path, default=None)

    history_parser = subparsers.add_parser(
        "history", help="InteractionHistory SQLite store (reviews -> events)"
    )
    history_sub = history_parser.add_subparsers(dest="history_command", required=True)
    history_download = history_sub.add_parser("download", help="download raw review file")
    history_download.add_argument("--category", required=True)
    history_download.add_argument("--dest", type=Path, default=None)
    history_download.add_argument("--force", action="store_true")
    history_build = history_sub.add_parser("build", help="build InteractionEvent rows")
    history_build.add_argument("--reviews", type=Path, required=True)
    history_build.add_argument("--db", type=Path, default=None)
    history_build.add_argument("--catalog", type=Path, default=None)
    history_build.add_argument("--limit", type=int, default=None)
    history_stats = history_sub.add_parser("stats", help="event and sequence statistics")
    history_stats.add_argument("--db", type=Path, default=None)
    history_stats.add_argument("--min-events", type=int, default=3)
    history_stats.add_argument("--cutoff-ts", type=int, default=None)

    index_parser = subparsers.add_parser("index", help="E5 + FAISS index bundles")
    index_sub = index_parser.add_subparsers(dest="index_command", required=True)
    index_build = index_sub.add_parser("build", help="encode a catalog and build FAISS")
    index_build.add_argument("--catalog", type=Path, required=True)
    index_build.add_argument("--out", type=Path, default=None)
    index_build.add_argument("--version", default=None)
    index_build.add_argument("--device", default=None)
    index_build.add_argument("--batch-size", type=int, default=None)
    index_info = index_sub.add_parser("info", help="verify a bundle and print its manifest")
    index_info.add_argument("--bundle", type=Path, required=True)

    tiger = subparsers.add_parser("tiger", help="TIGER RQ-VAE / Semantic IDs (Phase 4)")
    tiger_sub = tiger.add_subparsers(dest="tiger_command", required=True)
    tiger_rqvae = tiger_sub.add_parser("rqvae", help="train RQ-VAE and build Semantic IDs")
    tiger_rqvae.add_argument("--bundle", type=Path, required=True)
    tiger_rqvae.add_argument("--config", type=Path, required=True)
    tiger_rqvae.add_argument("--out", type=Path, default=None)
    tiger_rqvae.add_argument("--epochs", type=int, default=None)
    tiger_rqvae.add_argument("--device", default=None)
    tiger_semantic = tiger_sub.add_parser("semantic-ids", help="rebuild IDs from a checkpoint")
    tiger_semantic.add_argument("--bundle", type=Path, required=True)
    tiger_semantic.add_argument("--checkpoint-dir", type=Path, required=True)
    tiger_semantic.add_argument("--out", type=Path, default=None)
    tiger_sequences = tiger_sub.add_parser("sequences", help="build time-split TIGER samples")
    tiger_sequences.add_argument("--db", type=Path, default=None)
    tiger_sequences.add_argument("--semantic-ids", type=Path, required=True)
    tiger_sequences.add_argument("--out", type=Path, default=None)
    tiger_sequences.add_argument("--name", default="a1")
    tiger_sequences.add_argument("--min-events", type=int, default=3)
    tiger_sequences.add_argument("--min-history", type=int, default=2)
    tiger_sequences.add_argument("--max-history-items", type=int, default=10)
    tiger_generator = tiger_sub.add_parser("generator", help="train the AR generator")
    tiger_generator.add_argument("--samples-dir", type=Path, required=True)
    tiger_generator.add_argument("--semantic-ids", type=Path, required=True)
    tiger_generator.add_argument("--config", type=Path, required=True)
    tiger_generator.add_argument("--out", type=Path, default=None)
    tiger_generator.add_argument("--epochs", type=int, default=None)
    tiger_evaluate = tiger_sub.add_parser("evaluate", help="next-item metrics on time split")
    tiger_evaluate.add_argument("--generator-dir", type=Path, required=True)
    tiger_evaluate.add_argument("--samples-dir", type=Path, required=True)
    tiger_evaluate.add_argument("--semantic-ids", type=Path, required=True)
    tiger_evaluate.add_argument("--db", type=Path, default=None)
    tiger_evaluate.add_argument("--split", default="test")
    tiger_evaluate.add_argument("--out", type=Path, default=None)
    tiger_evaluate.add_argument("--beam-size", type=int, default=None)
    tiger_evaluate.add_argument(
        "--checkpoint", default="final", choices=["final", "best"], help="which checkpoint"
    )

    tiger_info = tiger_sub.add_parser("info", help="print RQ-VAE / Semantic ID manifests")
    tiger_info.add_argument("--dir", type=Path, required=True)

    evaluation = subparsers.add_parser("eval", help="Phase 7 evaluation dataset")
    evaluation_sub = evaluation.add_subparsers(dest="eval_command", required=True)
    eval_audit = evaluation_sub.add_parser("audit", help="recover and audit temporal samples")
    eval_audit.add_argument("--samples", type=Path, required=True)
    eval_audit.add_argument("--db", type=Path, default=None)
    eval_audit.add_argument("--semantic-ids", type=Path, required=True)
    eval_audit.add_argument("--catalog", type=Path, default=None)
    eval_audit.add_argument("--out", type=Path, default=None)
    eval_pilot = evaluation_sub.add_parser("pilot", help="profiles -> queries -> audits")
    eval_pilot.add_argument("--samples", type=Path, required=True)
    eval_pilot.add_argument("--db", type=Path, default=None)
    eval_pilot.add_argument("--semantic-ids", type=Path, required=True)
    eval_pilot.add_argument("--catalog", type=Path, default=None)
    eval_pilot.add_argument("--out", type=Path, default=None)
    eval_pilot.add_argument("--limit", type=int, default=None)
    eval_pilot.add_argument("--seed", type=int, default=None)

    pool = subparsers.add_parser("pool", help="retrieve -> fuse -> constraints -> pool")
    pool.add_argument("--bundle", type=Path, required=True)
    pool.add_argument("--catalog", type=Path, required=True)
    pool.add_argument("--query", action="append", required=True)
    pool.add_argument(
        "--constraint",
        action="append",
        default=None,
        help="field:operator:value[:unit], e.g. price:lte:30:USD",
    )
    pool.add_argument("--top-k", type=int, default=10)
    pool.add_argument("--per-query-k", type=int, default=None)
    pool.add_argument("--device", default=None)

    rerank = subparsers.add_parser("rerank", help="pool -> eligible -> Qwen3 rerank")
    rerank.add_argument("--bundle", type=Path, required=True)
    rerank.add_argument("--catalog", type=Path, required=True)
    rerank.add_argument("--query", action="append", required=True)
    rerank.add_argument("--constraint", action="append", default=None)
    rerank.add_argument("--rerank-query", default=None)
    rerank.add_argument("--top-k", type=int, default=5)
    rerank.add_argument("--per-query-k", type=int, default=None)
    rerank.add_argument("--device", default=None)

    search = subparsers.add_parser("search", help="query a built E5 bundle")
    search.add_argument("--bundle", type=Path, required=True)
    search.add_argument("--query", action="append", required=True)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--per-query-k", type=int, default=None)
    search.add_argument("--device", default=None)
    search.add_argument("--catalog", type=Path, default=None)
    search.add_argument("--with-texts", action="store_true")
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    return load_settings(config_dir=args.config_dir)


def _configure_logging(config: AppConfig) -> None:
    configure_logging(
        config.logging.level,
        json_output=config.logging.json_output,
        include_context=config.logging.include_context,
    )


def _cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_config_show(args: argparse.Namespace) -> int:
    settings = _settings(args)
    configure_logging(
        settings.config.logging.level,
        json_output=False,
        include_context=settings.config.logging.include_context,
    )
    print(json.dumps(settings.config.public_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_config_validate(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    configure_logging(config.logging.level, json_output=False)
    resolved = config.paths.resolved(config.project_root)
    print(f"config valid: {config.schema_version}")
    for name, path in resolved.items():
        state = "exists" if path.exists() else "missing"
        print(f"  {name:<16} {path}  [{state}]")
    print(f"  deepseek key      {'set' if settings.has_deepseek_api_key() else 'not set'}")
    return 0


def _cmd_catalog_download(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    destination = args.dest or (config.path("raw_dir") / f"meta_{args.category}.jsonl")
    url = metadata_url(args.category, endpoint=args.endpoint)
    result = download_file(url, destination, force=args.force)
    print(
        json.dumps(
            {
                "category": args.category,
                "path": str(result.path),
                "url": result.url,
                "size_bytes": result.size_bytes,
                "sha256": result.sha256,
            },
            indent=2,
        )
    )
    return 0


def _cmd_catalog_build(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    version = args.version or f"{args.category.lower()}_v1"
    out_root = args.out or config.path("catalog_dir")
    result = build_catalog(
        args.meta,
        out_root,
        catalog_version=version,
        category=args.category,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "catalog_dir": str(result.catalog_dir),
                "record_count": result.record_count,
                "manifest": result.manifest.model_dump(mode="json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _make_encoder(
    config: AppConfig, *, device: str | None = None, batch_size: int | None = None
) -> E5Encoder:
    e5 = config.e5
    return E5Encoder(
        model_id=e5.model_id,
        revision=e5.revision,
        device=device or e5.device,
        batch_size=batch_size or e5.batch_size,
        max_length=e5.max_length,
        query_prefix=e5.query_prefix,
        passage_prefix=e5.passage_prefix,
        normalize=e5.normalize,
        expected_dim=e5.embedding_dim,
    )


def _history_db_path(config: AppConfig, override: Path | None) -> Path:
    return override or config.resolve(config.memory.sqlite.path)


def _cmd_history_download(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    category = args.category
    destination = args.dest or (config.path("raw_dir") / f"review_{category}.jsonl")
    result = download_file(reviews_url(category), destination, force=args.force)
    print(
        json.dumps(
            {
                "category": category,
                "path": str(result.path),
                "url": result.url,
                "size_bytes": result.size_bytes,
                "sha256": result.sha256,
            },
            indent=2,
        )
    )
    return 0


def _cmd_history_build(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    db_path = _history_db_path(config, args.db)
    catalog_path = args.catalog
    if catalog_path is None:
        default_catalog = config.path("catalog_dir") / "all_beauty_v2"
        catalog_path = default_catalog if default_catalog.is_dir() else None
    products_path = (catalog_path / "products.jsonl") if catalog_path else None
    stats = build_interaction_history(
        args.reviews,
        db_path,
        catalog_products_path=products_path,
        limit=args.limit,
    )
    print(json.dumps({"db_path": str(db_path), "stats": stats.as_dict()}, indent=2, default=str))
    return 0


def _cmd_history_stats(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    db_path = _history_db_path(config, args.db)
    with InteractionStore(db_path) as store:
        base = store.stats(cutoff_ts=args.cutoff_ts)
        lengths = [
            len(events)
            for _, events in store.iter_user_sequences(
                min_events=args.min_events, cutoff_ts=args.cutoff_ts
            )
        ]
    payload = {
        "db_path": str(db_path),
        "event_stats": base,
        "min_events": args.min_events,
        "cutoff_ts": args.cutoff_ts,
        "sequence_summary": sequence_length_summary(lengths),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _cmd_index_build(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    encoder = _make_encoder(config, device=args.device, batch_size=args.batch_size)
    catalog_manifest = json.loads((args.catalog / "manifest.json").read_text(encoding="utf-8"))
    default_version = f"{config.e5.model_id.split('/')[-1]}_{catalog_manifest['catalog_version']}"
    bundle_dir = args.out or (config.path("faiss_dir") / (args.version or default_version))
    manifest = build_bundle(
        catalog_dir=args.catalog,
        output_dir=bundle_dir,
        encoder=encoder,
        index_type=config.retrieval.e5.index.type,
        normalize_vectors=config.retrieval.e5.index.normalize_vectors,
    )
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def _cmd_index_info(args: argparse.Namespace) -> int:
    manifest = verify_bundle(args.bundle)
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def parse_constraint_text(text: str) -> HardConstraint:
    """Parse ``field:operator:value[:unit]`` into a hard constraint."""
    parts = text.split(":")
    if len(parts) < 3:
        raise ValueError(f"invalid constraint '{text}'; expected field:operator:value[:unit]")
    field, operator, raw_value = parts[0], parts[1], parts[2]
    unit = parts[3] if len(parts) > 3 else None
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return HardConstraint(
        field=field,
        operator=ConstraintOperator(operator),
        value=value,
        unit=unit,
    )


def _tiger_git_commit(project_root: Path) -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() or None
    except OSError:
        return None


def _cmd_tiger_rqvae(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    from recommender.retrieval.tiger.config import load_tiger_config
    from recommender.retrieval.tiger.rqvae import (
        encode_vectors,
        load_bundle_vectors,
        train_rqvae,
        write_rqvae_manifest,
    )
    from recommender.retrieval.tiger.semantic_ids import SemanticIdTable

    tiger_config = load_tiger_config(args.config)
    if args.epochs is not None:
        tiger_config = tiger_config.model_copy(update={"epochs": args.epochs})
    if args.device is not None:
        tiger_config = tiger_config.model_copy(update={"device": args.device})
    out_dir = args.out or (config.path("semantic_ids_dir") / tiger_config.name)
    vectors, product_ids = load_bundle_vectors(args.bundle)
    model, metrics = train_rqvae(vectors, tiger_config, out_dir)
    manifest = write_rqvae_manifest(
        out_dir,
        config=tiger_config,
        bundle_dir=args.bundle,
        vectors_path=args.bundle / "embeddings.npy",
        product_ids_path=args.bundle / "product_ids.jsonl",
        metrics=metrics,
        git_commit=_tiger_git_commit(config.project_root),
    )
    codes = encode_vectors(model, vectors)
    table = SemanticIdTable.build(product_ids, codes)
    table_manifest = table.save(out_dir)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "metrics": metrics.as_dict(),
                "manifest": manifest,
                "semantic_id_manifest": table_manifest,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


def _cmd_tiger_semantic_ids(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    from recommender.retrieval.tiger.rqvae import (
        encode_vectors,
        load_bundle_vectors,
        load_rqvae,
    )
    from recommender.retrieval.tiger.semantic_ids import SemanticIdTable

    model, tiger_config = load_rqvae(args.checkpoint_dir / "checkpoint.pt")
    vectors, product_ids = load_bundle_vectors(args.bundle)
    if vectors.shape[1] != tiger_config.input_dim:
        raise SystemExit(
            f"bundle dim {vectors.shape[1]} != checkpoint dim {tiger_config.input_dim}"
        )
    codes = encode_vectors(model, vectors)
    table = SemanticIdTable.build(product_ids, codes)
    out_dir = args.out or args.checkpoint_dir
    manifest = table.save(out_dir)
    print(json.dumps({"out_dir": str(out_dir), "manifest": manifest}, indent=2, ensure_ascii=False))
    return 0


def _tiger_db_path(config: AppConfig, override: Path | None) -> Path:
    return override or config.resolve(config.memory.sqlite.path)


def _cmd_tiger_sequences(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    from recommender.interaction_history import InteractionStore
    from recommender.retrieval.tiger.semantic_ids import SemanticIdTable
    from recommender.retrieval.tiger.sequences import build_sequences, save_samples

    db_path = _tiger_db_path(config, args.db)
    out_dir = args.out or (config.path("semantic_ids_dir") / f"samples_{args.name}")
    table = SemanticIdTable.load(args.semantic_ids)
    with InteractionStore(db_path) as store:
        samples, stats = build_sequences(
            store,
            table,
            min_events=args.min_events,
            min_history=args.min_history,
            max_history_items=args.max_history_items,
        )
    manifest = save_samples(
        samples,
        out_dir,
        stats,
        extra_manifest={"semantic_ids_dir": str(args.semantic_ids), "db_path": str(db_path)},
    )
    print(
        json.dumps(
            {"out_dir": str(out_dir), "stats": stats.as_dict(), "manifest": manifest},
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


def _cmd_tiger_generator(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    from recommender.retrieval.tiger.generator import (
        load_generator_config,
        train_generator,
        write_generator_manifest,
    )
    from recommender.retrieval.tiger.sequences import load_samples

    generator_config = load_generator_config(args.config)
    if args.epochs is not None:
        generator_config = generator_config.model_copy(update={"epochs": args.epochs})
    out_dir = args.out or (config.path("semantic_ids_dir") / generator_config.name)
    train_rows = load_samples(args.samples_dir, "train")
    val_rows = load_samples(args.samples_dir, "validation")
    _model, metrics = train_generator(train_rows, val_rows, generator_config, out_dir)
    manifest = write_generator_manifest(
        out_dir,
        config=generator_config,
        samples_dir=args.samples_dir,
        semantic_ids_dir=args.semantic_ids,
        metrics=metrics,
        git_commit=_tiger_git_commit(config.project_root),
    )
    print(
        json.dumps(
            {"out_dir": str(out_dir), "metrics": metrics.as_dict(), "manifest": manifest},
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


def _cmd_tiger_evaluate(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    from recommender.interaction_history import InteractionStore
    from recommender.retrieval.tiger.generator import (
        evaluate_samples,
        load_generator,
        popularity_from_store,
    )
    from recommender.retrieval.tiger.semantic_ids import SemanticIdTable
    from recommender.retrieval.tiger.sequences import load_samples
    from recommender.utils.io import write_json

    out_dir = args.out or (args.generator_dir / "evaluation")
    checkpoint_name = "generator_best.pt" if args.checkpoint == "best" else "generator.pt"
    model, generator_config = load_generator(args.generator_dir / checkpoint_name)
    if args.beam_size is not None:
        generator_config = generator_config.model_copy(update={"beam_size": args.beam_size})
    table = SemanticIdTable.load(args.semantic_ids)
    rows = load_samples(args.samples_dir, args.split)
    db_path = _tiger_db_path(config, args.db)
    with InteractionStore(db_path) as store:
        popularity = popularity_from_store(store)
    report = evaluate_samples(model, generator_config, rows, table, popularity)
    report["split"] = args.split
    report["checkpoint"] = args.checkpoint
    write_json(out_dir / "evaluation.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


def _evaluation_paths(args: argparse.Namespace, config: AppConfig) -> tuple[Path, Path]:
    db_path = args.db or config.resolve(config.memory.sqlite.path)
    catalog = args.catalog or (config.path("catalog_dir") / "all_beauty_v2")
    return db_path, catalog


def _cmd_eval_audit(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    from recommender.evaluation.pilot import build_temporal_dataset

    db_path, catalog_path = _evaluation_paths(args, config)
    out_dir = args.out or (config.path("data_dir") / "evaluation" / "audit")
    samples, labels, stats = build_temporal_dataset(
        samples_path=args.samples,
        db_path=db_path,
        semantic_ids_dir=args.semantic_ids,
        catalog_path=catalog_path,
        out_dir=out_dir,
        dataset_version=config.evaluation.query.dataset_version,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "recovered": len(samples),
                "labels": len(labels),
                "stats": stats.as_dict(),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


def _cmd_eval_pilot(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    if not settings.has_deepseek_api_key():
        raise SystemExit("DEEPSEEK_API_KEY is required for the Query Generator/Auditor pilot")
    from recommender.evaluation.pilot import run_pilot

    db_path, catalog_path = _evaluation_paths(args, config)
    out_dir = args.out or (
        config.path("data_dir") / "evaluation" / config.evaluation.query.dataset_version
    )
    result = run_pilot(
        samples_path=args.samples,
        db_path=db_path,
        semantic_ids_dir=args.semantic_ids,
        catalog_path=catalog_path,
        out_dir=out_dir,
        limit=args.limit,
        seed=args.seed,
        settings=settings,
    )
    print(json.dumps(result["coverage"], indent=2, ensure_ascii=False, default=str))
    return 0


def _cmd_tiger_info(args: argparse.Namespace) -> int:
    from recommender.utils.io import read_json

    directory = args.dir
    payload: dict[str, object] = {"dir": str(directory)}
    for name in ("manifest.json", "metrics.json", "semantic_id_manifest.json"):
        path = directory / name
        if path.is_file():
            payload[name] = read_json(path)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


def _cmd_pool(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    encoder = _make_encoder(config, device=args.device)
    e5 = config.retrieval.e5
    retriever = load_retriever(
        args.bundle,
        encoder,
        top_k_per_query=args.per_query_k or e5.top_k_per_query,
        rrf_k=e5.rrf_k,
        max_queries=e5.max_queries,
    )
    output = retriever.search(args.query, top_k_per_query=args.per_query_k)
    outcome = fuse_paths(
        [output.fused],
        rrf_k=config.retrieval.fusion.rrf_k,
        max_candidates=config.retrieval.fusion.max_candidates,
    )
    products = {
        product.product_id: product for product in load_products(args.catalog / "products.jsonl")
    }
    constraints = [parse_constraint_text(item) for item in (args.constraint or [])]
    request_id = new_request_context().request_id
    pool = build_candidate_pool(
        request_id=request_id,
        outcome=outcome,
        constraints=constraints,
        products=products,
        rrf_k=config.retrieval.fusion.rrf_k,
    )
    payload = {
        "request_id": request_id,
        "queries": [result.query for result in output.per_query],
        "summary": pool.constraint_summary.model_dump(mode="json"),
        "constraints": [c.model_dump(mode="json") for c in constraints],
        "eligible": [
            {
                "rank": index,
                "product_id": candidate.product_id,
                "fusion_score": round(candidate.fusion_score, 6),
                "status": candidate.constraint_status.value
                if candidate.constraint_status
                else None,
                "checks": [check.model_dump(mode="json") for check in candidate.constraint_checks],
                "product": candidate.product_metadata,
            }
            for index, candidate in enumerate(pool.candidates[: args.top_k], start=1)
        ],
        "unverified_count": len(pool.unverified_candidates),
        "rejected_count": len(pool.rejected_candidates),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


def _cmd_rerank(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    encoder = _make_encoder(config, device=args.device)
    e5 = config.retrieval.e5
    retriever = load_retriever(
        args.bundle,
        encoder,
        top_k_per_query=args.per_query_k or e5.top_k_per_query,
        rrf_k=e5.rrf_k,
        max_queries=e5.max_queries,
    )
    output = retriever.search(args.query, top_k_per_query=args.per_query_k)
    outcome = fuse_paths(
        [output.fused],
        rrf_k=config.retrieval.fusion.rrf_k,
        max_candidates=config.retrieval.fusion.max_candidates,
    )
    products = {
        product.product_id: product for product in load_products(args.catalog / "products.jsonl")
    }
    constraints = [parse_constraint_text(item) for item in (args.constraint or [])]
    pool = build_candidate_pool(
        request_id=new_request_context().request_id,
        outcome=outcome,
        constraints=constraints,
        products=products,
        rrf_k=config.retrieval.fusion.rrf_k,
    )
    if not pool.candidates:
        print(
            json.dumps(
                {
                    "status": "no_eligible_candidates",
                    "summary": pool.constraint_summary.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        return 1
    texts = {
        pair.product_id: pair.rerank_text for pair in load_text_pairs(args.catalog / "texts.jsonl")
    }
    scorer = Qwen3Reranker(
        model_id=config.reranker.primary_model_id,
        device=args.device or config.reranker.device,
        max_length=config.reranker.max_length,
        batch_size=config.reranker.batch_size,
        instruction=config.reranker.instruction,
    )
    service = RerankService(scorer, texts)
    rerank_query = args.rerank_query or args.query[0]
    result = service.rerank(rerank_query, pool.candidates, top_k=args.top_k)
    metadata = {candidate.product_id: candidate.product_metadata for candidate in pool.candidates}
    print(
        json.dumps(
            {
                "request_id": pool.request_id,
                "constraint_summary": pool.constraint_summary.model_dump(mode="json"),
                "model_id": result.model_id,
                "batch_size": result.batch_size,
                "latency_ms": round(result.latency_ms, 1),
                "ranked": [
                    {
                        "rank": product.rerank_rank,
                        "product_id": product.product_id,
                        "rerank_score": product.rerank_score,
                        "fusion_rank": product.fusion_rank,
                        "product": metadata.get(product.product_id, {}),
                    }
                    for product in result.ranked
                ],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    settings = _settings(args)
    config = settings.config
    _configure_logging(config)
    encoder = _make_encoder(config, device=args.device)
    e5 = config.retrieval.e5
    retriever = load_retriever(
        args.bundle,
        encoder,
        top_k_per_query=args.per_query_k or e5.top_k_per_query,
        rrf_k=e5.rrf_k,
        max_queries=e5.max_queries,
    )
    output = retriever.search(args.query, top_k_per_query=args.per_query_k)
    hits = output.fused.hits[: args.top_k]
    catalog: dict[str, dict[str, object]] = {}
    texts: dict[str, str] = {}
    if args.catalog is not None:
        catalog = {
            product.product_id: {
                "title": product.title,
                "price": product.price,
                "main_category": product.main_category,
            }
            for product in load_products(args.catalog / "products.jsonl")
        }
        if args.with_texts:
            texts = {
                pair.product_id: pair.rerank_text
                for pair in load_text_pairs(args.catalog / "texts.jsonl")
            }
    payload = {
        "queries": [result.query for result in output.per_query],
        "latency_ms": round(output.fused.latency_ms, 2),
        "fused": [
            {
                "rank": hit.rank,
                "product_id": hit.product_id,
                "similarity": round(hit.raw_score, 6) if hit.raw_score is not None else None,
                "matched_query": hit.query,
                **({"catalog": catalog.get(hit.product_id, {})} if catalog else {}),
                **({"rerank_text": texts[hit.product_id]} if hit.product_id in texts else {}),
            }
            for hit in hits
        ],
        "per_query": [
            {
                "query": result.query,
                "hits": [
                    {"rank": hit.rank, "product_id": hit.product_id, "similarity": hit.raw_score}
                    for hit in result.hits[: args.top_k]
                ],
            }
            for result in output.per_query
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    commands = {
        ("version", None): _cmd_version,
        ("config", "show"): _cmd_config_show,
        ("config", "validate"): _cmd_config_validate,
        ("catalog", "download"): _cmd_catalog_download,
        ("catalog", "build"): _cmd_catalog_build,
        ("history", "download"): _cmd_history_download,
        ("history", "build"): _cmd_history_build,
        ("history", "stats"): _cmd_history_stats,
        ("index", "build"): _cmd_index_build,
        ("index", "info"): _cmd_index_info,
        ("tiger", "rqvae"): _cmd_tiger_rqvae,
        ("tiger", "semantic-ids"): _cmd_tiger_semantic_ids,
        ("tiger", "sequences"): _cmd_tiger_sequences,
        ("tiger", "generator"): _cmd_tiger_generator,
        ("tiger", "evaluate"): _cmd_tiger_evaluate,
        ("tiger", "info"): _cmd_tiger_info,
        ("eval", "audit"): _cmd_eval_audit,
        ("eval", "pilot"): _cmd_eval_pilot,
        ("pool", None): _cmd_pool,
        ("rerank", None): _cmd_rerank,
        ("search", None): _cmd_search,
    }
    handler = commands.get((args.command, getattr(args, f"{args.command}_command", None)))
    if handler is None:
        parser.error(f"unsupported command: {args.command}")
        return 2
    try:
        return handler(args)
    except Exception as exc:  # CLI boundary: concise error, no traceback of secrets
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
