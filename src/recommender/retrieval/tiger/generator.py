"""Autoregressive generator over Semantic ID tokens (Phase 4 pilot, decoder-only).

Training uses the project's own time-split user sequences. The model is a small causal
Transformer trained from scratch on CPU (decision Q11); no external weights are used.
Generation produces multi-level Semantic IDs which decode to catalog products through
the Semantic ID table (collisions return multiple products, ranked by interaction
popularity).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn import functional

from recommender.observability.logging import get_logger
from recommender.retrieval.tiger.semantic_ids import SemanticIdTable, format_semantic_id
from recommender.utils.errors import ArtifactError
from recommender.utils.io import ensure_dir, file_sha256, read_json, utc_now_iso, write_json

_LOGGER = get_logger(__name__)


class TigerGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "generator_pilot_a1"
    num_levels: int = 3
    codebook_size: int = 256
    embedding_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.1
    max_seq_len: int = 64
    max_history_items: int = 10
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    seed: int = 42
    device: str = "cpu"
    log_every_epochs: int = 5
    beam_size: int = 10
    top_k_products: int = 50


def load_generator_config(path: Path | str) -> TigerGeneratorConfig:
    import yaml

    source = Path(path)
    if not source.is_file():
        raise ArtifactError(f"generator config not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    return TigerGeneratorConfig(**payload)


class TigerGenerator(nn.Module):
    """Causal Transformer over flattened Semantic ID tokens."""

    def __init__(self, config: TigerGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.num_levels * config.codebook_size
        self.bos_id = self.vocab_size
        self.pad_id = self.vocab_size + 1
        self.token_embedding = nn.Embedding(self.pad_id + 1, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, self.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.encoder(hidden, mask=mask)
        return self.lm_head(self.norm(hidden))


@dataclass
class GeneratorMetrics:
    epochs: int
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_seconds: float = 0.0
    peak_rss_mb: float = 0.0
    num_parameters: int = 0
    best_epoch: int = 0
    best_val_loss: float = float("inf")

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_seconds": round(self.train_seconds, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "num_parameters": self.num_parameters,
            "best_epoch": self.best_epoch,
            "best_val_loss": round(self.best_val_loss, 6),
        }


def item_to_tokens(semantic_id: Sequence[int], codebook_size: int) -> list[int]:
    return [level * codebook_size + int(code) for level, code in enumerate(semantic_id)]


def tokens_to_item(tokens: Sequence[int], codebook_size: int, num_levels: int) -> tuple[int, ...]:
    if len(tokens) != num_levels:
        raise ValueError(f"expected {num_levels} tokens, got {len(tokens)}")
    codes: list[int] = []
    for level, token in enumerate(tokens):
        code = token - level * codebook_size
        if code < 0 or code >= codebook_size:
            raise ValueError(f"token {token} is not valid at level {level}")
        codes.append(code)
    return tuple(codes)


def sample_to_tokens(
    row: dict[str, Any], config: TigerGeneratorConfig
) -> tuple[list[int], list[int]]:
    """Return ``(prompt_tokens, target_tokens)`` for one sample row."""
    history = row["history_ids"][-config.max_history_items :]
    prompt = [t for item in history for t in item_to_tokens(item, config.codebook_size)]
    target = item_to_tokens(row["target_id"], config.codebook_size)
    return prompt, target


def _collate(
    rows: Sequence[dict[str, Any]], config: TigerGeneratorConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences: list[list[int]] = []
    for row in rows:
        prompt, target = sample_to_tokens(row, config)
        token_ids = prompt + target
        if len(token_ids) > config.max_seq_len:
            token_ids = token_ids[-config.max_seq_len :]
        sequences.append(token_ids)
    max_len = max(len(sequence) for sequence in sequences)
    pad_id = config.num_levels * config.codebook_size + 1
    inputs = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        inputs[index, :length] = torch.tensor(sequence, dtype=torch.long)
        labels[index, :length] = torch.tensor(sequence, dtype=torch.long)
        attention[index, :length] = True
    return inputs, labels, attention


def _loss(
    model: TigerGenerator, inputs: torch.Tensor, labels: torch.Tensor, attention: torch.Tensor
) -> torch.Tensor:
    logits = model(inputs)
    # Next-token prediction: shift inputs/labels.
    logits = logits[:, :-1, :].reshape(-1, model.vocab_size)
    target = labels[:, 1:].reshape(-1)
    mask = attention[:, 1:].reshape(-1)
    return functional.cross_entropy(logits[mask], target[mask])


def train_generator(
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    config: TigerGeneratorConfig,
    output_dir: Path | str,
) -> tuple[TigerGenerator, GeneratorMetrics]:
    if not train_rows:
        raise ArtifactError("no training samples")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    model = TigerGenerator(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    directory = ensure_dir(output_dir)
    metrics = GeneratorMetrics(
        epochs=config.epochs,
        num_parameters=sum(parameter.numel() for parameter in model.parameters()),
    )
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(len(train_rows))
        losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            batch_rows = [train_rows[index] for index in order[start : start + config.batch_size]]
            inputs, labels, attention = _collate(batch_rows, config)
            inputs, labels, attention = inputs.to(device), labels.to(device), attention.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, inputs, labels, attention)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics.train_loss.append(round(float(np.mean(losses)), 6))
        model.eval()
        with torch.no_grad():
            val_losses = []
            for start in range(0, len(val_rows), config.batch_size):
                val_batch = list(val_rows[start : start + config.batch_size])
                inputs, labels, attention = _collate(val_batch, config)
                val_losses.append(
                    float(
                        _loss(
                            model,
                            inputs.to(device),
                            labels.to(device),
                            attention.to(device),
                        ).cpu()
                    )
                )
            current_val = float(np.mean(val_losses)) if val_losses else math.inf
            metrics.val_loss.append(round(current_val, 6))
            if current_val < metrics.best_val_loss:
                metrics.best_val_loss = current_val
                metrics.best_epoch = epoch
                torch.save(
                    {"config": config.model_dump(), "state_dict": model.state_dict()},
                    directory / "generator_best.pt",
                )
        if epoch % config.log_every_epochs == 0:
            _LOGGER.info(
                "generator epoch",
                extra={
                    "epoch": epoch,
                    "train_loss": metrics.train_loss[-1],
                    "val_loss": metrics.val_loss[-1],
                },
            )
    metrics.train_seconds = time.perf_counter() - started
    try:
        import importlib

        resource = importlib.import_module("resource")
        metrics.peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:  # pragma: no cover
        metrics.peak_rss_mb = 0.0
    checkpoint_path = directory / "generator.pt"
    torch.save(
        {"config": config.model_dump(), "state_dict": model.state_dict()},
        checkpoint_path,
    )
    write_json(directory / "generator_metrics.json", metrics.as_dict())
    _LOGGER.info(
        "generator trained",
        extra={
            "checkpoint": str(checkpoint_path),
            "params": metrics.num_parameters,
            "train_seconds": round(metrics.train_seconds, 1),
        },
    )
    return model, metrics


def load_generator(checkpoint_path: Path | str) -> tuple[TigerGenerator, TigerGeneratorConfig]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ArtifactError(f"generator checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = TigerGeneratorConfig(**payload["config"])
    model = TigerGenerator(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def generate_ids(
    model: TigerGenerator,
    config: TigerGeneratorConfig,
    history_ids: Sequence[Sequence[int]],
    *,
    beam_size: int | None = None,
) -> list[tuple[tuple[int, ...], float]]:
    """Beam-search the next item's multi-level Semantic ID; returns (id, mean logprob)."""
    beam = beam_size or config.beam_size
    tokens = [
        token
        for item in list(history_ids)[-config.max_history_items :]
        for token in item_to_tokens(item, config.codebook_size)
    ][: config.max_seq_len - config.num_levels]
    device = next(model.parameters()).device
    model.eval()
    prefixes: list[tuple[list[int], float]] = [(tokens, 0.0)]
    for _ in range(config.num_levels):
        candidates: list[tuple[list[int], float]] = []
        for prefix, score in prefixes:
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(input_ids)[0, -1, : config.num_levels * config.codebook_size]
            log_probs = functional.log_softmax(logits, dim=-1)
            top_values, top_indices = torch.topk(log_probs, k=beam)
            for value, token in zip(top_values.tolist(), top_indices.tolist(), strict=True):
                candidates.append(([*prefix, int(token)], score + float(value)))
        candidates.sort(key=lambda item: item[1], reverse=True)
        prefixes = candidates[:beam]
    results: list[tuple[tuple[int, ...], float]] = []
    seen: set[tuple[int, ...]] = set()
    for prefix, score in prefixes:
        try:
            semantic_id = tokens_to_item(
                prefix[-config.num_levels :], config.codebook_size, config.num_levels
            )
        except ValueError:
            continue
        if semantic_id in seen:
            continue
        seen.add(semantic_id)
        results.append((semantic_id, score / config.num_levels))
    return results


def rank_products_for_ids(
    decoded: Sequence[tuple[tuple[int, ...], float]],
    table: SemanticIdTable,
    popularity: dict[str, int],
    *,
    top_k: int,
) -> list[tuple[str, float]]:
    scored: dict[str, float] = {}
    for semantic_id, logprob in decoded:
        products = table.products_for(semantic_id)
        for product_id in products:
            score = float(logprob)
            if product_id not in scored or score > scored[product_id]:
                scored[product_id] = score
    ordered = sorted(
        scored.items(),
        key=lambda item: (-item[1], -popularity.get(item[0], 0), item[0]),
    )
    return ordered[:top_k]


def evaluate_samples(
    model: TigerGenerator,
    config: TigerGeneratorConfig,
    rows: Sequence[dict[str, Any]],
    table: SemanticIdTable,
    popularity: dict[str, int],
    *,
    k_values: Sequence[int] = (10, 50),
) -> dict[str, Any]:
    """Next-item ranking metrics on time-split samples (Recall/NDCG/MRR)."""
    if not rows:
        return {"samples": 0}
    started = time.perf_counter()
    metrics = {f"recall@{k}": 0.0 for k in k_values}
    ndcg_hits = 0.0
    reciprocal_ranks: list[float] = []
    valid_ids = 0
    generated_total = 0
    for row in rows:
        decoded = generate_ids(model, config, row["history_ids"])
        generated_total += len(decoded)
        valid_ids += sum(1 for semantic_id, _ in decoded if table.products_for(semantic_id))
        ranked = rank_products_for_ids(decoded, table, popularity, top_k=max(k_values))
        ranked_ids = [product_id for product_id, _ in ranked]
        target = row["target_product_id"]
        for k in k_values:
            if target in ranked_ids[:k]:
                metrics[f"recall@{k}"] += 1.0
        if target in ranked_ids[:10]:
            ndcg_hits += 1.0 / math.log2(ranked_ids.index(target) + 2)
        rank = ranked_ids.index(target) + 1 if target in ranked_ids else None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    count = len(rows)
    report = {
        "samples": count,
        **{key: round(value / count, 4) for key, value in metrics.items()},
        "ndcg@10": round(ndcg_hits / count, 4),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "valid_id_rate": round(valid_ids / generated_total, 4) if generated_total else 0.0,
        "avg_generated_ids": round(generated_total / count, 2),
        "latency_ms_per_sample": round((time.perf_counter() - started) * 1000 / count, 1),
        "k_values": list(k_values),
    }
    _LOGGER.info("tiger evaluation finished", extra=report)
    return report


def write_generator_manifest(
    output_dir: Path | str,
    *,
    config: TigerGeneratorConfig,
    samples_dir: Path | str,
    semantic_ids_dir: Path | str,
    metrics: GeneratorMetrics,
    git_commit: str | None,
) -> dict[str, Any]:
    directory = Path(output_dir)
    manifest = {
        "schema_version": "1.0",
        "created_at": utc_now_iso(),
        "config": config.model_dump(),
        "samples_dir": str(samples_dir),
        "sequence_manifest_sha256": file_sha256(Path(samples_dir) / "sequence_manifest.json"),
        "semantic_ids_manifest_sha256": file_sha256(
            Path(semantic_ids_dir) / "semantic_id_manifest.json"
        ),
        "checkpoint_sha256": file_sha256(directory / "generator.pt"),
        "metrics": metrics.as_dict(),
        "git_commit": git_commit,
    }
    write_json(directory / "generator_manifest.json", manifest)
    return manifest


def read_generator_report(path: Path | str) -> dict[str, Any]:
    return read_json(path)


def popularity_from_store(store: object) -> dict[str, int]:
    """Product interaction counts used only to break Semantic ID collisions."""
    return store.product_interaction_counts()  # type: ignore[attr-defined]


def semantic_id_key(semantic_id: tuple[int, ...] | list[int]) -> str:
    return format_semantic_id(semantic_id)
