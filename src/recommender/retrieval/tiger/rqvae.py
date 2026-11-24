"""Residual-quantized VAE (RQ-VAE) for product Semantic IDs (Phase 4, A1 pilot).

The model is a small MLP encoder/decoder with ``num_levels`` residual codebooks. It is
trained on the project's own E5 product vectors (768-d) — no external weights are used
(decision Q1: B→A, retrain on own data).

Importing this module requires torch; CLI commands import it lazily.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from recommender.observability.logging import get_logger
from recommender.retrieval.tiger.config import RQVAEPilotConfig
from recommender.utils.errors import ArtifactError
from recommender.utils.io import ensure_dir, file_sha256, read_json, utc_now_iso, write_json

_LOGGER = get_logger(__name__)


@dataclass
class RQVAEOutput:
    recon: torch.Tensor
    codes: torch.Tensor
    recon_loss: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor


@dataclass
class RQVAEMetrics:
    epochs: int
    train_recon_mse: list[float] = field(default_factory=list)
    val_recon_mse: list[float] = field(default_factory=list)
    code_utilization: dict[str, float] = field(default_factory=dict)
    dead_codes: dict[str, int] = field(default_factory=dict)
    perplexity: dict[str, float] = field(default_factory=dict)
    train_seconds: float = 0.0
    peak_rss_mb: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "train_recon_mse": self.train_recon_mse,
            "val_recon_mse": self.val_recon_mse,
            "code_utilization": self.code_utilization,
            "dead_codes": self.dead_codes,
            "perplexity": self.perplexity,
            "train_seconds": round(self.train_seconds, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
        }


class RQVAE(nn.Module):
    """Multi-level residual VQ-VAE with straight-through gradients at each level."""

    def __init__(self, config: RQVAEPilotConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )
        self.codebooks = nn.ModuleList(
            [
                nn.Embedding(config.codebook_size, config.latent_dim)
                for _ in range(config.num_levels)
            ]
        )
        bound = 1.0 / config.codebook_size
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -bound, bound)

    def quantize(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized = torch.zeros_like(z)
        commitment = torch.zeros((), device=z.device)
        codebook_loss = torch.zeros((), device=z.device)
        codes: list[torch.Tensor] = []
        for codebook in self.codebooks:
            embeddings = codebook.weight
            distances = (
                residual.pow(2).sum(1, keepdim=True)
                - 2 * residual @ embeddings.t()
                + embeddings.pow(2).sum(1).unsqueeze(0)
            )
            indices = distances.argmin(dim=1)
            vectors = codebook(indices)
            commitment = commitment + functional.mse_loss(residual, vectors.detach())
            codebook_loss = codebook_loss + functional.mse_loss(residual.detach(), vectors)
            quantized = quantized + residual + (vectors - residual).detach()
            residual = residual - vectors.detach()
            codes.append(indices)
        return quantized, torch.stack(codes, dim=1), commitment, codebook_loss

    def forward(self, inputs: torch.Tensor) -> RQVAEOutput:
        latent = self.encoder(inputs)
        quantized, codes, commitment, codebook_loss = self.quantize(latent)
        recon = self.decoder(quantized)
        return RQVAEOutput(
            recon=recon,
            codes=codes,
            recon_loss=functional.mse_loss(recon, inputs),
            commitment_loss=commitment,
            codebook_loss=codebook_loss,
        )

    @torch.no_grad()
    def encode_codes(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(inputs)
        _, codes, _, _ = self.quantize(latent)
        return codes

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = torch.zeros(
            (codes.shape[0], self.config.latent_dim), device=codes.device, dtype=torch.float32
        )
        for level, codebook in enumerate(self.codebooks):
            quantized = quantized + codebook(codes[:, level])
        return self.decoder(quantized)


def _to_device(config: RQVAEPilotConfig) -> torch.device:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ArtifactError("RQ-VAE config requests cuda but torch.cuda is unavailable")
    return torch.device(config.device)


def _peak_rss_mb() -> float:
    try:
        import importlib

        resource = importlib.import_module("resource")
        # Linux ru_maxrss is kilobytes.
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:  # pragma: no cover - non-Linux
        return 0.0


def _code_usage(
    codes: np.ndarray, codebook_size: int
) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
    utilization: dict[str, float] = {}
    dead: dict[str, int] = {}
    perplexity: dict[str, float] = {}
    for level in range(codes.shape[1]):
        counts = np.bincount(codes[:, level], minlength=codebook_size).astype(np.float64)
        probabilities = counts[counts > 0] / counts.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        utilization[f"level_{level + 1}"] = round(float((counts > 0).mean()), 4)
        dead[f"level_{level + 1}"] = int(codebook_size - int((counts > 0).sum()))
        perplexity[f"level_{level + 1}"] = round(float(np.exp(entropy)), 2)
    return utilization, dead, perplexity


def _encode_in_batches(
    model: RQVAE, vectors: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(vectors), batch_size):
        batch = torch.from_numpy(vectors[start : start + batch_size]).to(device)
        outputs.append(model.encode_codes(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _numpy_kmeans(
    points: np.ndarray, clusters: int, *, iterations: int = 25, seed: int = 42
) -> np.ndarray:
    """Small deterministic k-means used for codebook initialization."""
    rng = np.random.default_rng(seed)
    count = len(points)
    if count < clusters:
        indices = rng.choice(count, size=clusters, replace=True)
    else:
        indices = rng.choice(count, size=clusters, replace=False)
    centers = points[indices].copy()
    for _ in range(iterations):
        distances = (
            (points**2).sum(1, keepdims=True)
            - 2 * points @ centers.T
            + (centers**2).sum(1)[None, :]
        )
        assignment = distances.argmin(1)
        new_centers = centers.copy()
        for cluster in range(clusters):
            members = points[assignment == cluster]
            if len(members):
                new_centers[cluster] = members.mean(0)
        shift = float(np.abs(new_centers - centers).max())
        centers = new_centers
        if shift < 1e-6:
            break
    return centers.astype(np.float32)


def _initialize_codebooks(
    model: RQVAE,
    vectors: np.ndarray,
    train_index: np.ndarray,
    config: RQVAEPilotConfig,
    device: torch.device,
) -> None:
    """Warm up the autoencoder, then k-means-initialize each residual level."""
    warmup = max(config.warmup_epochs, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed + 1000)
    model.train()
    for _ in range(warmup):
        order = rng.permutation(train_index)
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            batch = torch.from_numpy(vectors[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model.decoder(model.encoder(batch))
            loss = functional.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
    model.eval()
    sample_size = min(config.kmeans_sample_size, len(train_index))
    sample_index = rng.choice(train_index, size=sample_size, replace=False)
    with torch.no_grad():
        samples = [
            model.encoder(
                torch.from_numpy(vectors[sample_index[start : start + config.batch_size]]).to(
                    device
                )
            )
            .cpu()
            .numpy()
            for start in range(0, sample_size, config.batch_size)
        ]
    latents = np.concatenate(samples, axis=0).astype(np.float32)
    residual = latents
    for level, codebook in enumerate(model.codebooks):
        centers = _numpy_kmeans(residual, config.codebook_size, seed=config.seed + level)
        codebook.weight.data = torch.from_numpy(centers).to(device)
        distances = (
            (residual**2).sum(1, keepdims=True)
            - 2 * residual @ centers.T
            + (centers**2).sum(1)[None, :]
        )
        assignment = distances.argmin(1)
        residual = residual - centers[assignment]
    _LOGGER.info(
        "codebooks initialized with k-means",
        extra={"sample_size": sample_size, "warmup_epochs": warmup},
    )


def train_rqvae(
    vectors: np.ndarray,
    config: RQVAEPilotConfig,
    output_dir: Path | str,
    *,
    log_every_epochs: int | None = None,
) -> tuple[RQVAE, RQVAEMetrics]:
    """Train the RQ-VAE on product vectors and persist the checkpoint + metrics."""
    if vectors.ndim != 2 or vectors.shape[1] != config.input_dim:
        raise ArtifactError(
            f"expected vectors of shape [N, {config.input_dim}], got {vectors.shape}"
        )
    if not np.isfinite(vectors).all():
        raise ArtifactError("product vectors contain NaN/Inf")
    directory = ensure_dir(output_dir)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = _to_device(config)

    count = len(vectors)
    permutation = rng.permutation(count)
    val_count = round(count * config.val_ratio)
    val_index = np.sort(permutation[:val_count])
    train_index = np.sort(permutation[val_count:])
    if len(train_index) == 0 or len(val_index) == 0:
        raise ArtifactError("split produced an empty train/validation set")

    model = RQVAE(config).to(device)
    if config.kmeans_init:
        _initialize_codebooks(model, vectors, train_index, config, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    log_interval = log_every_epochs or config.log_every_epochs
    metrics = RQVAEMetrics(epochs=config.epochs)
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_rng = np.random.default_rng(config.seed + epoch)
        shuffled = epoch_rng.permutation(train_index)
        for start in range(0, len(shuffled), config.batch_size):
            index = shuffled[start : start + config.batch_size]
            batch = torch.from_numpy(vectors[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = (
                output.recon_loss
                + config.commitment_weight * output.commitment_loss
                + config.codebook_weight * output.codebook_loss
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(output.recon_loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_mse = []
            for start in range(0, len(val_index), config.batch_size):
                index = val_index[start : start + config.batch_size]
                batch = torch.from_numpy(vectors[index]).to(device)
                val_mse.append(float(model(batch).recon_loss.cpu()))
        metrics.train_recon_mse.append(round(float(np.mean(epoch_losses)), 6))
        metrics.val_recon_mse.append(round(float(np.mean(val_mse)), 6))
        if epoch % log_interval == 0:
            _LOGGER.info(
                "rqvae epoch",
                extra={
                    "epoch": epoch,
                    "train_recon_mse": metrics.train_recon_mse[-1],
                    "val_recon_mse": metrics.val_recon_mse[-1],
                },
            )

    metrics.train_seconds = time.perf_counter() - started
    codes = _encode_in_batches(model, vectors, config.batch_size, device)
    utilization, dead, perplexity = _code_usage(codes, config.codebook_size)
    metrics.code_utilization = utilization
    metrics.dead_codes = dead
    metrics.perplexity = perplexity
    metrics.peak_rss_mb = _peak_rss_mb()

    checkpoint_path = directory / "checkpoint.pt"
    torch.save(
        {
            "config": config.model_dump(),
            "state_dict": model.state_dict(),
            "epochs": config.epochs,
        },
        checkpoint_path,
    )
    np.save(directory / "codes.npy", codes.astype(np.int32))
    write_json(directory / "metrics.json", metrics.as_dict())
    _LOGGER.info(
        "rqvae trained",
        extra={
            "checkpoint": str(checkpoint_path),
            "codes": list(codes.shape),
            "train_seconds": round(metrics.train_seconds, 1),
        },
    )
    return model, metrics


def load_rqvae(checkpoint_path: Path | str) -> tuple[RQVAE, RQVAEPilotConfig]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ArtifactError(f"RQ-VAE checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = RQVAEPilotConfig(**payload["config"])
    model = RQVAE(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config


def encode_vectors(model: RQVAE, vectors: np.ndarray, *, batch_size: int = 2048) -> np.ndarray:
    device = next(model.parameters()).device
    return _encode_in_batches(model, vectors, batch_size, device)


def write_rqvae_manifest(
    output_dir: Path | str,
    *,
    config: RQVAEPilotConfig,
    bundle_dir: Path | str,
    vectors_path: Path | str,
    product_ids_path: Path | str,
    metrics: RQVAEMetrics,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Persist a manifest tying the checkpoint to data and code versions."""
    directory = Path(output_dir)
    checkpoint = directory / "checkpoint.pt"
    bundle_manifest_path = Path(bundle_dir) / "manifest.json"
    manifest = {
        "schema_version": "1.0",
        "created_at": utc_now_iso(),
        "config": config.model_dump(),
        "data": {
            "vectors_path": str(vectors_path),
            "vectors_sha256": file_sha256(vectors_path),
            "product_ids_path": str(product_ids_path),
            "product_ids_sha256": file_sha256(product_ids_path),
            "bundle_manifest_sha256": file_sha256(bundle_manifest_path)
            if bundle_manifest_path.is_file()
            else None,
        },
        "checkpoint_sha256": file_sha256(checkpoint),
        "metrics": metrics.as_dict(),
        "git_commit": git_commit,
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def read_rqvae_metrics(output_dir: Path | str) -> dict[str, Any]:
    return read_json(Path(output_dir) / "metrics.json")


def load_codes(output_dir: Path | str) -> np.ndarray:
    path = Path(output_dir) / "codes.npy"
    if not path.is_file():
        raise ArtifactError(f"codes.npy not found in {output_dir}")
    return np.load(path)


def load_bundle_vectors(bundle_dir: Path | str) -> tuple[np.ndarray, list[str]]:
    """Load product ids and E5 vectors from a retrieval bundle."""
    directory = Path(bundle_dir)
    vectors_path = directory / "embeddings.npy"
    ids_path = directory / "product_ids.jsonl"
    if not vectors_path.is_file() or not ids_path.is_file():
        raise ArtifactError(f"bundle {directory} is missing embeddings.npy/product_ids.jsonl")
    vectors = np.load(vectors_path).astype(np.float32)
    product_ids = [
        json.loads(line)["product_id"]
        for line in ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(product_ids) != len(vectors):
        raise ArtifactError(
            f"product_ids ({len(product_ids)}) != vectors ({len(vectors)}) in {directory}"
        )
    return vectors, product_ids
