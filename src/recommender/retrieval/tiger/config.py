"""TIGER configuration models (Phase 4 pilot).

These configs are deliberately separate from ``configs/models.yaml``: the pilot exposes
training hyper-parameters and must be recorded verbatim with each run (plan §8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from recommender.utils.errors import ConfigError


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RQVAEPilotConfig(_StrictModel):
    name: str = "rqvae_pilot_a1"
    input_dim: int = Field(default=768, gt=0)
    hidden_dim: int = Field(default=512, gt=0)
    latent_dim: int = Field(default=32, gt=0)
    num_levels: int = Field(default=3, ge=1, le=8)
    codebook_size: int = Field(default=256, ge=2)
    commitment_weight: float = Field(default=0.25, ge=0.0)
    codebook_weight: float = Field(default=1.0, ge=0.0)
    epochs: int = Field(default=20, gt=0)
    batch_size: int = Field(default=1024, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    val_ratio: float = Field(default=0.1, ge=0.0, lt=0.5)
    # Codebook collapse mitigation: autoencoder warm-up, then k-means initialization
    # of every residual level (standard RQ-VAE practice).
    warmup_epochs: int = Field(default=5, ge=0)
    kmeans_init: bool = True
    kmeans_sample_size: int = Field(default=20000, gt=0)
    seed: int = 42
    device: Literal["cpu", "cuda"] = "cpu"
    log_every_epochs: int = Field(default=1, gt=0)


def load_tiger_config(path: Path | str) -> RQVAEPilotConfig:
    """Load and validate one pilot YAML file."""
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"TIGER config not found: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid TIGER YAML {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"TIGER config must be a mapping: {source}")
    try:
        return RQVAEPilotConfig(**payload)
    except ValidationError as exc:
        raise ConfigError(f"invalid TIGER config {source}:\n{exc}") from exc
