"""RQ-VAE pilot, semantic ID table and TIGER config (tiny synthetic data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recommender.retrieval.tiger.config import RQVAEPilotConfig, load_tiger_config
from recommender.utils.errors import ConfigError

pytestmark = pytest.mark.unit


def test_load_a1_pilot_config() -> None:
    config = load_tiger_config("configs/tiger/rqvae_pilot_a1.yaml")
    assert config.num_levels == 3
    assert config.codebook_size == 256
    assert config.input_dim == 768
    assert config.device == "cpu"


def test_load_a2_pilot_config() -> None:
    config = load_tiger_config("configs/tiger/rqvae_pilot_a2.yaml")
    assert config.num_levels == 4
    assert config.codebook_size == 1024


def test_config_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nbogus: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_tiger_config(path)


def _tiny_config(**overrides: object) -> RQVAEPilotConfig:
    payload = {
        "name": "tiny",
        "input_dim": 16,
        "hidden_dim": 8,
        "latent_dim": 4,
        "num_levels": 2,
        "codebook_size": 8,
        "epochs": 2,
        "batch_size": 16,
        "learning_rate": 0.01,
        "val_ratio": 0.2,
        "seed": 7,
        "device": "cpu",
        "log_every_epochs": 1,
    }
    payload.update(overrides)
    return RQVAEPilotConfig(**payload)


def test_rqvae_trains_encodes_and_reloads(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from recommender.retrieval.tiger.rqvae import (
        encode_vectors,
        load_codes,
        load_rqvae,
        train_rqvae,
    )

    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(64, 16)).astype(np.float32)
    config = _tiny_config()
    _model, metrics = train_rqvae(vectors, config, tmp_path)
    assert len(metrics.train_recon_mse) == 2
    assert len(metrics.val_recon_mse) == 2
    assert all(value >= 0 for value in metrics.train_recon_mse)
    assert set(metrics.code_utilization) == {"level_1", "level_2"}

    codes = load_codes(tmp_path)
    assert codes.shape == (64, 2)
    assert codes.min() >= 0 and codes.max() < 8

    reloaded, loaded_config = load_rqvae(tmp_path / "checkpoint.pt")
    assert loaded_config.num_levels == 2
    np.testing.assert_array_equal(encode_vectors(reloaded, vectors), codes)
    decoded = reloaded.decode_codes(__import__("torch").from_numpy(codes))
    assert tuple(decoded.shape) == (64, 16)


def test_collision_stats_and_table_roundtrip(tmp_path: Path) -> None:
    from recommender.retrieval.tiger.semantic_ids import (
        SemanticIdTable,
        collision_stats,
        format_semantic_id,
        parse_semantic_id,
    )

    codes = np.asarray([[0, 0], [0, 0], [1, 2]], dtype=np.int32)
    stats = collision_stats(codes)
    assert stats["num_products"] == 3
    assert stats["unique_ids"] == 2
    assert stats["collision_ids"] == 1
    assert stats["collision_products"] == 2
    assert stats["max_products_per_id"] == 2
    assert format_semantic_id([1, 2]) == "1,2"
    assert parse_semantic_id("1,2") == (1, 2)

    table = SemanticIdTable.build(["p1", "p2", "p3"], codes)
    assert table.products_for([0, 0]) == ["p1", "p2"]
    assert table.semantic_id_of("p3") == (1, 2)
    manifest = table.save(tmp_path)
    assert manifest["num_unique_ids"] == 2
    reloaded = SemanticIdTable.load(tmp_path)
    assert reloaded.product_ids == ["p1", "p2", "p3"]
    np.testing.assert_array_equal(reloaded.codes, codes)
    assert reloaded.products_for("0,0") == ["p1", "p2"]


def test_tiger_cli_parser_accepts_commands() -> None:
    from recommender.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["tiger", "rqvae", "--bundle", "b", "--config", "c.yaml"])
    assert args.tiger_command == "rqvae"
    args = parser.parse_args(["tiger", "info", "--dir", "d"])
    assert args.tiger_command == "info"
