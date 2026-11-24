"""Configuration loading: precedence, isolation, validation and paths."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from recommender.config import (
    PROJECT_ROOT,
    load_config,
    load_secrets,
    load_settings,
)
from recommender.utils.errors import ConfigError

pytestmark = pytest.mark.unit


def test_real_configs_load() -> None:
    config = load_config(environ={})
    assert config.e5.model_id == "intfloat/e5-base-v2"
    assert config.e5.embedding_dim == 768
    assert config.retrieval.e5.top_k_per_query == 50
    assert config.retrieval.budgets.standard.e5_candidates == 100
    assert config.retrieval.budgets.expanded.e5_candidates >= 100
    assert config.retrieval.fusion.rrf_k == 60
    assert config.controller.quality.calibrated is False
    assert config.reranker.primary_model_id == "Qwen/Qwen3-Reranker-0.6B"
    assert config.memory.enabled is False


def test_env_override_wins(config_dir: Path) -> None:
    config = load_config(
        config_dir,
        environ={"AGR__RETRIEVAL__E5__RRF_K": "75", "AGR__RETRIEVAL__E5__MAX_QUERIES": "3"},
    )
    assert config.retrieval.e5.rrf_k == 75
    assert config.retrieval.e5.max_queries == 3


def test_cross_field_invariant_is_enforced_after_override(config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="e5_candidates must be >="):
        load_config(
            config_dir,
            environ={"AGR__RETRIEVAL__E5__TOP_K_PER_QUERY": "123"},
        )


def test_env_override_coerces_scalars(config_dir: Path) -> None:
    config = load_config(
        config_dir,
        environ={"AGR__CONTROLLER__QUALITY__CALIBRATED": "true"},
    )
    assert config.controller.quality.calibrated is True


def test_unknown_env_override_is_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(config_dir, environ={"AGR__RETRIEVAL__E5__NOT_A_FIELD": "1"})


def test_duplicate_keys_across_files_are_rejected(config_dir: Path) -> None:
    base = config_dir / "base.yaml"
    base.write_text(
        base.read_text(encoding="utf-8") + "\ne5:\n  model_id: duplicate\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_config(config_dir, environ={})


def test_missing_file_is_rejected(config_dir: Path) -> None:
    (config_dir / "controller.yaml").unlink()
    with pytest.raises(ConfigError):
        load_config(config_dir, environ={})


def test_expanded_budget_below_standard_is_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(
            config_dir,
            environ={"AGR__RETRIEVAL__BUDGETS__EXPANDED__TIGER_CANDIDATES": "10"},
        )


def test_malformed_override_key_is_rejected(config_dir: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(config_dir, environ={"AGR__RETRIEVAL__E5__": "1"})


def test_secrets_come_from_env_only(
    missing_env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    secrets = load_secrets(env_file=missing_env_file)
    assert secrets.deepseek_api_key is None
    with pytest.raises(ConfigError):
        load_settings(env_file=missing_env_file).require_deepseek_api_key()


def test_secrets_override_and_blank_normalisation(
    missing_env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    secrets = load_secrets(env_file=missing_env_file, deepseek_api_key="sk-unit-test")
    assert secrets.deepseek_api_key is not None
    assert secrets.deepseek_api_key.get_secret_value() == "sk-unit-test"
    blank = load_secrets(env_file=missing_env_file, deepseek_api_key="   ")
    assert blank.deepseek_api_key is None


def test_paths_resolve_against_project_root() -> None:
    config = load_config(environ={})
    assert config.path("faiss_dir") == PROJECT_ROOT / "artifacts" / "faiss"
    with pytest.raises(ConfigError):
        config.path("not_a_configured_path")


def test_ensure_directories(tmp_path: Path) -> None:
    config = load_config(environ={}).model_copy(update={"project_root": tmp_path})
    config.ensure_directories()
    for name in ("data_dir", "raw_dir", "processed_dir", "faiss_dir", "runs_dir"):
        assert config.path(name).is_dir()


def test_config_copy_is_isolated(tmp_path: Path, config_dir: Path) -> None:
    copied = tmp_path / "copy"
    shutil.copytree(config_dir, copied)
    config = load_config(copied, environ={})
    assert config.schema_version == "1.0"


def test_local_overlay_overrides_tracked_config(config_dir: Path) -> None:
    (config_dir / "local.yaml").write_text(
        "paths:\n  raw_dir: /tmp/agr/raw\nretrieval:\n  e5:\n    top_k_per_query: 25\n",
        encoding="utf-8",
    )
    config = load_config(config_dir, environ={})
    assert config.paths.raw_dir == Path("/tmp/agr/raw")
    assert config.retrieval.e5.top_k_per_query == 25
