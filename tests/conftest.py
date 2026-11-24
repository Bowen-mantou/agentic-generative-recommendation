"""Shared pytest fixtures.

Unit tests never call real APIs or load models; they exercise contracts and pure
logic. Integration tests that need models/indexes are marked ``integration``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from recommender.config import PROJECT_ROOT

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONFIGS_DIR = PROJECT_ROOT / "configs"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A mutable copy of the real configs/ directory."""
    destination = tmp_path / "configs"
    shutil.copytree(CONFIGS_DIR, destination)
    # Machine-local overlays must not leak into unit tests.
    (destination / "local.yaml").unlink(missing_ok=True)
    return destination


@pytest.fixture
def missing_env_file(tmp_path: Path) -> Path:
    """A path that does not exist, used to isolate tests from a developer .env."""
    return tmp_path / "does-not-exist.env"


def load_fixture_json(*relative_parts: str) -> dict[str, Any]:
    path = FIXTURES_DIR.joinpath(*relative_parts)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def plan_payload() -> dict[str, Any]:
    return load_fixture_json("plans", "initial_beauty.json")
