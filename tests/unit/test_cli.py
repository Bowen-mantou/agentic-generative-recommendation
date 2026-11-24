"""CLI smoke tests for the Phase 0 entry points."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recommender.cli import main

pytestmark = pytest.mark.unit


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_config_validate(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--config-dir", str(config_dir), "config", "validate"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "config valid" in captured.out


def test_config_show_is_json(config_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--config-dir", str(config_dir), "config", "show"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["e5"]["model_id"] == "intfloat/e5-base-v2"
    assert payload["retrieval"]["e5"]["top_k_per_query"] == 50
    assert "deepseek_api_key" not in json.dumps(payload)


def test_unknown_command_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["not-a-command"])
