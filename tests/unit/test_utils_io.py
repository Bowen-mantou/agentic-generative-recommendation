"""Filesystem utilities: atomic JSON/JSONL writes, hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from recommender.utils.io import (
    file_sha256,
    read_json,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)

pytestmark = pytest.mark.unit


def test_json_roundtrip_is_atomic(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "manifest.json"
    write_json(target, {"b": 2, "a": "中文"})
    assert read_json(target) == {"b": 2, "a": "中文"}
    assert [p.name for p in target.parent.iterdir()] == ["manifest.json"]


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "catalog.jsonl"
    rows = [{"product_id": "p1", "title": "面霜"}, {"product_id": "p2", "title": "Cream"}]
    write_jsonl(target, rows)
    assert list(read_jsonl(target)) == rows


def test_file_sha256_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_stable_hash_is_deterministic_and_short() -> None:
    first = stable_hash("lightweight moisturizer")
    assert first == stable_hash("lightweight moisturizer")
    assert len(first) == 16
    assert first != stable_hash("fragrance free")
