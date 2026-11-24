#!/usr/bin/env python3
"""Package the project and upload it to the MTT S4000 host over SFTP.

The archive excludes secrets, virtualenvs, datasets, artifacts and run logs. Remote
data/artifacts directories are never deleted: extraction only overwrites files that are
part of the archive.

Usage (from the project root):
    python scripts/sync_to_server.py
    python scripts/sync_to_server.py --dest ~/Agentic-Generative-Recommendation
Environment: S4000_HOST, S4000_USER, and the required S4000_PASSWORD.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import tarfile
import tempfile
from pathlib import Path

import paramiko

DEFAULT_HOST = "192.0.2.10"
DEFAULT_USER = "agr"
# No credentials in the repository: export S4000_PASSWORD (see .env.example).

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "data",
    "artifacts",
    "runs",
    "logs",
    ".idea",
    ".vscode",
    "node_modules",
}
EXCLUDED_FILES = ("configs/local.yaml", ".env", "*.pyc", "*.pyo", "*.egg-info")


def is_excluded(relative: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return True
    return any(fnmatch.fnmatch(relative.name, pattern) for pattern in EXCLUDED_FILES)


def collect_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(root)
            if not is_excluded(relative):
                files.append(relative)
    return sorted(files)


def build_archive(root: Path, files: list[Path], target: Path) -> int:
    with tarfile.open(target, "w:gz") as archive:
        for relative in files:
            archive.add(root / relative, arcname=relative.as_posix())
    return target.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="~/Agentic-Generative-Recommendation")
    parser.add_argument("--host", default=os.environ.get("S4000_HOST", DEFAULT_HOST))
    parser.add_argument("--user", default=os.environ.get("S4000_USER", DEFAULT_USER))
    parser.add_argument("--password", default=os.environ.get("S4000_PASSWORD"))
    args = parser.parse_args()
    if not args.password:
        parser.error(
            "S4000_PASSWORD is not set; export it or pass --password "
            "(credentials are never stored in the repository)"
        )

    root = Path(__file__).resolve().parents[1]
    files = collect_files(root)
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "project.tar.gz"
        size = build_archive(root, files, archive_path)
        print(f"packed {len(files)} files, {size / 1e6:.1f} MB")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            args.host,
            username=args.user,
            password=args.password,
            timeout=20,
            banner_timeout=30,
            auth_timeout=30,
            look_for_keys=False,
            allow_agent=False,
        )
        try:
            with client.open_sftp() as sftp:
                remote_tmp = "/tmp/agr_sync"
                try:
                    sftp.mkdir(remote_tmp)
                except OSError:
                    pass
                remote_archive = f"{remote_tmp}/project.tar.gz"
                print(f"uploading to {args.user}@{args.host}:{remote_archive}")
                sftp.put(str(archive_path), remote_archive)
            command = (
                f"set -e; mkdir -p {args.dest}; "
                f"tar -xzf /tmp/agr_sync/project.tar.gz -C {args.dest}; "
                f"echo extracted; ls {args.dest} | head"
            )
            _, stdout, stderr = client.exec_command(command, timeout=300)
            output = stdout.read().decode("utf-8", "replace")
            error = stderr.read().decode("utf-8", "replace")
            status = stdout.channel.recv_exit_status()
            print(output)
            if error.strip():
                print(error)
            if status != 0:
                print(f"remote extraction failed with status {status}")
                return status
        finally:
            client.close()
    print("sync complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
