#!/usr/bin/env python3
"""Build the human-review document/CSV for a frozen evaluation dataset candidate.

The output contains no target labels: reviewers judge the query against the profile only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _profile_summary(profile: dict) -> str:
    parts: list[str] = []
    for key in ("category_interests", "brand_interests", "attribute_interests"):
        values = [fact["value"] for fact in profile.get(key, [])[:3]]
        if values:
            parts.append(f"{key.split('_')[0]}: " + ", ".join(values))
    negatives = [fact["value"] for fact in profile.get("negative_signals", [])[:2]]
    if negatives:
        parts.append("negative: " + ", ".join(negatives))
    return "; ".join(parts)[:300]


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--dataset-version", default="v1.0")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    queue_path = args.pilot / "human_review" / "human_audit.csv"
    profiles = {
        row["sample_id"]: row
        for row in _read_jsonl(args.pilot / "profiles" / "test_profiles.jsonl")
    }
    audits = {
        row["sample_id"]: row for row in _read_jsonl(args.pilot / "audits" / "test_audits.jsonl")
    }
    queries = {
        row["sample_id"]: row for row in _read_jsonl(args.pilot / "queries" / "test_queries.jsonl")
    }
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        queue = list(csv.DictReader(handle))

    rows: list[dict[str, str]] = []
    for item in queue:
        sample_id = item["sample_id"]
        query = queries.get(sample_id, {})
        audit = audits.get(sample_id, {})
        profile = profiles.get(sample_id, {})
        rows.append(
            {
                "sample_id": sample_id,
                "query_id": item.get("query_id", query.get("query_id", "")),
                "query": (query.get("query") or "").replace("\n", " "),
                "profile_summary": _profile_summary(profile),
                "audit_status": item.get("audit_status", audit.get("audit_status", "")),
                "audit_reason": (item.get("audit_reason") or audit.get("reason", "")).replace(
                    "\n", " "
                ),
                "validation_errors": item.get("validation_errors", ""),
                "human_decision": "",
                "human_notes": "",
            }
        )

    # The reviewer must not see target labels; guard against accidental schema changes.
    assert not any("target" in key.lower() for key in rows[0] if rows), (
        "target field in queue"
    )

    by_status = Counter(item.get("audit_status", "") for item in queue)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys())
            if rows
            else [
                "sample_id",
                "query_id",
                "query",
                "profile_summary",
                "audit_status",
                "audit_reason",
                "validation_errors",
                "human_decision",
                "human_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Evaluation Dataset {args.dataset_version} — Human Review Queue",
        "",
        f"- Source candidate: `{args.pilot}`",
        f"- Rows: **{len(rows)}**",
        f"- Status mix: {dict(sorted(by_status.items()))}",
        "",
        "## How to fill",
        "",
        "For every row set `human_decision` to one of:",
        "",
        "- `PASS` — query is grounded in the profile, natural, and usable;",
        "- `FAIL` — unsupported claim, contradiction, unusable, or wrong association.",
        "",
        "Add a short `human_notes` reason. Do not look up the target label: this review is about",
        "query quality against the profile only. The reviewer sees no target in this document.",
        "",
        "Return the filled CSV (same `sample_id` values). Once returned, the pipeline will",
        "apply the decisions, report the human-agreement rate, and freeze the final dataset.",
        "",
        "## Summary",
        "",
        "| audit status | rows |",
        "|---|---:|",
        *[f"| {status or 'UNKNOWN'} | {count} |" for status, count in sorted(by_status.items())],
        "",
        "## Rows",
        "",
        "| # | sample_id | query | audit | issue | human_decision | human_notes |",
        "|---:|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(rows, start=1):
        query = _escape(row["query"])[:180]
        reason = _escape(row["audit_reason"])[:180]
        lines.append(
            f"| {index} | {row['sample_id']} | {query} | {row['audit_status']} | {reason} | | |"
        )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} rows to {args.out_csv} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
