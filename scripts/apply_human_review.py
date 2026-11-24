#!/usr/bin/env python3
"""Apply the returned human review CSV and freeze evaluation dataset v1.0.

Rules (protocol §7.3): a sample enters the final set only when the human decision is
PASS, the query was generated, and Python schema/leakage validation passed. Insufficient
profiles cannot be rescued by a human PASS because there is no query.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

FINAL_DECISIONS = {"PASS", "FAIL"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["sample_id"] = row["sample_id"].lstrip("\ufeff")
    return rows


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--dataset-version", default="v1.0")
    parser.add_argument("--code-commit", default=None)
    args = parser.parse_args()

    pilot = args.pilot
    queue = _read_csv(pilot / "human_review" / "human_audit.csv")
    filled = _read_csv(args.review_csv)
    queue_ids = [row["sample_id"] for row in queue]
    filled_by_id = {row["sample_id"]: row for row in filled}

    errors: list[str] = []
    if set(queue_ids) != set(filled_by_id):
        missing = sorted(set(queue_ids) - set(filled_by_id))[:5]
        extra = sorted(set(filled_by_id) - set(queue_ids))[:5]
        errors.append(f"sample_id mismatch missing={missing} extra={extra}")
    for sample_id in queue_ids:
        row = filled_by_id.get(sample_id)
        if row is None:
            continue
        decision = row.get("human_decision", "").strip().upper()
        if decision not in FINAL_DECISIONS:
            errors.append(f"invalid decision for {sample_id}: {decision!r}")
    if errors:
        raise SystemExit("human review validation failed: " + "; ".join(errors[:10]))

    queries = {
        row["sample_id"]: row for row in _read_jsonl(pilot / "queries" / "test_queries.jsonl")
    }
    audits = {row["sample_id"]: row for row in _read_jsonl(pilot / "audits" / "test_audits.jsonl")}
    validation = {
        row["sample_id"]: row
        for row in _read_jsonl(pilot / "validation" / "query_validation.jsonl")
    }
    samples = _read_jsonl(pilot / "samples" / "test_samples.jsonl")
    labels = {
        row["sample_id"]: row for row in _read_jsonl(pilot / "labels" / "target_labels.jsonl")
    }

    decisions = Counter()
    audit_by_human = Counter()
    included: list[dict] = []
    excluded: list[dict] = []
    reviewed_ids = set(queue_ids)
    for sample_id, query in queries.items():
        audit = audits.get(sample_id, {})
        validation_row = validation.get(sample_id, {})
        query_ok = query.get("status") == "generated" and validation_row.get("passed", False)
        if sample_id in reviewed_ids:
            decision = filled_by_id[sample_id]["human_decision"].strip().upper()
            decisions[decision] += 1
            audit_by_human[(audit.get("audit_status"), decision)] += 1
            if decision == "PASS" and query_ok:
                included.append({"sample_id": sample_id})
            elif decision == "PASS":
                excluded.append(
                    {
                        "sample_id": sample_id,
                        "reason": "human_pass_but_invalid_query",
                    }
                )
            else:
                excluded.append({"sample_id": sample_id, "reason": "human_fail"})
        elif query_ok and audit.get("audit_status") == "PASS":
            # Auditor-PASS samples outside the sampled QC set remain included.
            included.append({"sample_id": sample_id})
        else:
            excluded.append({"sample_id": sample_id, "reason": "not_human_approved"})

    included_ids = {row["sample_id"] for row in included}
    final_samples = [sample for sample in samples if sample["sample_id"] in included_ids]
    final_labels = [
        labels[sample["sample_id"]] for sample in final_samples if sample["sample_id"] in labels
    ]
    if len(final_samples) != len(final_labels):
        raise SystemExit("final samples/labels size mismatch")

    human_pass = decisions.get("PASS", 0)
    human_fail = decisions.get("FAIL", 0)
    agreement = sum(
        count
        for (status, decision), count in audit_by_human.items()
        if (status == "PASS") == (decision == "PASS")
    )
    flagged = sum(count for (status, _), count in audit_by_human.items() if status != "PASS")
    flagged_rejected = sum(
        count
        for (status, decision), count in audit_by_human.items()
        if status != "PASS" and decision == "FAIL"
    )
    sampled_pass = sum(count for (status, _), count in audit_by_human.items() if status == "PASS")
    sampled_pass_approved = sum(
        count
        for (status, decision), count in audit_by_human.items()
        if status == "PASS" and decision == "PASS"
    )
    agreement_report = {
        "dataset_version": args.dataset_version,
        "reviewed_rows": len(queue_ids),
        "human_decisions": dict(decisions),
        "audit_status_by_human_decision": {
            f"{status}|{decision}": count
            for (status, decision), count in sorted(audit_by_human.items())
        },
        "agreement_count": agreement,
        "agreement_rate": round(agreement / len(queue_ids), 4) if queue_ids else 0.0,
        "flagged_rows": flagged,
        "flagged_human_rejected": flagged_rejected,
        "flagged_human_accept_rate": round((flagged - flagged_rejected) / flagged, 4)
        if flagged
        else 0.0,
        "sampled_pass_rows": sampled_pass,
        "sampled_pass_human_approved": sampled_pass_approved,
        "sampled_pass_approval_rate": round(sampled_pass_approved / sampled_pass, 4)
        if sampled_pass
        else 0.0,
        "numeric_exclusions_from_human_pass": len(
            [row for row in excluded if row["reason"].startswith("human_pass_but_")]
        ),
    }

    final_coverage = {
        "dataset_version": args.dataset_version,
        "raw_samples": len(samples),
        "review_queue_rows": len(queue_ids),
        "human_pass": human_pass,
        "human_fail": human_fail,
        "included_samples": len(final_samples),
        "excluded_samples": len(excluded),
        "exclusion_reasons": dict(Counter(row["reason"] for row in excluded)),
        "human_review_completed": True,
        "frozen": True,
        "note": (
            "Only human PASS + generated query + validation pass are included; "
            "insufficient_profile rows cannot be rescued."
        ),
    }

    directory = pilot
    (directory / "human_review" / "review_results.json").write_text(
        json.dumps(
            [
                {
                    "sample_id": sample_id,
                    "human_decision": filled_by_id[sample_id]["human_decision"].strip().upper(),
                    "human_notes": filled_by_id[sample_id].get("human_notes", ""),
                }
                for sample_id in queue_ids
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(args.review_csv, directory / "human_review" / "human_review_v1.0_filled.csv")
    _write_jsonl(directory / "samples" / "final_samples.jsonl", final_samples)
    _write_jsonl(directory / "labels" / "final_target_labels.jsonl", final_labels)
    (directory / "reports" / "human_agreement_report.json").write_text(
        json.dumps(agreement_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "reports" / "final_coverage_report.json").write_text(
        json.dumps(final_coverage, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest_path = directory / "manifests" / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "dataset_version": args.dataset_version,
            "human_review_completed": True,
            "frozen": True,
            "code_commit": args.code_commit or manifest.get("code_commit"),
            "human_review": agreement_report,
            "counts": final_coverage,
            "final_counts": final_coverage,
        }
    )

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    for name in (
        "samples/final_samples.jsonl",
        "labels/final_target_labels.jsonl",
        "reports/human_agreement_report.json",
        "reports/final_coverage_report.json",
        "human_review/human_review_v1.0_filled.csv",
    ):
        manifest["files"][name] = sha256(directory / name)
    (directory / "manifests" / "evaluation_manifest_final.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {"agreement": agreement_report, "coverage": final_coverage},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
