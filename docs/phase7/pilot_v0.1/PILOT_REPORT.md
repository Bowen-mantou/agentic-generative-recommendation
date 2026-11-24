# Evaluation Dataset Pilot Report

- dataset_version: `v0.1-pilot`
- raw test samples: 2685
- recovered: 2685
- pilot samples: 100
- query status: {'insufficient_profile': 3, 'generated': 97}
- audit status: {'ERROR': 3, 'REVIEW': 14, 'PASS': 82, 'FAIL': 1}
- validation passed: 97
- PASS & validation-pass (unreviewed): 82
- human review pending: 20
- leakage violations: 0

## Honest status

This is a pilot. REVIEW/FAIL/ERROR and the sampled PASS rows await human review; the dataset is **not frozen as a final test set** until that review is completed.

## Target isolation

Profiles/queries were built without reading target labels; the validator scanned for exact target product IDs and full target titles.
