# Agentic Generative Recommendation

[![CI](https://github.com/Bowen-mantou/agentic-generative-recommendation/actions/workflows/ci.yml/badge.svg)](https://github.com/Bowen-mantou/agentic-generative-recommendation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

An experimental, auditable **agentic generative recommender** for shopping requests:

```
User request (+ Mem0 memory)
      │
      ▼
Planner / Replanner (DeepSeek) ──► RecommendationPlan
      │
      ├──► TIGER: RQ-VAE Semantic IDs + autoregressive generation
      └──► E5 + FAISS: multi-query semantic retrieval
                 │
                 ▼
        Candidate Fusion (two-stage RRF)
                 │
                 ▼
   Python hard-constraint evaluator (pass / fail / unknown)
                 │
                 ▼
        Qwen3-Reranker-0.6B (frozen, CPU baseline)
                 │
                 ▼
     Per-product Judgment (DeepSeek, structured)
                 │
                 ▼
   Adaptive Controller (LangGraph + Python): REPLAN / STOP / CLARIFY
                 │
                 ▼
   Grounded Generator (DeepSeek)  ──►  Feedback
                                          ├──► Mem0 (semantic memory)
                                          └──► SQLite (InteractionEvent)
```

This is a **research prototype**, not a production service. Offline evaluation uses
Amazon Reviews 2023 review/rating proxy events; results are **not** purchase or conversion
metrics. The repository documents negative results as well as positive ones — the TIGER
pilot did **not** pass its quality gate and is marked **No-Go** instead of being silently
replaced with a mock.

> **Live status snapshot:** [docs/STATUS.md](docs/STATUS.md) (updated 2026-11-24 23:46 Asia/Shanghai).

## Current status (2026-11-24)

| Area | Status | Notes |
|---|---|---|
| Phase 0 — project skeleton, config, logging, trace | ✅ done | Pydantic contracts, observability |
| Phase 1 — catalog + E5 + FAISS | ✅ done | 112,590 All_Beauty products, 768-dim E5, CPU index, manifests |
| Phase 2 — fusion, constraints, candidate pool | ✅ done | Two-stage RRF, three-state constraint evaluator |
| Phase 3 — Qwen3-Reranker-0.6B | ✅ done | Transformers-native CPU baseline, frozen |
| Phase 4 — TIGER | ⛔ **No-Go** | Real RQ-VAE + autoregressive generator trained; Recall@10 0.15%/0.25%/0.07% (A1/B1/B2) vs popularity 0.34–0.82% and E5-profile 2.9–4.4%. No full training. [Report](docs/phase4/D1_B1_B2_GATE_REPORT.md) |
| Phase 5 — Planner / Judgment / Controller | ✅ implemented + hardened | DeepSeek client, Planner/Replanner, per-product Judgment, LangGraph workflow; real end-to-end CLI wiring still pending |
| Phase 6 — Generator / Memory | 🟡 partial | Qdrant + multilingual E5 deployment validated; Generator and Mem0 adapter not implemented |
| Phase 7 — evaluation | ✅ v1.0 frozen | 2,685 temporal samples recovered; 2,551 queries; 528 human-reviewed (67 PASS / 461 FAIL); **2,224 final samples frozen**; 0 leakage. [Final report](docs/phase7/v1.0/final_coverage_report.json) |

Test snapshot: **260 passed / 14 skipped** locally; **271 passed / 3 skipped** on the
CPU server (LangGraph integration tests included). Unit tests run without GPUs,
model weights, API keys or datasets.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"

pytest tests/unit -q               # no external services required
ruff check src tests
mypy
agr config validate
```

Optional retrieval stack (E5 + FAISS; installs torch/transformers):

```bash
python -m pip install -e ".[retrieval]"
```

Copy [`.env.example`](.env.example) to `.env` for `DEEPSEEK_API_KEY` and the HF mirror.
`.env` is gitignored; never commit keys.

### Amazon Reviews 2023 data

The raw dataset is **not** included (it is large and subject to its own terms). Download it
from the official Amazon Reviews 2023 distribution, then build the catalog and index:

```bash
agr catalog download --category All_Beauty
agr catalog build --meta data/raw/meta_All_Beauty.jsonl --category All_Beauty
agr index build --catalog artifacts/catalog/all_beauty_v2
agr search --bundle artifacts/faiss/e5-base-v2_all_beauty_v2 \
  --query "lightweight daily moisturizer" --query "fragrance-free face cream" --top-k 5
```

See [docs/audit/phase1-report.md](docs/audit/phase1-report.md) for exact sizes, checksums and
reproduction commands. Original data, model weights, indexes and caches are excluded from Git.

## Repository layout

```text
src/recommender/
  schemas/        Pydantic contracts (source of truth)
  catalog/        Amazon metadata, ProductRecord, text templates
  retrieval/e5/   E5 encoder, FAISS index, multi-query RRF
  retrieval/tiger/RQ-VAE, Semantic IDs, time-split sequences, AR generator
  fusion/         two-stage RRF and candidate pool
  constraints/    single Python hard-constraint evaluator
  reranking/      Qwen3-Reranker-0.6B (Transformers native)
  api/ planner/ judgment/ controller/  Phase 5
  generation/ memory/                  Phase 6 placeholders
  evaluation/     temporal samples, profiles, query generator/auditor, validation, pilot
  interaction_history/ SQLite event store and review→event ETL
  observability/  JSON logs, request/trace context, JSONL trace
configs/          validated YAML (local overrides gitignored)
docs/             design docs, audit reports, decisions, risk register
scripts/          data/index/deployment/evaluation helpers
tests/unit/       fast tests, no external services
tests/integration/ model/index/service tests (opt-in; skipped by default)
```

## Documentation

- Design baseline and engineering rules: [AGENTS.md](AGENTS.md), [STARTUP_PROMPT.md](STARTUP_PROMPT.md)
- Phase reports: [Phase 1](docs/audit/phase1-report.md), [Phase 2](docs/audit/phase2-report.md), [Phase 3](docs/audit/phase3-report.md)
- TIGER: [implementation plan](docs/phase4/TIGER_IMPLEMENTATION_PLAN.md), [pilot](docs/phase4/TIGER_PILOT_REPORT.md), [D11 gate report](docs/phase4/D1_B1_B2_GATE_REPORT.md)
- Phase 5: [Planner / Judgment / Controller](docs/phase5/PLANNER_JUDGMENT_CONTROLLER.md)
- Phase 6: [Qdrant + multilingual E5 deployment](docs/phase6/D9_MEMORY_DEPLOYMENT.md)
- Phase 7: [dataset protocol](docs/phase7/EVALUATION_DATASET_CONSTRUCTION_PROTOCOL_V1.0.md), [pilot report](docs/phase7/PILOT_REPORT_V0.1.md), [human-query protocol](docs/phase7/EVALUATION_PROTOCOL_D10.md)
- Decisions and risk register: [docs/decisions/FINAL_DECISIONS_AND_RISK_REGISTER.md](docs/decisions/FINAL_DECISIONS_AND_RISK_REGISTER.md)

## What is real vs. not yet verified

**Real and reproducible**

- E5/FAISS index built from 112,590 real All_Beauty products; real retrieval latency and manifests.
- Qwen3-Reranker-0.6B loaded through Transformers on CPU, verified yes/no scoring.
- DeepSeek Planner and per-product Judgment smoke tests with real API calls (JSON mode).
- TIGER RQ-VAE and generator trained on real event sequences; evaluation includes popularity and
  E5-profile baselines and paired bootstrap confidence intervals.

**Not yet verified / intentionally incomplete**

- End-to-end real CLI (`Planner → E5 → pool → reranker → Judgment → LangGraph controller`) is not wired.
- Phase 6 Generator and Mem0 adapter are not implemented; `Memory.add` is untested.
- Controller thresholds are uncalibrated (`calibrated: false`); the Phase 7 v1.0 evaluation set is frozen (2,224 samples), but full Reranker comparison and end-to-end service wiring are still pending.
- TIGER is No-Go at current data scale and is not used as the primary recall path.

## Limitations

- Amazon All_Beauty metadata is sparse for price/description; hard constraints frequently return
  `unknown` and are never treated as satisfied.
- TIGER pilot used 8,201 users with ≥3 events (All_Beauty only); results do not generalize to
  larger catalogs without re-running the gate.
- CPU-only baseline; GPU/MUSA acceleration is not part of this release.
- LLM outputs are schema-validated, but structural validity is not the same as semantic correctness.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please do not include datasets, weights, indexes, API keys
or private infrastructure details in issues or pull requests.

## License

[MIT](LICENSE) © 2026 Agentic Generative Recommendation contributors.

## Acknowledgements

Amazon Reviews 2023 (data), `intfloat/e5-base-v2`, Qwen3-Reranker-0.6B, DeepSeek API,
Mem0, Qdrant, LangGraph, FAISS. The TIGER implementation was informed by the public TIGER and
RQ-VAE literature; the Cornac audit is documented in [docs/phase4/TIGER_IMPLEMENTATION_PLAN.md](docs/phase4/TIGER_IMPLEMENTATION_PLAN.md).
