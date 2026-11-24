# Contributing

Thanks for taking a look. This repository is a research prototype; small, well-tested
changes are preferred over large rewrites.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,workflow]"
pytest tests/unit -q
ruff check src tests
mypy
```

E5/FAISS, reranker and real API integration tests are opt-in and require model weights,
an index, or `DEEPSEEK_API_KEY` (see `.env.example`). They are skipped by default.

## Pull requests

- Keep the Pydantic contracts in `src/recommender/schemas/` as the source of truth.
- Add or update unit tests; state which tests you actually ran.
- Do not commit datasets, model weights, indexes, `.env`, API keys or private infrastructure
  details.
- Distinguish real results from mocks and unverified assumptions in docs.
- Mark synthetic data with `is_synthetic=true`; never mix it into real evaluation metrics.

## Reporting issues

Include the commit, Python version, command, expected vs. actual behaviour and whether any
external model/API was involved. Do not paste API keys, passwords or private hostnames.
