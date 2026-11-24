# Security policy

This project is a research prototype and is not hardened for production.

- Do not deploy it on a public network without adding authentication, TLS and rate limiting.
- API keys are read from environment variables / `.env` (gitignored); never commit them.
- Datasets, model weights and indexes are not part of the repository.
- Qdrant, model weights and any LLM API are optional external services; secure them separately.

To report a vulnerability, open a private security advisory on GitHub instead of a public issue.
Please avoid including secrets or personal data in reports.
