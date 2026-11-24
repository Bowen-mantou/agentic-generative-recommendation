"""Configuration loading and typed settings.

Precedence (lowest to highest):

1. YAML files in ``configs/`` (base.yaml, models.yaml, retrieval.yaml,
   controller.yaml, memory.yaml), deep-merged with duplicate-key detection.
2. Environment overrides using the ``AGR__`` prefix and ``__`` nesting separator,
   e.g. ``AGR__RETRIEVAL__E5__TOP_K_PER_QUERY=100``.
3. Explicit constructor/test overrides.

Secrets (``DEEPSEEK_API_KEY``, ``MEM0_API_KEY``, ``HF_TOKEN``) are never part of the
YAML config and never serialised into traces. Relative paths are resolved against the
project root, not the current working directory.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from recommender.utils.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_OVERRIDE_PREFIX = "AGR__"
CONFIG_DIR_ENV = "AGR_CONFIG_DIR"
# Optional machine-local overlay (gitignored). Values here override the tracked YAML.
LOCAL_OVERRIDE_FILE = "local.yaml"

# Section key -> file name. Keeping the mapping explicit prevents "magic" config files.
CONFIG_FILES: dict[str, str] = {
    "base": "base.yaml",
    "models": "models.yaml",
    "retrieval": "retrieval.yaml",
    "controller": "controller.yaml",
    "memory": "memory.yaml",
}

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(_StrictModel):
    """Project data, artifact and run directories (relative to the project root)."""

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    samples_dir: Path = Path("data/samples")
    artifacts_dir: Path = Path("artifacts")
    embeddings_dir: Path = Path("artifacts/embeddings")
    faiss_dir: Path = Path("artifacts/faiss")
    semantic_ids_dir: Path = Path("artifacts/semantic_ids")
    catalog_dir: Path = Path("artifacts/catalog")
    checkpoints_dir: Path = Path("artifacts/checkpoints")
    runs_dir: Path = Path("runs")

    def resolved(self, root: Path) -> dict[str, Path]:
        """Return every configured path as an absolute path."""
        return {
            name: (value if value.is_absolute() else (root / value))
            for name, value in self.model_dump().items()
        }


class LoggingConfig(_StrictModel):
    level: LogLevel = "INFO"
    json_output: bool = True
    include_context: bool = True


class ObservabilityConfig(_StrictModel):
    trace_enabled: bool = True
    trace_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class E5ModelConfig(_StrictModel):
    """E5 encoder settings (checkpoint, prefixes, dimension, batching)."""

    model_id: str = Field(min_length=1)
    revision: str = "main"
    query_prefix: str = ""
    passage_prefix: str = ""
    embedding_dim: int = Field(gt=0)
    max_length: int = Field(default=512, gt=0, le=8192)
    batch_size: int = Field(default=64, gt=0)
    device: str = "auto"
    normalize: bool = True


class RerankerConfig(_StrictModel):
    primary_model_id: str = Field(min_length=1)
    baseline_model_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    max_length: int = Field(default=2048, gt=0, le=32768)
    batch_size: int = Field(default=8, gt=0)
    device: str = "auto"


class RQVAEConfig(_StrictModel):
    latent_dim: int = Field(default=32, gt=0)
    hidden_dim: int = Field(default=128, gt=0)
    commitment_weight: float = Field(default=0.25, ge=0.0)
    epochs: int = Field(default=20, gt=0)
    batch_size: int = Field(default=1024, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)


class TigerGeneratorConfig(_StrictModel):
    # Deliberately nullable until Phase 4 selects a compatible backbone; a mock must
    # never be substituted silently (AGENTS.md §1, module 3 §3.5).
    backbone_model_id: str | None = None
    max_length: int = Field(default=16, gt=0)
    beam_size: int = Field(default=20, gt=0)
    top_k: int = Field(default=50, gt=0)
    device: str = "auto"


class TigerConfig(_StrictModel):
    embedding_model_id: str = Field(min_length=1)
    num_semantic_id_levels: int = Field(default=3, ge=1, le=8)
    codebook_size: int = Field(default=256, ge=2)
    rqvae: RQVAEConfig = RQVAEConfig()
    generator: TigerGeneratorConfig = TigerGeneratorConfig()


class DeepSeekConfig(_StrictModel):
    base_url: str = Field(min_length=1)
    planner_model: str = Field(min_length=1)
    judgment_model: str = Field(min_length=1)
    generator_model: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout_s: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_concurrency: int = Field(default=4, ge=1)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")


class GenerationConfig(_StrictModel):
    top_k: int = Field(default=5, ge=1, le=20)


class E5IndexConfig(_StrictModel):
    type: Literal["IndexFlatIP", "IndexFlatL2", "IndexIVFFlat"] = "IndexFlatIP"
    normalize_vectors: bool = True
    metric: Literal["cosine", "inner_product", "l2"] = "cosine"

    @model_validator(mode="after")
    def _check_type_metric_consistency(self) -> E5IndexConfig:
        if self.metric == "cosine" and not self.normalize_vectors:
            raise ValueError("cosine metric requires normalize_vectors=true")
        if self.type == "IndexFlatIP" and self.metric == "l2":
            raise ValueError("IndexFlatIP cannot be paired with l2 metric")
        if self.type == "IndexFlatL2" and self.metric != "l2":
            raise ValueError("IndexFlatL2 requires metric=l2")
        return self


class E5RetrievalConfig(_StrictModel):
    top_k_per_query: int = Field(default=50, gt=0, le=1000)
    max_queries: int = Field(default=5, ge=1, le=10)
    query_max_chars: int = Field(default=300, gt=0)
    rrf_k: int = Field(default=60, ge=1)
    index: E5IndexConfig = E5IndexConfig()


class CandidateBudgetConfig(_StrictModel):
    tiger_candidates: int = Field(default=50, ge=0)
    e5_candidates: int = Field(default=100, ge=0)


class BudgetTiers(_StrictModel):
    standard: CandidateBudgetConfig
    expanded: CandidateBudgetConfig

    @model_validator(mode="after")
    def _expanded_not_smaller(self) -> BudgetTiers:
        for field in ("tiger_candidates", "e5_candidates"):
            if getattr(self.expanded, field) < getattr(self.standard, field):
                raise ValueError(f"expanded.{field} must be >= standard.{field}")
        return self


class FusionConfig(_StrictModel):
    rrf_k: int = Field(default=60, ge=1)
    max_candidates: int = Field(default=300, gt=0)


class RetrievalConfig(_StrictModel):
    e5: E5RetrievalConfig = E5RetrievalConfig()
    budgets: BudgetTiers
    fusion: FusionConfig = FusionConfig()


class ControllerBudgetConfig(_StrictModel):
    remaining_replan_rounds: int = Field(default=2, ge=0)
    remaining_retrieval_calls: int = Field(default=6, ge=0)
    remaining_api_calls: int = Field(default=60, ge=0)


class QualityThresholds(_StrictModel):
    required_count: int = Field(default=5, ge=1)
    min_valid_candidates: int = Field(default=3, ge=0)
    min_mean_rerank_score: float | None = None
    max_unknown_attribute_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    min_diversity: float | None = Field(default=None, ge=0.0, le=1.0)
    # False until the validation split fixes the thresholds; do not cite uncalibrated
    # values as validated results (AGENTS.md §8.8).
    calibrated: bool = False

    @model_validator(mode="after")
    def _min_valid_not_above_required(self) -> QualityThresholds:
        if self.min_valid_candidates > self.required_count:
            raise ValueError("min_valid_candidates must be <= required_count")
        return self


class LoopGuardConfig(_StrictModel):
    max_replan_rounds: int = Field(default=2, ge=0)
    max_rounds: int = Field(default=3, ge=1)
    duplicate_query_similarity: float = Field(default=0.95, ge=0.0, le=1.0)


class ControllerConfig(_StrictModel):
    budget: ControllerBudgetConfig = ControllerBudgetConfig()
    quality: QualityThresholds = QualityThresholds()
    loop_guard: LoopGuardConfig = LoopGuardConfig()


class Mem0VectorStoreConfig(_StrictModel):
    provider: str = "qdrant"
    host: str = "127.0.0.1"
    port: int = Field(default=6333, gt=0, le=65535)
    collection: str = Field(default="agr_user_memory", min_length=1)


class Mem0EmbedderConfig(_StrictModel):
    provider: str = "huggingface"
    model: str = Field(min_length=1)


class Mem0Config(_StrictModel):
    vector_store: Mem0VectorStoreConfig = Mem0VectorStoreConfig()
    embedder: Mem0EmbedderConfig


class SqliteConfig(_StrictModel):
    path: Path = Path("data/processed/interaction_history.db")


class MemoryConfig(_StrictModel):
    enabled: bool = False
    mem0: Mem0Config
    sqlite: SqliteConfig = SqliteConfig()


class AppConfig(_StrictModel):
    """Validated project configuration (everything except secrets)."""

    schema_version: str = "1.0"
    paths: PathsConfig = PathsConfig()
    logging: LoggingConfig = LoggingConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    e5: E5ModelConfig
    reranker: RerankerConfig
    tiger: TigerConfig
    deepseek: DeepSeekConfig
    generation: GenerationConfig = GenerationConfig()
    retrieval: RetrievalConfig
    controller: ControllerConfig = ControllerConfig()
    memory: MemoryConfig
    project_root: Path = Field(default_factory=lambda: PROJECT_ROOT, exclude=True)

    @model_validator(mode="after")
    def _check_cross_section_limits(self) -> AppConfig:
        if self.generation.top_k > max(
            self.controller.quality.required_count,
            self.controller.quality.min_valid_candidates,
        ):
            raise ValueError("generation.top_k must be <= controller.quality.required_count")
        if self.retrieval.budgets.standard.e5_candidates < self.retrieval.e5.top_k_per_query:
            raise ValueError(
                "retrieval.budgets.standard.e5_candidates must be >= e5.top_k_per_query"
            )
        if self.memory.enabled and not self.tiger.generator.backbone_model_id:
            raise ValueError("memory.enabled requires a real tiger generator backbone")
        return self

    def resolve(self, path: Path | str) -> Path:
        """Resolve ``path`` against the project root."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.project_root / candidate)

    def path(self, name: str) -> Path:
        """Resolve one configured path entry by field name (e.g. ``faiss_dir``)."""
        paths = self.paths.resolved(self.project_root)
        if name not in paths:
            raise ConfigError(f"Unknown configured path '{name}'")
        return paths[name]

    def ensure_directories(self) -> None:
        """Create every configured directory (idempotent)."""
        for path in self.paths.resolved(self.project_root).values():
            path.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict[str, Any]:
        """Config snapshot for traces/logs. Contains no secrets."""
        return self.model_dump(mode="json", exclude={"project_root"})


class Secrets(BaseSettings):
    """Secret material, loaded only from environment variables / local ``.env``."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore", case_sensitive=False)

    deepseek_api_key: SecretStr | None = None
    mem0_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None

    @field_validator("deepseek_api_key", "mem0_api_key", "hf_token", mode="before")
    @classmethod
    def _empty_string_is_missing(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value


@dataclass(frozen=True)
class Settings:
    """Loaded configuration + secrets. Do not log this object directly."""

    config: AppConfig
    secrets: Secrets

    def require_deepseek_api_key(self) -> str:
        """Return the DeepSeek key or fail loudly (never degrade to a fake result)."""
        key = self.secrets.deepseek_api_key
        if key is None or not key.get_secret_value().strip():
            raise ConfigError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or export the variable in the environment."
            )
        return key.get_secret_value()

    def has_deepseek_api_key(self) -> bool:
        key = self.secrets.deepseek_api_key
        return key is not None and bool(key.get_secret_value().strip())


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Missing configuration file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - malformed YAML is rare
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}")
    return data


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any], *, origin: str) -> None:
    for key, value in source.items():
        if key in target:
            existing = target[key]
            if isinstance(existing, dict) and isinstance(value, Mapping):
                _deep_merge(existing, value, origin=origin)
                continue
            raise ConfigError(f"Duplicate configuration key '{key}' while merging {origin}")
        target[key] = value


def _deep_override(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Merge ``source`` into ``target`` where source wins (machine-local overlay)."""
    for key, value in source.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _deep_override(existing, value)
        else:
            target[key] = value


def _apply_env_overrides(data: dict[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
    for raw_key in sorted(environ):
        if not raw_key.startswith(ENV_OVERRIDE_PREFIX):
            continue
        dotted = raw_key[len(ENV_OVERRIDE_PREFIX) :].lower()
        parts = dotted.split("__")
        if not dotted or any(not part for part in parts):
            raise ConfigError(f"Malformed environment override: {raw_key}")
        raw_value = environ[raw_key]
        try:
            value: Any = yaml.safe_load(raw_value) if raw_value != "" else ""
        except yaml.YAMLError:
            value = raw_value
        node: dict[str, Any] = data
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(
                    f"Environment override {raw_key} conflicts with non-mapping key '{part}'"
                )
            node = child
        node[parts[-1]] = value
    return data


def find_config_dir(
    config_dir: Path | str | None = None, environ: Mapping[str, str] | None = None
) -> Path:
    """Locate the config directory without touching global state."""
    env = os.environ if environ is None else environ
    if config_dir is not None:
        candidate = Path(config_dir)
    elif env.get(CONFIG_DIR_ENV):
        candidate = Path(env[CONFIG_DIR_ENV])
    else:
        candidate = PROJECT_ROOT / "configs"
    if not candidate.is_dir():
        raise ConfigError(f"Configuration directory not found: {candidate}")
    return candidate


def load_config(
    config_dir: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load, deep-merge, override and validate the project configuration."""
    env = dict(os.environ if environ is None else environ)
    directory = find_config_dir(config_dir, env)
    merged: dict[str, Any] = {}
    for filename in CONFIG_FILES.values():
        _deep_merge(merged, _read_yaml(directory / filename), origin=filename)
    local_override = directory / LOCAL_OVERRIDE_FILE
    if local_override.is_file():
        _deep_override(merged, _read_yaml(local_override))
    merged = _apply_env_overrides(merged, env)
    try:
        return AppConfig(**merged)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc


def load_secrets(
    env_file: Path | str | None = None,
    **overrides: Any,
) -> Secrets:
    """Load secrets from the environment (optionally ``.env``) plus explicit overrides."""
    path = Path(env_file) if env_file is not None else (PROJECT_ROOT / ".env")
    env_file_arg = str(path) if path.is_file() else None
    values: dict[str, Any] = dict(overrides)
    if env_file_arg is not None:
        values["_env_file"] = env_file_arg
    try:
        return Secrets(**values)
    except ValidationError as exc:  # pragma: no cover - SecretStr accepts any string
        raise ConfigError(f"Invalid secrets configuration: {exc}") from exc


def load_settings(
    config_dir: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    env_file: Path | str | None = None,
    **secret_overrides: Any,
) -> Settings:
    """Convenience loader returning validated config and secrets."""
    return Settings(
        config=load_config(config_dir=config_dir, environ=environ),
        secrets=load_secrets(env_file=env_file, **secret_overrides),
    )
