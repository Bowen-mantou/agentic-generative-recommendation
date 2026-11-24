#!/usr/bin/env bash
# Bootstrap the Python environment for Agentic Generative Recommendation on the
# MTT S4000 host. Run *on the server*: bash scripts/bootstrap_s4000.sh
#
# Host facts (2026-11-24):
#   * python3.10.12 exists, but python3-venv/ensurepip is NOT installed -> `python3 -m
#     venv` fails. `uv` (~/.local/bin/uv) creates the venv instead.
#   * /home has ~3.7 GB free; the torch + transformers stack needs ~2.5-3 GB, so the
#     venv lives on /tmp (122 GB root LV). Re-run this script after a reboot.
#   * pypi.org is slow; Tsinghua mirror ~119 MB/s. HuggingFace needs hf-mirror.com.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Agentic-Generative-Recommendation}"
VENV_DIR="${VENV_DIR:-/tmp/agr/venv}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.10 || command -v python3)}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
# The 6 GB uv cache must not live in /home (only ~3-9 GB free, shared with teammates).
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/agr/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp/agr/tmp}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

cd "$PROJECT_DIR"
echo "project: $PROJECT_DIR"
echo "python:  $("$PYTHON_BIN" -V 2>&1)"
echo "uv:      $("$UV_BIN" --version 2>&1)"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  mkdir -p "$(dirname "$VENV_DIR")"
  "$UV_BIN" venv --python "$PYTHON_BIN" "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "installing CPU torch from $TORCH_INDEX_URL"
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
  --index-url "$TORCH_INDEX_URL" "torch==2.6.0"

echo "installing project dependencies from $PIP_INDEX_URL"
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" --index-url "$PIP_INDEX_URL" --quiet \
  "pydantic>=2.9,<3" \
  "pydantic-settings>=2.4,<3" \
  "PyYAML>=6.0.1,<7" \
  "numpy>=1.26,<3" \
  "httpx>=0.27,<1" \
  "tenacity>=8.3,<10" \
  "transformers>=4.44,<5" \
  "sentence-transformers>=3.0,<6" \
  "faiss-cpu>=1.8" \
  "huggingface-hub>=0.24,<2" \
  "tqdm>=4.66,<5" \
  "datasets>=2.20,<5" \
  "pytest>=8.2,<10" \
  "pytest-cov>=5,<8" \
  "ruff>=0.5,<1" \
  "mypy>=1.10,<2" \
  "types-PyYAML>=6.0"

echo "installing the project (editable, no deps)"
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" --no-deps -e . --quiet

echo "installing the CLI entry point shim"
if [ ! -x "$VENV_DIR/bin/agr" ]; then
  cat > "$VENV_DIR/bin/agr" <<EOF
#!/usr/bin/env bash
exec "$VENV_DIR/bin/python" -m recommender.cli "\$@"
EOF
  chmod +x "$VENV_DIR/bin/agr"
fi

echo "verifying imports and versions"
python - <<'PY'
import faiss, numpy, pydantic, sentence_transformers, torch, transformers
print("python", __import__("sys").version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("sentence-transformers", sentence_transformers.__version__)
print("faiss", faiss.__version__)
print("numpy", numpy.__version__)
print("pydantic", pydantic.VERSION)
PY

echo "running unit tests"
python -m pytest -q

echo "bootstrap complete"
