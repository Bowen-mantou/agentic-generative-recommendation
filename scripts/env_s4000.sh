#!/usr/bin/env bash
# Server-side environment for the S4000 host. `source scripts/env_s4000.sh`.
#
# - HF_ENDPOINT: optional model mirror.
# - HF_HOME: keep model/cache files outside the repository / small home partition.
# - venv lives on /tmp (recreate with scripts/bootstrap_s4000.sh after a reboot).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/agr-data/hf}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export VENV_DIR="${VENV_DIR:-/tmp/agr/venv}"

if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "warning: venv missing at $VENV_DIR (run scripts/bootstrap_s4000.sh)" >&2
fi
