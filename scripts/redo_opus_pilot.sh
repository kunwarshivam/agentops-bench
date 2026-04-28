#!/usr/bin/env bash
# Redo Opus only on the pilot_v1 grid, after the temperature-deprecation fix.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env.local ]]; then
  set -a; source .env.local; set +a
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY not set" >&2
  exit 1
fi

LOG=results/pilot_v1_opus_redo.log
mkdir -p results

source .venv/bin/activate

agentops-bench run \
  --tasks tasks \
  --agents claude-opus-4-7 \
  --conditions clean,noisy,adversarial \
  --runs 3 \
  --budget 30 \
  --output results/pilot_v1 \
  2>&1 | tee "$LOG"
