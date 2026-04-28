#!/usr/bin/env bash
# Redo Sonnet + Opus on the existing pilot_v1 grid, overwriting the failed
# JSONs from the first run (which died when the Anthropic credit balance
# ran out partway through Sonnet).
#
# Usage:
#   ANTHROPIC_API_KEY=sk-... bash scripts/redo_anthropic_pilot.sh
#
# Output goes to results/pilot_v1/, replacing the bad anthropic_*_report.json
# and per-run JSONs for those two agents only. Haiku and OpenAI files are
# untouched.
set -euo pipefail

cd "$(dirname "$0")/.."

# Allow the key to come from a local untracked file so the user can run this
# from any shell without re-exporting.
if [[ -f .env.local ]]; then
  set -a; source .env.local; set +a
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY not set (export it, or put it in .env.local)" >&2
  exit 1
fi

LOG=results/pilot_v1_anthropic_redo.log
mkdir -p results

source .venv/bin/activate

agentops-bench run \
  --tasks tasks \
  --agents claude-sonnet-4-6,claude-opus-4-7 \
  --conditions clean,noisy,adversarial \
  --runs 3 \
  --budget 30 \
  --output results/pilot_v1 \
  2>&1 | tee "$LOG"
