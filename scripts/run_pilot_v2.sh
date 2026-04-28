#!/usr/bin/env bash
# Pilot v2 launcher: 6 flagship agents × 20 tasks × 3 conditions × 3 runs.
#
# Output goes to results/pilot_v2/. AGENTOPS_LIVE=0 forces fixture-only mode
# so external tool calls are deterministic.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env.local ]]; then
  set -a; source .env.local; set +a
fi

for var in ANTHROPIC_API_KEY OPENAI_API_KEY TAVILY_API_KEY; do
  if [[ -z "${!var:-}" ]]; then
    echo "$var not set (export it or put it in .env.local)" >&2
    exit 1
  fi
done

mkdir -p results/pilot_v2
LOG=results/pilot_v2.log

source .venv/bin/activate

export AGENTOPS_LIVE=0

agentops-bench run \
  --tasks tasks \
  --agents claude-haiku,claude-sonnet-4-6,claude-opus-4-7,gpt-5.5,gpt-5.4-mini,o4-mini \
  --conditions clean,noisy,adversarial \
  --runs 3 \
  --budget 200 \
  --output results/pilot_v2 \
  2>&1 | tee "$LOG"
