#!/usr/bin/env bash
# Pilot v1 seeded launcher: 8 frontier-class agents x 100 tasks x 3 conditions x 3 runs.
#
# Output goes to results/pilot_v1_seeded/<agent>/. AGENTOPS_LIVE=0 forces
# fixture-only mode so external tool calls are deterministic; per-cell RNG
# seeding is sha256(task_id|condition|run_number), set by the runner.
#
# The eight agents matching the v1.0 paper are:
#   - anthropic/claude-opus-4-7               (closed-weights)
#   - anthropic/claude-sonnet-4-6             (closed-weights)
#   - anthropic/claude-haiku-4-5-20251001     (closed-weights)
#   - openai/gpt-5.5                          (closed-weights)
#   - openrouter/meta-llama/llama-4-scout     (open-weights)
#   - openrouter/qwen/qwen3-max               (open-weights)
#   - openrouter/deepseek/deepseek-v3.2       (open-weights)
#   - openrouter/mistralai/mistral-large-2512 (open-weights)
#
# This script launches all eight agents in parallel as background processes
# so total wall-clock is bounded by the slowest agent (deepseek ~8 hours).
# After it completes, run scripts/combine_reports.py to produce the
# _combined/ rollup that analyze_pilot.py and make_figures.py consume.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env.local ]]; then
  set -a; source .env.local; set +a
fi

for var in ANTHROPIC_API_KEY OPENAI_API_KEY OPENROUTER_API_KEY TAVILY_API_KEY; do
  if [[ -z "${!var:-}" ]]; then
    echo "$var not set (export it or put it in .env.local)" >&2
    exit 1
  fi
done

mkdir -p results/pilot_v1_seeded
source .venv/bin/activate
export AGENTOPS_LIVE=0

AGENTS=(
  "claude-opus-4-7"
  "claude-sonnet-4-6"
  "claude-haiku-4-5-20251001"
  "gpt-5.5"
  "llama-4-scout"
  "qwen-3-max"
  "deepseek-v3-2"
  "mistral-large-2512"
)

PIDS=()
for agent in "${AGENTS[@]}"; do
  slug=$(echo "$agent" | tr '/' '_')
  outdir="results/pilot_v1_seeded/${slug}"
  mkdir -p "$outdir"
  log="results/pilot_v1_seeded/${slug}.log"
  echo "[$(date)] launching $agent -> $outdir (log: $log)"
  agentops-bench run \
    --tasks tasks \
    --agents "$agent" \
    --conditions clean,noisy,adversarial \
    --runs 3 \
    --budget 200 \
    --output "$outdir" \
    > "$log" 2>&1 &
  PIDS+=("$!")
done

echo "[$(date)] all 8 agents launched; pids: ${PIDS[*]}"
echo "tail any of the per-agent logs to follow progress:"
for agent in "${AGENTS[@]}"; do
  slug=$(echo "$agent" | tr '/' '_')
  echo "  tail -f results/pilot_v1_seeded/${slug}.log"
done

wait "${PIDS[@]}"
echo "[$(date)] all 8 agents finished."
echo "next steps:"
echo "  python3 scripts/combine_reports.py results/pilot_v1_seeded"
echo "  python3 scripts/analyze_pilot.py    results/pilot_v1_seeded/_combined"
echo "  python3 scripts/make_figures.py     results/pilot_v1_seeded/_combined"
