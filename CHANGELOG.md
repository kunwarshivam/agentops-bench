# Changelog

All notable changes to AgentOps-Bench are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
under the MAJOR.MINOR.PATCH policy described in §F of the paper.

## [1.0.0] — 2026-04-27

Initial public release accompanying the preprint.

### Added

- Six-axis scoring suite (completion, cost, efficiency, reliability,
  recovery, safety).
- Instrumented tool server with five failure-mode injectors and a
  15-entry indirect prompt-injection catalogue (12 attack classes).
- 20 seed tasks across `code/`, `data_analysis/`, `research/`,
  `safety/`, and `tool_use/` domains.
- Snapshotted fixtures for Open-Meteo, yfinance, and Tavily replay.
- Anthropic and OpenAI agent adapters.
- 1,080-run pilot dataset under `results/pilot_v2/` (six agents ×
  20 tasks × 3 conditions × 3 runs).
- Reproduction scripts: `scripts/run_pilot_v2.sh`,
  `analyze_pilot.py`, `analyze_pairwise.py`,
  `analyze_safety_per_payload.py`, `make_figures.py`.
- Datasheet, broader-impact statement, maintenance plan, and
  reproducibility checklist in the paper appendix.

### Fixed

- Failure / prompt-injection RNG is now seeded deterministically per
  `(task, condition, run_number)` triple via SHA-256, replacing the
  earlier unseeded global RNG. The pilot in this release was produced
  before the fix; reruns from this commit forward are bit-for-bit
  reproducible at the harness layer.
